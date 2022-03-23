import os
import argparse
import random
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from modules.model import *
from commons import NERdataset, logger, init_logger
from processor import NERProcessor
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_bert import BertTokenizer
from sklearn.metrics import classification_report, f1_score


def build_dataset(args, processor, data_type='train', feature=None, device=torch.device('cpu')):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_example(data_type, feature is not None)

        features = processor.convert_examples_to_features(examples, args.max_seq_length, feature)
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    return NERdataset(features, device)


def caculator_metric(preds, golds, labels):
    pred_iob_labels = [labels[label_id - 1] for label_id in preds]
    gold_iob_labels = [labels[label_id - 1] for label_id in golds]

    pred_labels = [labels[label_id - 1].split("-")[-1].strip() for label_id in preds]
    gold_labels = [labels[label_id - 1].split("-")[-1].strip() for label_id in golds]

    iob_metric = classification_report(pred_iob_labels, gold_iob_labels, output_dict=True)
    metric = classification_report(pred_labels, gold_labels, output_dict=True)

    return iob_metric, metric


def update_model_weights(model, iterator, optimizer, scheduler):
    # init static variables
    tr_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(iterator, desc="Iteration")):
        tokens, token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks, feats = batch
        loss, _ = model.calculate_loss(token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks,
                                       feats)
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
    return tr_loss


def evaluate(model, iterator, label_map):
    # init static variables
    preds = []
    golds = []
    eval_loss = 0
    model.eval()

    for step, batch in enumerate(tqdm(iterator, desc="Iteration")):
        tokens, token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks, feats = batch
        loss, (logits, labels) = model.calculate_loss(token_ids, attention_masks, token_mask, segment_ids, label_ids,
                                                      label_masks, feats)
        eval_loss += loss.item()
        logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
        pred = logits.detach().cpu().numpy()
        gold = labels.to('cpu').numpy()
        preds.extend(pred)
        golds.extend(gold)

    iob_metric, metric = caculator_metric(preds, golds, label_map)

    return eval_loss, iob_metric, metric


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    summary_writer = SummaryWriter(args.log_dir)
    init_logger(f"{args.output_dir}/vner_trainning.log")

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    processor = NERProcessor(args.data_dir, tokenizer)
    num_labels = processor.get_num_labels()
    logger.info("Build model ...")
    config, model, feature = model_builder(model_name_or_path=args.model_name_or_path,
                                           num_labels=num_labels,
                                           feat_config_path=args.feat_config,
                                           one_hot_embed=args.one_hot_emb,
                                           use_lstm=args.use_lstm,
                                           device=device)
    model.to(device)
    logger.info("Prepare dataset ...")
    train_data = build_dataset(args, processor, data_type='train', feature=feature, device=device)
    eval_data = build_dataset(args, processor, data_type='test', feature=feature, device=device)

    train_sampler = RandomSampler(train_data)
    train_iterator = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_sampler = RandomSampler(eval_data)
    eval_iterator = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(train_iterator) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)