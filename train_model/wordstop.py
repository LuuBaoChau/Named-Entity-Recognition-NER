# Import các thư viện cần thiết
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
import underthesea # Thư viện tách từ
from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu

from transformers import AutoModel, AutoTokenizer # Thư viện BERT

# Thư viện train SVM
from sklearn.svm import SVC
from joblib import dump
# Hàm load model BERT
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw