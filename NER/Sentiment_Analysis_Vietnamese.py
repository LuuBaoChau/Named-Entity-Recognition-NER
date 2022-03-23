import underthesea
from underthesea import pos_tag
pos_tag('COVID-19 đã ảnh hưởng rất nhiều đến nền kinh tế thế giới')

from underthesea import ner
text = 'Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump'
ner(text)
pos_tag('Thousand of demonstrators have marched through London to protest the')

from underthesea import sentiment
sentiment('món ăn này rất ngon, đáng quay lại')
sentiment('thật là giỏi')
sentiment('Tôi thất vọng về dịch vụ tại đây')
sentiment('Xem lại vẫn thấy xúc động và tự hào về BIDV của mình', domain='bank')

from underthesea import sent_tokenize
a = ('Một khách sạn tuyệt vời, tôi chưa bao giờ nghĩ đến một khách sạn tuyệt vời như vậy ở Việt Nam, hy vọng tôi sẽ quay lại đây và tìm thấy một kỳ nghỉ tuyệt vời ở đây một lần nữa. ở đây ngay trung tâm thành phố gầnĐại dịch COVID-19, còn được gọi là đại dịch coronavirus, là một đại dịch bệnh truyền nhiễm với tác nhân là virus SARS-CoV-2, đang diễn ra trên phạm vi toàn cầu. bến xe và chợ Vũng Tàu tôi có thể tìm thấy mọi thứ ở đây là một nơi tốt để đi du lịch ')
sent_tokenize(a)
sentiment(a)

!pip install underthesea
from underthesea import word_tokenize
sentence = 'Đại dịch COVID-19, còn được gọi là đại dịch coronavirus, là một đại dịch bệnh truyền nhiễm với tác nhân là virus SARS-CoV-2, đang diễn ra trên phạm vi toàn cầu.'
word_tokenize(sentence)