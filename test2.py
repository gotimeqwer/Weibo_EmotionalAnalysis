from bs4 import BeautifulSoup
import requests
from paddlehub.text.tokenizer import CustomTokenizer
from paddlenlp.transformers import BertTokenizer
from paddlenlp.data import JiebaTokenizer
import paddlenlp.data.tokenizer as tk
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset
from typing import Dict, List, Optional, Union, Tuple
from time import time
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from functools import partial
import numpy as np
import pandas as pd
import csv
import xlrd
import sys
import interface
import paddlehub as hub
import json
from tqdm import trange
from harvesttext import HarvestText
import pyhanlp
import re
import os


def clean_text(file, save_dir):
    ht = HarvestText()  # 可调用HarvestText库的功能接口
    CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('%s -> data over' % file)
    num_null = 0
    cleaned_data = []
    for i in trange(len(data)):
        content = CharTable.convert(data[i]['content'])
        cleaned_content = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                                 '', ht.clean_text(content, emoji=False), flags=re.MULTILINE)  # 过滤@后最多6个字符
        num_null += 1 if cleaned_content == '' else 0
        # 删除train中的自带的空数据或清洗后出现的空数据
        if 'train' in file and (not content or not cleaned_content):
            continue
        if 'eval' in file or 'test' in file:
            cleaned_data.append(
                {'id': data[i]['id'], 'content': cleaned_content})
        else:
            cleaned_data.append(
                {'id': data[i]['id'], 'content': cleaned_content, 'label': data[i]['label']})
    filename = file.split('/')[-1]
    save_json(cleaned_data, os.path.join(save_dir, filename))
    print('num data: ', num_null)


clean_text('C:/Users/23106/Desktop/train/virus_test.txt',
           'C:/Users/23106/Desktop/train/clean')
clean_text('C:/Users/23106/Desktop/trainusual_test.txt',
           'C:/Users/23106/Desktop/train/clean')


data.loc[data['label'] == 1, 'label'] = '愤怒'
data.loc[data['label'] == 2, 'label'] = '积极'
data.loc[data['label'] == 3, 'label'] = '悲伤'
data.loc[data['label'] == 4, 'label'] = '恐惧'
data.loc[data['label'] == 5, 'label'] = '惊奇'
data.loc[data['label'] == 6, 'label'] = '无情绪'


# 单条文本情感分类

def Single_classification(ui):
    content = ui.textEdit.toPlainText()  # 获取输入的要进行情感分类的文本
    # 要进行情感分类的文本内容不能为空
    if content == '':
        ui.label_3.setVisible(False)     # 隐藏结果
        ui.lineEdit_5.setVisible(False)
        ui.warn1()   # 提示补全文本内容
    else:
        # 格式处理：
        data = []
        list = []
        list.append(content)
        data.append(list)
        t1 = time()
        # 对单条文本进行预测
        # 若下载了GPU的paddle，可以将此处use_gpu设置为True
        label = model.predict(data, max_seq_len=128,
                              batch_size=16, use_gpu=False)
        t2 = time()
        # 单条预测时间检测
        print('单条文本分类CPU环境下预测耗时（毫秒）：%.3f' % ((t2 - t1) * 1000.0))
        ui.lineEdit_5.setText(label[0])   # 完成预测后在界面显示预测的情感类别
        ui.label_3.setVisible(True)
        ui.lineEdit_5.setVisible(True)

# 批量文本情感分类


def Batch_classification(ui):
    excel_path = ui.lineEdit_2.text()   # 获取输入文件路径
    output_path = ui.lineEdit_4.text()  # 获取输出文件路径
    # 路径不能为空
    if excel_path == '':
        ui.warn2()  # 提示未选择要进行批量情感分类的excel文件
    elif output_path == '':
        ui.warn3()  # 提示未选择生成结果文件输出路径
    else:
        df = pd.read_excel(excel_path)
        # 格式处理：
        news = pd.DataFrame(columns=['content'])
        news['content'] = df["content"]
        # 首先将数据转化为数组
        data_array = np.array(news)
        # 然后转化为list形式
        data_list = data_array.tolist()

        # 批量文本预测
        # 若下载了GPU的paddle，可以将此处use_gpu设置为True
        results = model.predict(
            data_list, max_seq_len=128, batch_size=16, use_gpu=False)

        df['label'] = results  # 将结果填充到label列上
        # 保存结果文件为excel文件
        df.to_excel(output_path, sheet_name='预测结果', index=False, header=True)
        # ui.cancelloading() # 完成预测后取消显示加载中
        ui.success()  # 提示分类完成


if __name__ == '__main__':

    # 定义要进行情感分类的7个类别
    label_list = ['难过', '愉快', '喜欢', '愤怒', '害怕', '惊讶', '厌恶']
    label_map = {
        idx: label_text for idx, label_text in enumerate(label_list)
    }

    # 加载训练好的模型
    model = hub.Module(
        name='ernie_tiny',
        version='2.0.2',  # 与训练时统一好，若未指定版本将自动下载最新的版本
        task='seq-cls',
        num_classes=7,
        load_checkpoint='../Ernie-model/model.pdparams',   # 注意模型参数一定要加载对！
        label_map=label_map
    )

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = interface.Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()

    # 为按钮绑定相关功能函数完成功能添加：
    # 单条文本情感分类
    ui.pushButton.clicked.connect(partial(Single_classification, ui))
    # 批量文本情感分类
    ui.pushButton_4.clicked.connect(partial(Batch_classification, ui))

    sys.exit(app.exec_())


# 定义数据清洗函数
def clean_data(old_path, save_path):
    data = defaultdict(list)
    filename = os.path.basename(old_path)
    with open(old_path, 'r', encoding='utf8') as f:
        texts = f.readlines()
        for line in tqdm(texts, desc=old_path):
            label, text = line.strip().split(',')
            left_square_brackets_pat = re.compile(r'\[+')
            right_square_brackets_pat = re.compile(r'\]+')
            punct = [',', '\\.', '\\!', '，', '。', '！', '、', '\?', '？']
            text = left_square_brackets_pat.sub('', text)
            text = right_square_brackets_pat.sub('', text)
            for p in punct:
                pattern = char + '{2,}'
                if char.startswith('\\'):
                    char = char[1:]
                    text = re.sub(pattern, char, string)
            data['label'].append(label)
            data['text'].append(text)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, filename), index=False,
              encoding='utf8', header=False, sep=',')
    return df


# 定义数据清洗函数
def clean_data(old_path, save_path):
    with open('C:/Users/23106/Desktop/weibopy/train_usual.csv', 'r', encoding='utf8') as f:
        data = defaultdict(list)
        texts = f.readlines()
        for line in tqdm(texts, desc='C:/Users/23106/Desktop/weibopy/train_usual.csv'):
            label, text = line.strip().split(',', 1)
            data['label'].append(label)
            emoji = json.load(
                open('C:/Users/23106/Desktop/weibopy/emoji.json', 'r', encoding='utf8'))
            for emoji, emoji_text in emoji.items():
                text = text.replace(emoji, emoji_text)
            new_text = re.sub(
                '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", text)
            data['text'].append(new_text)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join('C:/Users/23106/Desktop/weibopy/new/',
                  'train_usual.csv'), index=False, encoding='utf8', header=False, sep=',')
    return


with open('C:/Users/23106/Desktop/weibopy/train_usual.csv', 'r', encoding='utf8') as f:
    data = defaultdict(list)
    texts = f.readlines()
    for line in tqdm(texts, desc='C:/Users/23106/Desktop/weibopy/train_usual.csv'):
        label, text = line.strip().split(',', 1)
        data['label'].append(label)
        emoji = json.load(
            open('C:/Users/23106/Desktop/weibopy/emoji.json', 'r', encoding='utf8'))
        for emoji, emoji_text in emoji.items():
            text = text.replace(emoji, emoji_text)
        new_text = re.sub(
            '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", text)
        data['text'].append(new_text)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join('C:/Users/23106/Desktop/weibopy/new/',
              'train_usual.csv'), index=False, encoding='utf8', header=False, sep=',')


class OCEMOTION(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'train_usual.csv'
        elif mode == 'test':
            data_file = 'test_usual.csv'

        super(OCEMOTION, self).__init__(
            base_path="C:\Users\23106\Desktop\weibopy",
            data_file=data_file,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            is_file_with_header=True,
            label_list=label_list
        )

    # 解析文本文件里的样本
    def _read_file(self, input_file, is_file_with_header: bool = False):
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter=",")
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    try:
                        example = InputExample(
                            guid=seq_id, text_a=line[0], label=line[1])
                        seq_id += 1
                        examples.append(example)
                    except:
                        continue
                return examples


# max_seq_len根据具体文本长度进行确定，但需注意max_seq_len最长不超过512
train_dataset = OCEMOTION(model.get_tokenizer(), mode='train', max_seq_len=128)
dev_dataset = OCEMOTION(model.get_tokenizer(), mode='dev', max_seq_len=128)
test_dataset = OCEMOTION(model.get_tokenizer(), mode='test', max_seq_len=128)

# 查看训练集前3条
for e in train_dataset.examples[:3]:
    print(e)
# 查看验证集前3条
for e in dev_dataset.examples[:3]:
    print(e)
# 查看测试集前3条
for e in test_dataset.examples[:3]:
    print(e)


DATA_HOME = "./datasets/data/"


class TagDatasets(TextClassificationDataset):
    def __init__(self,  tokenizer: Union[BertTokenizer, CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'data_train_list.tsv'
        elif mode == 'test':
            data_file = 'data_test_list.tsv'
        else:
            data_file = 'data_all_list.tsv'
        super().__init__(
            base_path=DATA_HOME,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            data_file=data_file,
            label_list=['0', '1', '2', '3', '4'],  # 改动2
            is_file_with_header=True)


train_dataset = TagDatasets(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='train')
test_dataset = TagDatasets(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='test')


def clean_data(old_path, save_path, save_name):
    with open(old_path, 'r', encoding='utf8') as f:
        data = defaultdict(list)
        texts = f.readlines()
        for line in tqdm(texts, desc=old_path):
            label, text = line.strip().split(',', 1)
            emoji = json.load(open('emoji.json', 'r', encoding='utf8'))
            for emoji, emoji_text in emoji.items():
                text = text.replace(emoji, emoji_text)
            new_text = re.sub(
                '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", text)
            data['text'].append(new_text)
            data['label'].append(label)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, save_name), index=False,
                  encoding='utf8', header=False, sep='\t')
    return


data = pd.read_csv('train_usual.csv', header=None)
data.columns = ["label", "text"]
data.head()

model = hub.Module(name="ernie_tiny", task='seq-cls',
                   num_classes=6, label_map=label_map)


class TEXTALS(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'train.csv'
        elif mode == 'test':
            data_file = 'test.csv'
        elif mode == 'valid':
            data_file = 'valid.csv'

        super(TEXTALS, self).__init__(
            base_path=DIR,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=label_list,
            is_file_with_header=True,

        )

    def _read_file(self, input_file, is_file_with_header: bool = False):
        with io.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            header = next(reader) if is_file_with_header else None
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[1], text_a=line[0])
                seq_id += 1
                examples.append(example)
        return examples


train_dataset = TEXTALS(model.get_tokenizer(), mode='train',
                        max_seq_len=128)
test_dataset = TEXTALS(model.get_tokenizer(), mode='test', max_seq_len=128)
valid_dataset = TEXTALS(model.get_tokenizer(), mode='valid', max_seq_len=128)


train_dataset = TEXTALS(model.get_tokenizer(), mode='train', max_seq_len=128)
test_dataset = TEXTALS(model.get_tokenizer(), mode='test', max_seq_len=128)
valid_dataset = TEXTALS(model.get_tokenizer(), mode='valid', max_seq_len=128)


optimizer = paddle.optimizer.AdamW(
    learning_rate=4e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt',
                      use_gpu=True, use_vdl=True)


trainer.train(train_dataset, epochs=5, batch_size=32,
              eval_dataset=valid_dataset, save_interval=1)


# weibo paxong


def fetchUrl(pid, uid, max_id):
    # url
    url = "https://weibo.com/ajax/statuses/buildComments"
    # 请求头
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
    }
    # 参数
    params = {
        "flow": 0,
        "is_reload": 1,
        "id": pid,
        "is_show_bulletin": 3,
        "is_mix": 0,
        "max_id": max_id,
        "count": 20,
        "uid": uid,
    }

    r = requests.get(url, headers=headers, params=params)
    return r.json()


def parseJson(jsonObj):

    data = jsonObj["data"]
    max_id = jsonObj["max_id"]
    commentData = []
    for item in data:
        # 评论id
        comment_Id = item["id"]
        # 评论内容
        content = BeautifulSoup(item["text"], "html.parser").text
        # 评论时间
        created_at = item["created_at"]
        # 点赞数
        like_counts = item["like_counts"]
        # 评论数
        total_number = item["total_number"]

        # 评论者 id，name，city
        user = item["user"]
        userID = user["id"]
        userName = user["name"]
        userCity = user["location"]
        emoji = json.load(
            open('C:/Users/23106/Desktop/weibopy/emoji.json', 'r', encoding='utf8'))
        for emoji, emoji_text in emoji.items():
            content = content.replace(emoji, emoji_text)
        content = re.sub(
            '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", content)
        if content == '':
            continue
        dataItem = [content]
        print(dataItem)
        commentData.append(dataItem)

    return commentData, max_id


def save_data(data, path, filename):

    if not os.path.exists(path):
        os.makedirs(path)

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path + filename, mode='a',
                     index=False, sep=',', header=False)


if __name__ == "__main__":

    pid = 4727457598933626      # 微博id，固定
    uid = 2140522467            # 用户id，固定
    max_id = 0
    path = "C:/Users/23106/Desktop/weibopy/new/"           # 保存的路径
    filename = "comments.csv"   # 保存的文件名

    # csvHeader = [["评论id", "发布时间", "用户id", "用户昵称", "用户城市", "点赞数", "回复数", "评论内容"]]
    # save_data(csvHeader, path, filename)

    while(True):
        html = fetchUrl(pid, uid, max_id)
        comments, max_id = parseJson(html)
        save_data(comments, path, filename)
        # max_id 为 0 时，表示爬取结束
        if max_id == 0:
            break;


# jeishu


<div class = "container" >
        <div class = "row clearfix" >
            <div class = "col-md-4 column" >
            </div >
            <div class = "col-md-4 column" >
                <div class = "page-header" >
                    <h1 >
                        中文微博情感分析系统
                    </h1 >
                </div >
                <form role = "form" action = "/text_deal/" method = "post" >
                    { % csrf_token % }
                    <div class = "form-group" >
                        <label style = "font-size:large" > 待分析文本 < /label > <input type = "text" class = "form-control" id = "text"
                            name = "text" required = "required" / >
                    </div >

                    <!-- < div class = "form-group" >
                        <label style = "font-size:large" > 待分析微博链接 < /label > <input type = "text" class = "form-control" id = "url"
                            name = "url" required = "required" / >
                    </div > - ->

                    <div style = "text-align:right" >
                        <button class = "btn btn-lg btn-primary btn-block" type = "submit"
                            class = "btn btn-default" > 分析 < /button >
                    </div >
                </form >
                <div >
                    <label style = "color:lightslategrey;font-size:large" > {{msg}} < /label >
                </div >
                <br / >
                <br / >
                <br / >
                <br / >
                <br / >
                <form role = "form" action = "/data_deal/" method = "post" >
                    { % csrf_token % }
                    <div class = "form-group" >
                        <label style = "font-size:large" > 待分析微博链接 < /label > <input type = "text" class = "form-control" id = "ana_text"
                            name = "ana_text" required = "required" / >
                    </div >

                    <!-- < div class = "form-group" >
                        <label style = "font-size:large" > 待分析微博链接 < /label > <input type = "text" class = "form-control" id = "url"
                            name = "url" required = "required" / >
                    </div > - ->

                    <div style = "text-align:right" >
                        <button class = "btn btn-lg btn-primary btn-block" type = "submit"
                            class = "btn btn-default" > 分析 < /button >
                    </div >
                </form >
        </div >
        <div class = "col-md-4 column" >
        </div >
    </div>
    </div>








def getCom(com_url):
    str = com_url
        new_str = str.split("/", 5)
        com_id = new_str[4]  # 微博id
        user_id = new_str[3]  # 用户id
        max_id = 0
        url = "https://weibo.com/ajax/statuses/buildComments"
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        }
        if os.path.exists("C:/Users/23106/Desktop/weibopy/new/comments.csv"):
            os.remove("C:/Users/23106/Desktop/weibopy/new/comments.csv")
        if os.path.exists("C:/Users/23106/Desktop/weibopy/new/result.csv"):
            os.remove("C:/Users/23106/Desktop/weibopy/new/result.csv")
        path = "C:/Users/23106/Desktop/weibopy/new/comments.csv"  # 文件存储路径
        num = 0
        while(True):
            params = {
                "flow": 0,
                "is_reload": 1,
                "id": com_id,
                "is_show_bulletin": 2,
                "is_mix": 0,
                "max_id": max_id,
                "count": 20,
                "uid": user_id,
            }
            print(params)
            req = requests.get(url, headers=headers, params=params)
            html = req.json()
            data = html["data"]
            max_id = html["max_id"]
            comments = []
            for item in data:
                comment = BeautifulSoup(item["text"], "html.parser").text
                print(comment)
                num = num+1
                emoji = json.load(
                    open('C:/Users/23106/Desktop/weibopy/emoji.json', 'r', encoding='utf8'))
                for emoji, emoji_text in emoji.items():
                    comment = comment.replace(emoji, emoji_text)
                comment = re.sub(
                    '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", comment)
                if comment == '':
                    continue
                comment_data = [comment]
                # print(comment_data)
                comments.append(comment_data)
            dataframe = pd.DataFrame(comments)
            dataframe.to_csv(path, mode='a', index=False,
                             sep=',', header=False)
            print(max_id)
            if max_id == 0:
                break
            if num > 1000:
                break
        return