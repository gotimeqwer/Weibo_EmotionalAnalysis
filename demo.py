import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import json
import re


def getCom(com_url):
    str = com_url
    new_str = str.split("/", 5)
    com_id = new_str[4]  # 微博id
    user_id = new_str[3]  # 用户id
    print(com_id, user_id)
    max_id = 0
    url = "https://weibo.com/ajax/statuses/buildComments"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
    }
    path = "C:/Users/23106/Desktop/weibopy/new/comments.csv"  # 文件存储路径
    print (path)
    while(True):
        params = {
            "flow": 0,
            "is_reload": 1,
            "id": com_id,
            "is_show_bulletin": 3,
            "is_mix": 0,
            "max_id": max_id,
            "count": 20,
            "uid": user_id,
        }
        print (params)
        req = requests.get(url, headers=headers, params=params)
        html = req.json()
        data = html["data"]
        max_id = html["max_id"]
        comments = []
        for item in data:
            comment = BeautifulSoup(item["text"], "html.parser").text
            print(comment)
            emoji = json.load(
                open('C:/Users/23106/Desktop/weibopy/emoji.json', 'r', encoding='utf8'))
            for emoji, emoji_text in emoji.items():
                comment = comment.replace(emoji, emoji_text)
            comment = re.sub(
                '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", comment)
            if comment == '':
                continue
            comment_data = [comment]
            print(comment_data)
            comments.append(comment_data)
        dataframe = pd.DataFrame(comments)
        dataframe.to_csv(path, mode='a', index=False,
                             sep=',', header=False)
        if max_id == 0:
            break;
    return


getCom("https://m.weibo.cn/1850268574/4756072126743158")