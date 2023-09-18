import paddlehub as hub
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import json
import re


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
            break

url = "https://weibo.com/ajax/statuses/buildComments"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
}
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
req = requests.get(url, headers=headers, params=params)
html = req.json()
data = html["data"]
for item in data:
    comment = BeautifulSoup(item["text"], "html.parser").text
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
        comments.append(comment_data)
        dataframe = pd.DataFrame(comments)
        dataframe.to_csv(path, mode='a', index=False,
                         sep=',', header=False)


#


#
model = hub.Module(
    name='ernie_tiny',
    task='seq-cls',
    num_classes=6,
    load_checkpoint='C:/Users/23106/Desktop/weibopy/ckpt/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data_list, max_seq_len=128,
                        batch_size=32, use_gpu=True)
data = defaultdict(list)
for index, text in enumerate(data_list):
    data['text'].append(text[0])
    data['label'].append(results[index])
df = pd.DataFrame(data)


data_list = [
    {"value": num2, "name": '悲伤'},
    {"value": num1, "name": '积极'},
    {"value": num5, "name": '惊奇'},
    {"value": num0, "name": '愤怒'},
    {"value": num3, "name": '无情绪'},
    {"value": num4, "name": '恐惧'}
]
$.ajax({
    url: "/result/pie/",
    type: "get",
    dataType: "JSON",
    success: function(res) {
         if (res.status) {
             option.series[0].data=res.data
             myChart.setOption(option)
         }
    }
})









{ % for obj in history_list % }
   <tr>
       <td>
            {{forloop.counter}}
        </td>
        <td>
            {{obj.history}}
        </td>
    </tr>
{ % endfor % }
