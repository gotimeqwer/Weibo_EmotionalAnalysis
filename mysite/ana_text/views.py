from django.http import HttpResponse, JsonResponse, FileResponse
from django.shortcuts import render, redirect
from ana_text.models import UserInfo, UserHistory, UsersHistory
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import pandas as pd
import re
import os
import csv
import paddlehub as hub
import paddle
from collections import defaultdict

# Create your views here.

# 主界面


def index(request):
    id = request.session.get("info")
    if not id:
        return redirect('/login/')
    return render(request, "index.html", {"id_msg": id})

# 文本处理


def text(request):
    id = request.session.get("info")
    if not id:
        return redirect('/login/')
    if request.method == "GET":
        return render(request, "index.html")
    else:
        data = []
        list = []
        UsersHistory.objects.create(
            userid=id, history=request.POST.get("text"))
        list.append(request.POST.get("text"))
        data.append(list)
        print(data)
        label_list = ['愤怒', '积极', '悲伤', '无情绪', '恐惧', '惊奇']
        label_map = {idx: label_text for idx,
                     label_text in enumerate(label_list)}
        print(label_map)
        print(label_list)
        print(type(data))
        model = hub.Module(
            name='ernie_tiny',
            task='seq-cls',
            num_classes=7,
            load_checkpoint='C:/Users/23106/Desktop/weibopy/ckpt/best_model/model.pdparams',
            label_map=label_map)
        results = model.predict(data, max_seq_len=128,
                                batch_size=32, use_gpu=True)
        for index, text in enumerate(data):
            print('Text: {} \t Label: {}'.format(text[0], results[index]))
        # return render(request, "/index/",{"msg":results[idx]})
        return render(request, "re_text.html", {"text_msg": text[0], "label_msg": results[index]})
        #return HttpResponse('Text: {} \t Label: {}'.format(text[0], results[index]))
        # return render(request, "text.html")

# 文件处理


def csvDeal(request):
    id = request.session.get("info")
    if not id:
        return redirect('/login/')
    if request.method == "GET":
        return render(request, "index.html")
    else:
        ana_url = request.POST.get("ana_text")
        print(ana_url)
    UsersHistory.objects.create(userid=id, history=ana_url)
    print("1")

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
    getCom(ana_url)
    label_list = ['愤怒', '积极', '悲伤', '无情绪', '恐惧', '惊奇']
    label_map = {idx: label_text for idx, label_text in enumerate(label_list)}
    print(label_list)
    data = pd.read_csv(
        'C:/Users/23106/Desktop/weibopy/new/comments.csv', header=None)
    # 格式处理：
    data_array = np.array(data)
    data_list = data_array.tolist()
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
        print('Text: {} \t Label: {}'.format(text[0], results[index]))
        data['text'].append(text[0])
        data['label'].append(results[index])
    df = pd.DataFrame(data)
    df.to_csv("C:/Users/23106/Desktop/weibopy/new/result.csv",
              index=False, encoding='utf8', header=False, sep='\t')
    # 将数据转化为数组
    return redirect('/result/')
    # return HttpResponse('Text: {} \t Label: {}'.format(text[0], results[index]))


# 登录操作
def login(request):
    msg = ""
    if request.method == "GET":
        return render(request, "login.html")
    else:
        id = request.POST.get("id")
        passw = request.POST.get("password")
        print(id, passw)
        user_list = UserInfo.objects.filter(userid=id).first()
        if user_list.password == passw:
            msg = ""
            request.session["info"] = id
            return redirect("/index/")
        msg = "用户名或密码错误"
        print(passw)
        return render(request, "login.html", {"error_msg": msg})

# 结果展示


def result(request):
    id = request.session.get("info")
    if not id:
        return redirect('/login/')
    else:

        return render(request, "result.html")
        # return HttpResponse(id)


def bar(request):
    data = pd.read_csv(
        'C:/Users/23106/Desktop/weibopy/new/result.csv', sep='\t', header=None)
    data.columns = ["text", "label"]
    data_num = data.shape
    print(data_num[0])  # 输出爬取的微博数量
    num0 = sum(data['label'] == "愤怒")
    num1 = sum(data['label'] == "积极")
    num2 = sum(data['label'] == "悲伤")
    num3 = sum(data['label'] == "无情绪")
    num4 = sum(data['label'] == "恐惧")
    num5 = sum(data['label'] == "惊奇")
    num = [num1, num5, num3, num2, num0, num4]
    data_list = [
        {
            "name": '评论数量',
            "type": 'bar',
            "data": [num1, num5, num3, num2, num0, num4],
        }
    ]
    result = {
        "status": True,
        "data": {
            'series_list': data_list,
        }
    }
    print('愤怒：', num0)
    print('积极：', num1)
    print('悲伤：', num2)
    print('无情绪：', num3)
    print('恐惧：', num4)
    print('惊奇：', num5)
    return JsonResponse(result)


def pie(request):
    data = pd.read_csv(
        'C:/Users/23106/Desktop/weibopy/new/result.csv', sep='\t', header=None)
    data.columns = ["text", "label"]
    num0 = sum(data['label'] == "愤怒")
    num1 = sum(data['label'] == "积极")
    num2 = sum(data['label'] == "悲伤")
    num3 = sum(data['label'] == "无情绪")
    num4 = sum(data['label'] == "恐惧")
    num5 = sum(data['label'] == "惊奇")
    data_list = [
        {"value": num2, "name": '悲伤'},
        {"value": num1, "name": '积极'},
        {"value": num5, "name": '惊奇'},
        {"value": num0, "name": '愤怒'},
        {"value": num3, "name": '无情绪'},
        {"value": num4, "name": '恐惧'}
    ]
    result = {
        "status": True,
        "data": data_list,
    }
    return JsonResponse(result)


def file_deal(request):
    filename = r'C:/Users/23106/Desktop/weibopy/new/result.csv'
    down_path = open(filename, 'rb')
    d = HttpResponse(down_path)
    d['content_type'] = "application/octet-stream"
    d['Content-Disposition'] = 'attachment; filename=' + \
        os.path.split(filename)[1]
    return d


# 历史记录


def history(request):
    id = request.session.get("info")
    if not id:
        return redirect('/login/')
    else:
        print(id)
        history_list = UsersHistory.objects.filter(userid=id)
        print(history_list)
        return render(request, "history.html", {"history_list": history_list,"length":len(history_list)})


# 用户注册
def register(request):
    if request.method == "GET":
        return render(request, "register.html")
    else:
        id = request.POST.get("id")
        password = request.POST.get("password")
        mail = request.POST.get("mail")
        UserInfo.objects.create(userid=id, password=password,mail=mail)
        print("注册成功")
        data_list = UserInfo.objects.all()
        print(data_list)
    return render(request, "login.html")


def logout(request):
    request.session.clear()
    return render(request, "login.html")


def re_text(request):
    request.session.clear()
    return render(request, "login.html")



def revise(request):
    if request.method == "GET":
        return render(request, "revise.html")



def revise_password(request):
    id = request.session.get("info")
    old_pass = request.POST.get("old_password")
    print(old_pass)
    new_pass = request.POST.get("new_password")
    print(id, old_pass)
    user_list = UserInfo.objects.filter(userid=id).first()
    if user_list.password == old_pass:
        UserInfo.objects.filter(userid=id).update(password=new_pass)
        print("1")
        return redirect("/index/")
    else:
        print("2")
        return render(request, "revise.html")

    

def mail(request):
    if request.method == "GET":
        id = request.session.get("info")
        user_list = UserInfo.objects.filter(userid=id).first()
        mail = user_list.mail
        print(mail)
        return render(request, "mail.html", {"mail_msg": mail})
        #return render(request, "mail.html")



def revise_mail(request):
    id = request.session.get("info")
    new_mail = request.POST.get("mail")
    UserInfo.objects.filter(userid=id).update(mail=new_mail)
    print("1")
    return redirect("/index/")