"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ana_text import views
urlpatterns = [
    #path('admin/', admin.site.urls),
    # 系统主界面
    path('index/', views.index),
    # 单文本处理
    path('text_deal/', views.text),
    # 微博链接处理
    path('data_deal/', views.csvDeal),
    # 用户登录
    path('login/', views.login),
    # 结果输出
    path('result/', views.result),
    path('result/text/', views.re_text),
    path('result/bar/', views.bar),
    path('result/pie/', views.pie),
    path('result/file/', views.file_deal),
    # 历史信息展示
    path('history/', views.history),
    # 用户注册
    path('register/', views.register),
    path('logout/', views.logout),
    path('revise/', views.revise),
    path('revise_password/', views.revise_password),
    path('mail/', views.mail),
    path('revise_mail/', views.revise_mail),
]
