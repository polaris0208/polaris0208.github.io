---
layout: post
title: Django 기초 2
subtitle: TIL Day 80
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic 2

## CRUD 적용

### `models.py`
- `class Article(models.Model):` : 제목/내용 으로 구성

```py
from django.db import models

# Create your models here.
class Article(models.Model):
  title = models.CharField(max_length=50)
  content = models.TextField()
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)

  def __str__(self):
    return self.title
```

### `forms.py`
- **Django**에서 제공하는 **form** 양식

```py
from django import forms
from .models import Article
```

#### Django Form
- 모델 작성과 큰 차이가 없음
- 중복 작성하는 느낌

```py
class ArticleForm(forms.Form):
    title = forms.CharField(max_length=50)
    content = forms.CharField(widget=forms.Textarea)
```

#### Model Form
- 모델과 직접 가져옴
- 데이터를 전부 가져온 뒤 `exclude`로 일부 제외처리

```py
class ArticleForm(forms.ModelForm):
  class Meta:
    model = Article
    fields = "__all__"
    # exclude = ('created_at', 'updated_at')
```

### `urls.py`
- `from . import views` : 해당 앱의 `views.py` 불러오기
- 기본 **url** 구조 : `/articles/`
- `app_name = 'articles'` : **namespace** 설정
  - 결과 : `articles:create` 형식으로 접근
- `<int:pk>`: `pk`, **id** 값을 받아 처리

```py
app_name = 'articles'
urlpatterns = [
    path('', views.articles, name='articles'),
    #
    path('create/', views.create, name='create'),
    path('<int:pk>/', views.detail, name='detail'),
    path('<int:pk>/delete/', views.delete, name='delete'),
    path('<int:pk>/update/', views.update, name='update'),
]
```

### templates
- **namespace** 적용
- `<app이름>/templates/<app이름>/html파일`
- `<app이름>/html파일` 형식으로 접근

### `views.py`
- `render, redirect` : 연결 기능
- 모델, 폼 호출

```py
from django.shortcuts import render, redirect
from .models import Article
from .forms import ArticleForm
```

#### `article`
- 데이터 목록 페이지
- `.order_by("-created_at")` : 생성시간을 기준으로 내림차순으로 정렬
- `Article.objects.all()` : 데이터를 모두 조회
  - `context = {"articles": articles}` : `context`에 포함
    - `return render(request, "articles/articles.html", context)` : `html` 파일에 전달

```py
def articles(request):
    articles = Article.objects.all().order_by("-created_at")
    context = {"articles": articles}
    return render(request, "articles/articles.html", context)
```

#### `detail`
- 데이터 항목의 상세 페이지
- `.get(pk=pk)` : `pk`를 기준으로 데이터 호출
  - `pk` : **Primary Key**
  - 호출한 데이터를 전달

```py
def detail(request, pk):
    article = Article.objects.get(pk=pk)
    context = {"article": article}
    return render(request, "articles/detail.html", context)
```

#### `create`
- 새로운 데이터 생성
- `if request.method == "POST":`: `POST` 형식으로 들어온 요청일 경우
  - `article = form.save()` : 저장
  - `return redirect("articles:detail", article.pk)` : 상세 페이지로 연결
- `else: form = ArticleForm()` : 아닌 경우 입력 폼 호출
  - `return render(request, "articles/create.html", context)` : 생성 페이지로 전달

```py
def create(request):
    if request.method == "POST":
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save()
            return redirect("articles:detail", article.pk)
    else:
        form = ArticleForm()

    context = {"form": form}
    return render(request, "articles/create.html", context)
```

#### `delete`
- `.get(pk=pk)` : `pk`를 기준으로 데이터 호출
  - `article.delete()` : 삭제
  - 삭제 후 목록으로 이동

```py
def delete(request, pk):
    if request.method == "POST":
        article = Article.objects.get(pk=pk)
        article.delete()
        return redirect("articles:articles")
    return redirect("articles:detail", pk)
```

#### `update`
- `form = ArticleForm(request.POST, instance=article)`
  - `instance` 기준으로 호출

```py
def update(request, pk):
    article = Article.objects.get(pk=pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid(): 
            form.save()
            return redirect('articles:detail', article.pk)
    else: 
        form = ArticleForm(instance=article)

    context = {
        "form": form,
        'article' : article
        }
    return render(request, "articles/update.html", context)
```

## Login
- **Django** 기능 제공

### Accounts 기능 생성
- `python3 manage.py startapp accounts`
- `settings.py`
  - `INSTALLED_APPS`에 등록
- `urls.py`에 등록
  - `path('accounts/', include('accounts.urls'))`

### `urls.py`
- 기능별로 생성

```py
app_name = 'accounts'
urlpatterns = [
  path('login/', views.login, name='login'),
  path('logout/', views.logout, name='logout'),
]
```

### `views.py`

#### 모듈
- `render, redirect` : 연결 기능
- `AuthenticationForm` : 회원 정보 입력 폼
- `import login as auth_login` : 로그인 기능
  - `login` 함수와 충돌 방지하기 위해 이름 변경
- `import logout as auth_logout` : 로그아웃 기능

```py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
```

#### 기능 구현
- `auth_login(request, form.get_user())`
  - `form.get_user()` : 사용자 입력에서 로그인에 필요한 정보를 가져옴

```py
def login(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            return redirect('articles:articles')
    else:
        form = AuthenticationForm()
        context = {"form": form}
        return render(request, "accounts/login.html", context)
```