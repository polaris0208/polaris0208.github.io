---
layout: post
title: Django 기초
subtitle: TIL Day 79
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic
> **Python** 기반 웹 프레임워크

## 개념
- **DRY(Don’t Repeat Yourself)** 원칙
    - 코드 중복을 최소화
- 보안, 관리자기능, Auth 등 기능 제공
- 풍부한 레퍼런스의 검증된 프레임워크

## 개발환경 구성

### LTS(Long Term Support)
- 소프트웨어나 기술 제품의 장기 지원
- **Version** 선택의 중요한 고려요소
  - **Version** 구성 : `Major.Minor.Patch`
    - `Major` : 큰 변화, 호환성 차이
    - `Minor` : 작은 변화, 주로 기능 추가
    - `Patch` : 개선
- 예시 : **Python**
  - 최신 버젼 `Python 3.12`
  - 안정적인 버젼 `Python 3.10`

### 가상환경 생성
- `python3 -m venv my_env`
  - `-m` : 생성
  - `venv` : 가상환경
  - `my_env` : 가상환경 이름/폴더 이름
- `source my_env/bin/activate`
  - `source` : 명령 실행
  - `my_env/bin/` : 명령 디렉토리
  - `activate` : 실행 명령

## Django 프로젝트

### 생성
- `django-admin startproject <프로젝트 이름> <생성 디렉토리>`
- `<생성 디렉토리>` 내에 `<프로젝트 이름>` 폴더 생성
  - `<프로젝트 이름>` : 같은 이름의 하위 폴더
    - `__init__.py` : 패키지 초기화(인식) 파일
    - `settings.py` : 프로젝트의 설정을 관리
    - `urls.py` : 어떤 요청을 처리할지 판단
    - `wsgi.py` : 웹 서버 관련 설정 파일
  - `manage.py` : **Django** 프로젝트 유틸리티


### 서버 실행
- `cd <프로젝트 이름>` : 같은 이름의 하위 폴더로 이동
- `python3 manage.py runserver`

```bash
django-admin startproject my_pjt .
cd my_pjt
python3 manage.py runserver
```

### App
- 앱 생성 -> 앱 등록 단계

#### 생성
- `python manage.py startapp <앱 이름>`

#### 등록
- `settings.py`
  - `INSTALLED_APPS` 리스트 수정
    - `<앱 이름>` 추가

#### 결과
- `admin.py` : 관리자용 페이지 관련 설정
- `apps.py` : 앱 관련 정보 설정
- `models.py` : **DB**관련 데이터 정의 파일
- `tests.py` : 테스트 관련 파일 
- `views.py` : 요청을 처리하고 처리한 결과를 반환하는 파일

## Django 디자인 패턴
- **MVC** 패턴의 변형
    
### MVC 디자인 패턴
- **Model** : 데이터와 관련된 로직을 관리
- **View** : 레이아웃과 관련된 화면을 처리
- **Controller** : Model과 View를 연결하는 로직을 처리

### Django MTV Pattern
- **View** 의 역할에 주의

**MVC vs MTV**

| MVC | MTV |
| --- | --- |
| Model | Model |
| View | Template |
| Controller | View |

- **Model**
    - 데이터와 관련된 로직을 처리
- **Template**
    - 레이아웃과 화면상의 로직을 처리      
- **View**
    - 메인 비지니스 로직을 담당
    - 클라이언트의 요청에 대해 처리를 분기하는 역할

## Urls
- `urlpatterns = []` : `path` 형식으로 저장
- `path('index/', views.index, name='index')`
  - `index/` : 경로
  - `views.index` : `views.py` 내부의 함수로 연결
  - `name` : 별명 설정/**url**을 변수로 호출
- `path('articles/', include('articles.urls'))`
  - `include('articles.urls')` : `articles` 앱 내부의 **url**에 연결
  - 내부 **url** 은 `/aricles/`를 기본 경로로 가짐

```py
from django.contrib import admin
from django.urls import path, include
from articles import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index, name='index'),
    path('articles/', include('articles.urls')),
    path('users/', include('users.urls')),
]
```

## Templates
- `views.py`가 요청을 보낼 `html` 파일 저장
  - 상위 디렉토리에 폴더 생성 
    - 기본 디렉토리 설정 필요
    - 저장된 `html` 파일을 호출해서 사용 가능 : `base.html`
  - 각 앱의 디렉토리에 폴더 생성

```py
BASE_DIR = Path(__file__).resolve().parent.parent
# 프로젝트 루트 디렉토리
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

## Views
- 요청을 받아 처리
- `return render(request, 'hello.html', context)`
  - `request` : 요청
  - `hello.html` : 사용할`html` 파일
  - `context` : `html`에 사용될 데이터
- `def profile(request, username):`
  - `/users/<username>` : 페이지의 **url** `/users` 뒤의 `<username>` 인자로 전달됨

```py
def hello(request):
  name = 'Polaris'
  tags = ['python', 'django', 'html', 'css']
  books = ['설국', '인간실격', '상실의 시대']
  context = {
    'name' : name,
    'tags' : tags,
    'books' : books,
  }
  return render(request, 'hello.html', context)

def profile(request, username):
  context = {
    'username' : username
  }
  return render(request, 'profile.html', context)
```

## Model

```py
from django.db import models


class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True) 
    updated_at = models.DateTimeField(auto_now=True)
```

### Migration
- **Django**는 마이그레이션(**migration**)을 만들고 이 단위로 데이터베이스에 변경사항을 반영
- `python manage.py makemigrations ` : 생성
- `python manage.py migrate` : 반영

### 데이터베이스 확인
- `db.sqlite3` 파일
- **SQLite** : 확장기능 설치
- `command + shift + p` `SQLite: Open Database` 선택

### Django Shell
- 내장 터미널 개념

```bash
django shell
python manage.py shell
```

#### 확장 프로그램 설치
- 코드 색상 등 적용, 가시성 향상
- `settings.py`에 등록 필요
- `python manage.py shell_plus` : 실행 명령어 변경

```bash
pip install django-extensions
pip install ipython
```


### CRUD(Create Read Update Delete)
#### 전체 조회
  - `Article.objects.all()`
​
#### 생성
- `article.save()` : 저장하지 않으면 반영되지 않음

```bash
# 1
article = Article()
article.title = 'first_title'
article.content = 'content1'
# 2
article = Article(title = 'second_title', content = 'content2')
# 3 
Article.objects.create(title='third_title', content= 'content3')
# save()가 필요하지 않음

Article.objects.all() 
# 비어있음
article.save()
# 저장
Article.objects.all()
# 1개 생성
```

#### 조회

```bash
article.title
# 제목 
article.content
# 내용
article.create_at
# 생성일시
article.id
# id

Article.objects.get(id=1)
# id로 조회
Article.objects.get(content='content1')
# content로 조회
# 중복될 경우 오류
​
Article.objects.filter(content='my_content')
# 조건 추가
​
article = Article.objects.get(id=1)
article.title = 'updated title'
article.save()
# 업데이트
​
article = Article.objects.get(id=2)
article.delete()
# 삭제
```