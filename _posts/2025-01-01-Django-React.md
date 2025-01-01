---
layout: post
title: Django - React 연결
subtitle: TIL Day 100
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

## 실행
- `python manage.py ruserver` : **Django** 구동 / 8000 포트
- `yarn start` : **React** 구동 / 3000 포트

## Cors
- 교차 출처 리소스 공유 **(Cross-Origin Resource Sharing)** 
  - 서로 다른 서버의 자원에 접근할 수 있는 권한을 부여
  - 기본적으로는 보안을 이유로 교차 출처 **HTTP** 요청 제한

## `django-cors-hearders`
- **Django**에서 **React** 화면 출력

### 설치
- `pip install django-cors-headers`

### `settings.py`
- `TEMPLATES` : **React** 템플릿 경로 지정
- `STATICFILES_DIRS` : **React** 정적 파일 경로 지정

```py
INSTALLED_APPS = [
    ...
    # Third party
    'corsheaders',
    ...
MIDDLEWARE = [
    ...
    # Third party
    'corsheaders.middleware.CorsMiddleware',
    ...
CORS_ALLOW_ALL_ORIGINS = True
    ...
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, 'frontent', 'build')],
    ...
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'frontent', 'build', 'static')]
    ...
```

### `urls.py`
- **URL**에서 템플릿을 호출
  - 설정한 **React** 경로에서 `index.html`을 찾아 출력

```py
from django.contrib import admin
from django.urls import path, include

from django.views.generic import TemplateView

urlpatterns = [
    ...
    path("", TemplateView.as_view(template_name='index.html')),
    ...
]
```