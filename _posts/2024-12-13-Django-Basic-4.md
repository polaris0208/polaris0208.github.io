---
layout: post
title: Django 기초 4
subtitle: TIL Day 82
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic 4

## Static

### `settings.py`
- 경로 설정
  - `static` : 정적 파일 경로 설정
  - `media` : 미디어 파일 세팅
    - 개발 환경에서만 테스트를 위해 작동할 경로
    - 실제 배포환경에서는 별도의 공간 구축
  
```py
BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]  # 경로 참조
STATIC_ROOT = BASE_DIR / "staticfiles"  # 배포시 사용
# 미디어 세팅
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

### 이미지 기능 추가

#### 모델 수정
- 이미지의 경로가 들어갈 테이블 추가
  - `blank`: 공백 허용 설정
- 수정 후에는 **migration** 실시 해 적용

```py
class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    #
    image = models.ImageField(upload_to='images/', blank=True) # 비어있어도 된다

    def __str__(self):
        return self.title
```

#### 정적파일 경로
- `<app이름>/static/<app이름>` 구조로 생성
- `<app이름>/<파일명>` 으로 접근

#### 템플릿 반영
- **Django Template Tag** 사용
  - `load static` : 템플릿 상단에서 선언
  - `static` 태그 사용 : 설정한 경로로 접근
- `article.image` 와 `article.image.url`로 접근 가능

## 커스텀 User 모델
- 프로젝트 생성 시 설정 권장
  - 중간에 수정하는 경우 **DB** 초기화 필요
  - **SQLite** 파일, 생성된 `migrations` 파일 삭제

### `settings.py`
- 사용자의 `User` 모델 지정
  - `AUTH_USER_MODEL = "accounts.User"`
  - `model = get_user_model()`로 작성된 곳에 자동 적용

### `models.py`
- `accounts`에서 설정
- `class User(AbstractUser): : pass` 
  - 커스텀 모델로 설정 후 프로젝트를 진행하면서 기능 추가

## 작성자 기능

### `models.py`
- 작성자 테이블 추가
- `"accounts.User"` : 사용자의 **User** 모델 활용
- `CASCADE` : 사용자 정보가 삭제되면 같이 삭제
- `article.author`로 데이터에 접근 가능

```py
class Article(models.Model):
  ...
      author = models.ForeignKey("accounts.User", on_delete=models.CASCADE, related_name="articles")
  ...
```

## 좋아요 기능

### `urls.py`
- 작성글에 귀속되도록 설정

```py
from django.contrib import admin
from django.urls import path
from . import views

app_name = "articles"
# 'articles:`
urlpatterns = [
    ...
    path("<int:pk>/like/" , views.like, name='like'),
    ...
```

### `models.py`
- 좋아요 테이블 추가
- `ManyToManyField` : 다 대 다, **N : M** 관계
  - 사용자는 여러 글에 좋아요 가능/ 게시글은 여러 사용자에게 좋아요를 받을 수 있음
  - **Django** 중계 테이블을 생성하여 기능 제공

```py
class Article(models.Model):
  ...
    like_user = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name="like_articles",
    )
  ...
```

### `views.py`
- `@require_POST` : 데이터 테이블 사용하기 떄문에 `POST` 요철
- `if request.user.is_authenticated:` : 로그인 후 사용
- `if article.like_user.filter(pk=request.user.pk).exists():`
  - 이미 좋아요가 있다면 제거

```py 
@require_POST
def like(request, pk):
    if request.user.is_authenticated:
        article = get_object_or_404(Article, pk=pk)
        if article.like_user.filter(pk=request.user.pk).exists():
            article.like_user.remove(request.user)
        else:
            article.like_user.add(request.user)
        return redirect("articles:articles")
    else:
        return redirect("accounts:login")
```

### 템플릿 설정
- `if request.user in article.like_user.all` : 좋아요를 누른 명단에 있는지 확인
  - 결과에 따라 다른 버튼 표시

## 팔로우 기능

### `models.py`
- `accounts` : 계정과 연동되는 기능
- 커스텀 **User** 모델에 기능 추가
  - `self` : 자기 자신을 참조하여 작동
  - `symmetrical` : 대칭 기능

```py
class User(AbstractUser):
  following = models.ManyToManyField(
    "self", related_name='followers', symmetrical=False
    )
``` 

### `urls.py`
- `users.py` : 사용자 페이지에서 기능 작동

```py
app_name = 'users'
urlpatterns = [
    path('', views.users, name='users'),
    path('profile/<str:username>/', views.profile, name='profile'),
    path('<int:user_id>/follow/', views.follow, name='follow'),
]
```

### `views.py`

#### `profile`
- 팔로우 버튼이 작동할 공간
- `get_object_or_404(get_user_model(), username=username)`
  - 사용자의 정보를 가져와서 페이지 표시
- 템플릿 설정
  - 본인이 접속한 경우 팔로우 버튼 비활성화
  - `if request.user != member`

```py
def profile(request, username):
    member = get_object_or_404(get_user_model(), username=username)
    context = {"member": member}
    return render(request, "users/profile.html", context)
```

#### `follow`
- 팔로우 기능 처리
- `if request.user.is_authenticated:` : 로그인 필요
- `if member != request.user:` : 본인은 팔로우 불가
- `if member.followers.filter(pk=request.user.pk).exists():`
  - 이미 팔로우 중인경우 클릭 시 언팔로우

```py
@require_POST
def follow(request, user_id):
    if request.user.is_authenticated:
        member = get_object_or_404(get_user_model(), pk=user_id)
        if member != request.user:
            if member.followers.filter(pk=request.user.pk).exists():
                member.followers.remove(request.user)
            else:
                member.followers.add(request.user)
        return redirect('users:profile', username=member.username)
    else:
        return redirect("accounts:login")
```

## 댓글 기능 수정
- 작성자 표시 및 삭제 권한 설정

### `models.py`
- `articles` : 게시글의 부속 기능
- 사용자 정보 테이블 추가

```py
class Comment(models.Model):
    ...
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="comments"
    )
    ...
```

### 템플릿
- `comment.user`으로 접근 가능