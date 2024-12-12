---
layout: post
title: Django 기초 3
subtitle: TIL Day 81
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic 3

## Decorator
- 함수 앞에서 작동하여 추가 기능을 구현
- 전제 조건 또는 포장 개념

### 로그인/로그아웃
- 기존 함수에 데코레이터 추가
- `from django.views.decorators.http`
- `require_POST` : **POST** 요청만 받도록 제한
- `require_http_methods` : 요청 방식 제한, 설정 가능
- `next_url`: 로그인 후 쿼리스트링 `next` 뒤에 정의된 위치로 이동

```py
@require_http_methods(["GET", "POST"])
def login(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            next_url = request.GET.get("next") or "articles:hello"
            return redirect(next_url)
    else:
        form = AuthenticationForm()
        context = {"form": form}
        return render(request, "accounts/login.html", context)


@require_POST
def logout(request):
    auth_logout(request)
    return redirect("accounts:login")
```

## 회원 기능
- **Django** 가 제공하는 **Form** 을 가공하여 구축

```py
from django.contrib.auth.forms import (
    AuthenticationForm,
    UserCreationForm,
    PasswordChangeForm,
)
from .forms import CustomUserUpdateForm
from django.contrib.auth import update_session_auth_hash
```

### 회원 가입
- 작성된 내용이 **POST** 요청으로 들어오면 저장
- 처음 **URL**로 **GET**요청이 들어오면 입력**Form**을 보여줌
- `auth_login(request, user)` : 회원가입이 완료되면 해당 정보를 가지고 로그인

```py
@require_http_methods(["GET", "POST"])
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("articles:articles")
    else:
        form = UserCreationForm()
    context = {"form": form}
    return render(request, "accounts/signup.html", context)
```

### 수정
-`get_user_model` : `settings.py` 활성화 된 모델
  - 없으면 디폴트 값을 호출 : `User` 모델

```py 
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth import get_user_model
from django.urls import reverse
```

#### Form 생성
- 필요한 기능만 선택해 상속

```py
class CustomUserUpdateForm(UserChangeForm):

    class Meta:
        model = get_user_model()
        fields = [
            "first_name",
            "last_name",
            "email",
        ]
```

#### 수정 기능
- 작성된 내용이 **POST** 요청으로 들어오면 저장
- 처음 **URL**로 **GET**요청
  - 입력**Form** 호출
  - 회원 정보를 `instance`로 불러와 보여줌

```py
@require_http_methods(["GET", "POST"])
def update(request):
    if request.method == "POST":
        form = CustomUserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("articels:articles")
    else:
        form = form = CustomUserUpdateForm(instance=request.user)

    context = {"form": form}
    return render(request, "accounts/update.html", context)
```

### 비밀번호 변경
- 회원 정보 수정 **Form** 하단에 안내 생성

#### 비밀번호 변경 안내 문구 수정
- `reverse('accounts:change-password')` : 사용자의 양식을 역으로 참조하여 **URL** 생성

```py
class CustomUserUpdateForm(UserChangeForm):

    class Meta:
        model = get_user_model()
        fields = [
            "first_name",
            "last_name",
            "email",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fields.get("password"):
            password_help_text = (
                "You can change the password " '<a href="{}">here</a>.'
            ).format(f"{reverse('accounts:change-password')}")

            self.fields["password"].help_text = password_help_text
```

#### 비밀번호 변경 기능
- `update_session_auth_hash(request, form.user)`
  - 변경된 비밀번호의 세션 데이터로 업데이트

```py
@require_http_methods(["GET", "POST"])
def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            return redirect("articles:articels")
    else:
        form = PasswordChangeForm(request.user)

    context = {"form": form}
    return render(request, "accounts/change-password.html", context)
```

### 탈퇴
- `auth_logout(request)` : 쿠키 제거를 위해 로그아웃 처리

```py
@require_POST
def resign(request):
    if request.user.is_authenticated:
        request.user.delete()
        auth_logout(request)
    return redirect("articles:articles")
```

## 댓글 기능

### `urls.py`
- `pk` : 데이터의 키를 기준으로 작성

```py 
path("<int:pk>/comments/", views.comment_create, name="comment_create"),
path("<int:pk>/comments/<int:comment_pk>/delete/", views.comment_delete, name="comment_delete",),
```

### `models.py`
- 작성 후 **migration** 처리
  - `python3 manage.py makemigrations`
  - `python3 manage.py migrate`
- `ForeignKey` : `Article` 모델의 **PK**를 **FK**로 사용하여 연결
- `CASCADE` : 참조하는 데이터가 삭제되면 같이 삭제
  - 글이 삭제되면 댓글도 삭제
- `related_name='comments'` : 매니저 이름 변경
  - 기본값 : `commet_set`

```py
class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='comments')
    content = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.content
```

### `forms.py`
- `exclude = {'article'}` : 본문은 제외

```py
class CommentForm(forms.ModelForm):
  class Meta:
    model = Comment
    fields = '__all__'
    exclude = {'article'}
```

### `views.py`

#### comment 기능
- `get_object_or_404`: 올바르지 않은 요청에 `404` 응답처리

```py
@require_POST 
def comment_create(request, pk):
    article = get_object_or_404(Article, pk=pk)
    form = CommentForm(request.POST)
    if form.is_valid():
        comment = form.save(commit=False)
        comment.article = article
        comment.save()
        return redirect('articles:detail', article.pk)

@require_POST
def comment_delete(request, pk, comment_pk):
    comment = get_object_or_404(Comment, pk=comment_pk)
    comment.delete()
    return redirect("articles:detail", pk)
```

#### `detail` 페이지 수정
- 댓글 작성 폼/댓글 목록 전달
- `comments = article.comments.all().order_by("-pk")` : 내림차순으로 정렬

```py
def detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    comment_form = CommentForm()
    comments = article.comments.all().order_by("-pk")
    context = {
        "article": article,
        "comment_form" : comment_form,
        'comments' : comments,
        }
    return render(request, "articles/detail.html", context)
```

## 관리자 기능
- `admin.py`로 관리

### APP 관리자 설정
- 기본 사이트 기능 제공
- 커스텀 가능

```py
from django.contrib import admin
from .models import Article

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ("title", "created_at")
    search_fields = ("title", "content")
    list_filter = ("created_at",)
    date_hierarchy = "created_at"
    ordering = ("-created_at",)
```