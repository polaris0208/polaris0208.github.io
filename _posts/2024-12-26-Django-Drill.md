---
layout: post
title: Django Drill
subtitle: TIL Day 95
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

## INDEX
### 1. 프로젝트 개요 [¶](#프로젝트-개요)
### 2. App 설명 [¶](#app-설명)

<hr>

# 프로젝트 개요
> **Django** 프레임워크를 이용한 상품 페이지 구현

### 프로젝트 실행

#### Docker 사용
- 자동화
  - `migration`
  - `superuser` 생성
  - `seed` 생성
  - `runserver` 진행

```bash
git clone https://github.com/polaris0208/django_assignment
docker-compose up --build
```

#### Python 사용
- `Mac OS` : `python3` 시도

```bash
git clone https://github.com/polaris0208/django_assignment
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

### 구현 기능

#### users [¶](#users)
> 사용자 관련 기능
- 프로필 기능
  - 프로필 사진 변경
  - 팔로우 기능
  - 좋아요/찜한 삼품 목록

#### accounts [¶](#accounts)
> 계정 관련 기능
- 로그인/로그아웃 기능
- 회원 정보 기능
  - 회원가입/탈퇴 기능
  - 회원정보 수정/비밀번호 변경 기능

#### products [¶](#products)
> 상품 관련 기능
- 상품 관리 기능
  - 상품 등록/수정/삭제 기능
  - 좋아요/찜 기능
  - 해시태그 기능
- 상품 조회 기능
  - 검색 기능
  - 정렬 기능
  - 해시태그로 조회 기능

### 프로젝트 구조

```
django_assignment/
│
├── README.md : 프로젝트 설명
├── requirements.txt : 의존성 목록
├── .gitignore : 버전관리 제외 목록
├── .dockerignore : 도커 실행 시 제외 목록
├── Dockerfile : 컨테이너 생성 설정
├── docker-compose.yml : 컨테이너 실행 설정
├── .github/workflows/ : CI/CD 경로
│
├── manage.py : 프로젝트 관리 파일
├── spartamarket/ : 프로젝트 앱
├── static/css : 정적 자원 경로
├── media/ : 동적 자원 경로
├── templates : 프로젝트 템플릿
│
├── homepage/ : 홈페이지 앱
├── users/ : 사용자 앱
├── accounts/ : 계정 앱
└── products/ : 상품 앱
```

### ERD
![ERD](/assets/img/ERD.png)
- **User**
  - **User ↔ Products**
    - **1:N** : 하나의 사용자에 여러 상품이 있을 수 있음
  - **User ↔ Products** 좋아요/찜
    - **M:N** : 사용자가 여러 상품을 좋아요할 수 있음
  - **User ↔ Comment**
    - **1:N** : 하나의 사용자에 여러 댓글이 있을 수 있음
  - **User ↔ User** 팔로우/팔로워
    - **M:N** : 사용자가 다른 사용자들을 팔로우할 수 있고, 다른 사용자는 그들을 팔로우할 수 있음

- **Products**
  - **Products ↔ Comment**
    - **1:N** : 하나의 상품에 여러 댓글이 있을 수 있음
  - **Products ↔ HashTag**
    - **M:N** : 여러 해시태그와 여러 상품이 연결될 수 있음

## 프로젝트 진행 과정
![진행 과정](/assets/img/log.png)

## 프로젝트 기본 설정

### `settings.py`
- 앱 등록
- 사용자 모델 설정
- 언어/시간 설정
- 템플릿 및 자원 경로 설정

```py
...
INSTALLED_APPS = [
    ...
    # third_party
    'django_seed',
    # local apps
    'users',
    'homepage',
    'accounts',
    'products',
]
...
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],
        ...
    },
]
...
AUTH_USER_MODEL = "users.User"
...
LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'
...
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"  
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

### `urls.py`
- 관리자 페이지 경로 설정
- 앱 `urls` 연결
- 동적 자원 경로 설정 : 개발 과정에서 사용

```py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
]

urlpatterns += [
  path('users/', include('users.urls'))
]

urlpatterns += [
  path('home/', include('homepage.urls'))
]

urlpatterns += [
  path('accounts/', include('accounts.urls'))
]

urlpatterns += [
  path('products/', include('products.urls'))
]

# 개발 환경에서 MEDIA 파일 관리
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
        )
```

<hr>

# App 설명

## users
> [¶](#구현-기능) 사용자 모델 정의, 프로필 기능

### `models.py`
- 사용자 모델 정의
  - 프로필 이미지
  - 팔로우 기능 : 대칭 기능 제거(자동으로 맞팔로우 기능)
  - 팔로우/팔로잉 계산 기능

```py
from django.db import models
from django.contrib.auth.models import AbstractUser

def user_profile_image_path(instance, filename):
    return f"profile/{instance.username}/{filename}"


class User(AbstractUser):
    profile_image = models.ImageField(
        default='profile/default.png',
        upload_to=user_profile_image_path, blank=True, null=True
    )
    following = models.ManyToManyField(
        "self",
        related_name="followers",
        symmetrical=False,
        blank=True,
    )  # symmetrical 대칭 기능

    @property
    def follower_counter(self):
        return self.followers.count()  # 역참조 해서 확인 / 나를 팔로우

    @property
    def following_counter(self):
        return self.following.count()  # 정참조 해서 확인 / 내가 팔로우

    def __str__(self):
        return self.username
```

### `views.py`
- `memeber` : 서버에 등록된 사용자
- `liked_products` : 해당 사용자가 좋아요/찜한 물건
- `def follow(request, user_id):` : 팔로우 기능
  - `if request.user.is_authenticated:` : 로그인 필요
  - `if member != request.user:` : 자신은 팔로우 불가
  - `if member.followers.filter(pk=request.user.pk).exists():`
    - 팔로우가 존재하는 지 확인
    - 존재하는 경우 제거
    - 없는 경우 추가

```py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import get_user_model
from django.views.decorators.http import require_POST
from products.models import Products

def profile(request, username):
    member = get_object_or_404(get_user_model(), username=username)
    context = {
        "member": member,
        'follower_count': member.follower_counter,  # 팔로워 수
        'following_count': member.following_counter,
        'liked_products' : Products.objects.filter(like_user=member)
        }
    return render(request, "users/profile.html", context)

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

### `urls.py`
- `username` 기준으로 프로필 접근
- `user_id` 기준으로 팔로우 기능 사용

```py
from django.urls import path
from . import views

app_name = 'users'
urlpatterns = [
    path('profile/<str:username>/', views.profile, name='profile'),
    path('<int:user_id>/follow/', views.follow, name='follow'),
]
```

### INDEX [¶](#index)
- 구현 기능 항목으로 이동

<hr>

## accounts
> 회원 기능, 로그인/로그아웃 기능

### `forms.py`
- `get_user_model()` : 사용자 모델 가져오기
- `def __init__(self, *args, **kwargs):`
  - 비밀번호 변경 `url` 설정 변경

```py
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth import get_user_model
from django.urls import reverse


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = UserCreationForm.Meta.fields


class CustomUserChangeForm(UserChangeForm):
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
            password_help_text = ('<a href="{}">비밀번호 변경</a>.').format(
                f"{reverse('accounts:change-password')}"
            )
            self.fields["password"].help_text = password_help_text
```

### `views.py`
- `auth_login, auth_logout` : 로그인/로그아웃 함수와 혼동을 피하기 위해 설정
- `update_session_auth_hash` : 변경한 비밀번호를 가지고 다시 로그인

```py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from .forms import CustomUserCreationForm, CustomUserChangeForm
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.views.decorators.http import require_http_methods, require_POST
```

#### 로그인/로그아웃
- `next_url = request.GET.get("next")`
  - 로그인 요구을 받기 전에 가려던 페이지로 이동

```py
@require_http_methods(["GET", "POST"])
def login(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            next_url = request.GET.get("next") or "homepage:homepage"
            return redirect(next_url)
    else:
        form = AuthenticationForm()
        context = {"login_form": form}
        return render(request, "accounts/login.html", context)


@require_POST
def logout(request):
    auth_logout(request)
    return redirect("accounts:login")
```

#### 회원 관리 기능
- `def resign(request):`
  - `auth_logout()` : 사용자를 삭제해도 쿠키가 남기 때문에 로그아웃으로 삭제 처리
- `def update(request):`
  - `instance=request.user` : `form`에 사용자의 원래 정보를 담아서 가져오기

```py
@require_http_methods(["GET", "POST"])
def signin(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("homepage:homepage")
    else:
        form = CustomUserCreationForm()
        context = {"sign_form": form}
        return render(request, "accounts/signin.html", context)


@require_POST
def resign(request):
    if request.user.is_authenticated:
        request.user.delete()
        auth_logout()
    return redirect("homepage:homepage")


@require_http_methods(["GET", "POST"])
def update(request):
    if request.method == "POST":
        form = CustomUserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("homepage:homepage")
    else:
        form = form = CustomUserChangeForm(instance=request.user)

    context = {"form": form}
    return render(request, "accounts/update.html", context)


@require_http_methods(["GET", "POST"])
def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            return redirect("homepage:homepage")
    else:
        form = PasswordChangeForm(request.user)

    context = {"form": form}
    return render(request, "accounts/change-password.html", context)
```

### `urls.py`

```py
from django.urls import path
from . import views

app_name = 'accounts'
urlpatterns = [
  path('login/', views.login, name='login'),
  path('logout/', views.logout, name='logout'),
  path('signin/', views.signin, name='signin'),
  path('resign/', views.resign, name='resign'),
  path('update/', views.update, name='update'),
  path('change-password/', views.change_password, name='change-password'),
]
```

### INDEX [¶](#index)
- 구현 기능 항목으로 이동

<hr>

## products
> 상품 조회/관리 기능

### `models.py`
- `def products_image_path(instance, filename):`
  - 이미지 경로 생성 함수
  - 콜백 함수로 사용
- `def validation_hashtag(value):` 
  - 해시 태그 정당성 검사 함수
  - 해시 태그 필드 생성시 추가

```py
from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
import re


def products_image_path(instance, filename):
    return f"products/{instance.user.username}/{filename}"


def validation_hashtag(value):
    if not re.match(r"^[0-9a-zA-Z_]+$", value):
        # ^: 시작 / $ : 특정 패턴 끝
        raise ValidationError


class HashTag(models.Model):
    name = models.CharField(max_length=50, unique=True, validators=[validation_hashtag])

    def __str__(self):
        return f"#{self.name}"
```

#### 상품 모델
- `def like_user_counter(self):` : 좋아요/찜 계산 함수
- `def view_counter(self):` : 조회수 계산 함수

```py
class Products(models.Model):
    # 제목
    title = models.CharField(max_length=50)
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="products"
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # 상품 설명
    product_name = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    quantity = models.PositiveIntegerField()
    image = models.ImageField(upload_to=products_image_path, blank=True, null=True)
    # 좋아요/찜
    like_user = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name="like_products",
        blank=True
    )

    # 해쉬태그
    hashtags = models.ManyToManyField(HashTag, related_name='products', blank=True)
    # 조회수
    views = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.title
    
    @property
    def like_user_counter(self):
        return self.like_user.count()
    
    def view_counter(self):
        self.views = self.views + 1
        self.save()
        return self.views
```

#### 댓글 모델
- `CASCADE` : 상품이 삭제되면 같이 삭제
- `on_delete=models.CASCADE` : 사용자가 삭제되면 같이 삭제

```py
class Comment(models.Model):
    products = models.ForeignKey(
        Products, on_delete=models.CASCADE, related_name="comments"
    )
    # CASCADE 참조하는 데이터가 삭제되면 같이 삭제
    content = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    #
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="comments"
    )

    def __str__(self):
        return self.content
```

### `forms.py`
- `hashtags_str = forms.CharField(required=False)`
  - 해시태그 입력 필드 정의
- `self.user = kwargs.pop("user", None)`
  - 해시태그 생성을 위해 사용자 추출
- `exclude = ("author", "like_user", "views", "hashtags")`
  - 입력 양식에서 제외할 필드
  - `author` : 사용자로 자동 등록
  - `like_user, views` : 별도의 로직으로 작동
  - `hashtags` : `hashtags_str` 를 통해 작성
- `def save(self, commit=True):`
  - `commit=False` : **DB** 에 바로 반영하지 않고 별도의 작업 진행
  - 추출한 사용자를 작성자로 등록
  - 입력한 해시태그를 전처리하여 등록
  - 작업이 종료되면 **DB**에 반영

```py
from django import forms
from .models import Products, Comment, HashTag


class ProductsForm(forms.ModelForm):
    hashtags_str = forms.CharField(required=False)

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)

    class Meta:
        model = Products
        fields = "__all__"
        exclude = ("author", "like_user", "views", "hashtags")

    def save(self, commit=True):
        product = super().save(commit=False)

        if self.user:
            product.user = self.user
            product.author = self.user

        if commit:
            product.save()

        hashtags_input = self.cleaned_data.get("hashtags_str", "")
        hashtag_list = [h for h in hashtags_input.replace(",", " ").split() if h]
        new_hashtags = []
        for ht in hashtag_list:
            ht_obj, created = HashTag.objects.get_or_create(name=ht)
            new_hashtags.append(ht_obj)
        product.hashtags.set(new_hashtags)

        if not commit:
            product.save()

        return product
```

#### 댓글 기능
- `exclude = ("products", "user", "product_name")`
  - `user` : 자동 등록
  - `products, product_name` : 본문 제외하고 댓글만 가져옴

```py
class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = "__all__"
        exclude = ("products", "user", "product_name")
```

#### 검색 기능

```py
class SearchForm(forms.Form):
    search_word = forms.CharField(label="Search Word")
```

### `views.py`

```py
from django.shortcuts import render, redirect, get_object_or_404
from .models import Products, Comment, HashTag
from .forms import ProductsForm, CommentForm, SearchForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_http_methods
from django.views.generic import FormView
from django.db.models import Q
from django.db.models import Count
```

#### 전체 조회/정렬 기능
- `annotate` : **ORM** 집계함수 실행
- `sort = request.GET.get('sort', 'date') `
  - 쿼리 스트링 사용
  - `products/?sort=likes/` : 좋아요/찜 기준으로 정렬

```py
def products(request):
    sort = request.GET.get('sort', 'date')  # 기본값은 날짜순으로 설정
    if sort == 'likes':
        products = Products.objects.annotate(like_count=Count('like_user')).order_by('-like_count', '-created_at')  # 좋아요 순으로 정렬
    elif sort == 'comments':
        products = Products.objects.annotate(comment_count=Count('comments')).order_by('-comment_count', '-created_at')  # 댓글 순으로 정렬
    else:  
        products = Products.objects.all().order_by('-created_at')  # 날짜 순으로 정렬

    context = {
        'products': products,
    }
    return render(request, "products/products.html", context)
```

#### 상품 CRUD 기능
- 상세 확인
  - 댓글 데이터/댓글 작성 양식을 포함해 전달
- 생성
  - `request.FILES, user=request.user`
  - 상품 이미지 저장을 위해 파일 전달
  - 작성자, 해시태그 저장을 위해 사용자 전달

```py
def detail(request, pk):
    products = get_object_or_404(Products, pk=pk)
    comment_form = CommentForm()
    comments = products.comments.all().order_by("-pk")
    context = {
        "products": products,
        "comment_form": comment_form,
        "comments": comments,
    }
    return render(request, "products/detail.html", context)


@login_required
@require_http_methods(["GET", "POST"])
def create(request):
    if request.method == "POST":
        form = ProductsForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            form.save()
            # return redirect("products:detail", products.pk)
            return redirect("products:products")
    else:
        form = ProductsForm()

    context = {"form": form}
    return render(request, "products/create.html", context)


@login_required
@require_POST
def delete(request, pk):
    if request.user.is_authenticated:
        products = get_object_or_404(Products, pk=pk)
        products.delete()
    return redirect("products:products")


@login_required
@require_http_methods(["GET", "POST"])
def update(request, pk):
    products = get_object_or_404(Products, pk=pk)
    if request.method == "POST":
        form = ProductsForm(request.POST, instance=products)
        if form.is_valid():
            form.save()
            return redirect("products:detail", products.pk)
    else:
        form = ProductsForm(instance=products)

    context = {"form": form, "products": products}
    return render(request, "products/update.html", context)
```

#### 좋아요/찜 기능
- `if request.user.is_authenticated:` : 로그인 필요
- `if products.like_user.filter(pk=request.user.pk).exists():`
  - 좋아요/찜 있으면 삭제
  - 없으면 추가

```py
@login_required
@require_POST
def like(request, pk):
    if request.user.is_authenticated:
        products = get_object_or_404(Products, pk=pk)
        if products.like_user.filter(pk=request.user.pk).exists():
            products.like_user.remove(request.user)
        else:
            products.like_user.add(request.user)
        return redirect("products:products")
    else:
        return redirect("accounts:login")
```

#### 댓글 기능
- `commit=False` : **DB**에 바로 바로 반영하지 않고 별도의 작업 진행
  - 상품 정보, 사용자 정보 저장
  - 작업이 종료되면  **DB** 반영

```py
@login_required
@require_POST
def comment_create(request, pk):
    products = get_object_or_404(Products, pk=pk)
    form = CommentForm(request.POST)
    if form.is_valid():
        comment = form.save(commit=False)
        comment.products = products
        comment.user = request.user
        comment.save()
        return redirect("products:detail", products.pk)


@login_required
@require_POST
def comment_delete(request, pk, comment_pk):
    if request.user.is_authenticated:
        comment = get_object_or_404(Comment, pk=comment_pk)
        if comment.user == request.user:
            comment.delete()
    return redirect("products:detail", pk)
```

#### 해시태그별 상품 조회

```py
@login_required
def hashtag(request, hash_pk):
    hashtag = get_object_or_404(HashTag, pk=hash_pk)
    products = hashtag.products.order_by("-pk")
    context = {
        "hashtag": hashtag,
        "products": products,
    }
    return render(request, "products/hashtag.html", context)
```

#### 검색 기능
- **ORM** 검색 기능 활용
- `Q` : 쿼리 사용
- `|` : **OR** 조건 사용

```py
class SearchFormView(FormView):
    form_class = SearchForm
    template_name = "products/search.html"

    def form_valid(self, form):
        searchWord = form.cleaned_data["search_word"]
        products_list = Products.objects.filter(
            Q(title__icontains=searchWord)
            | Q(product_name__icontains=searchWord)
            | Q(content__icontains=searchWord)
        ).distinct()

        context = {
            "form": form,
            "searchWord": searchWord,
            "products_list": products_list,
        }

        return render(self.request, "products/search.html", context)
```

### `products/templates/make_link.py`
- 해시태그에서 필요한 데이터를 추출하여 링크 생성
- 템플렛에서 `tag|hashtag_link|safe` 형태로 사용

```py
from django import template

register = template.Library()

@register.filter
def hashtag_link(hashtag):
    tag_link = f'<a href="/products/{hashtag.pk}/hashtag/">{hashtag.name}</a>'
    return tag_link
```

### `urls.py`

```py
from django.urls import path
from . import views

app_name = "products"
urlpatterns = [
    path("", views.products, name="products"),
    path("create/", views.create, name="create"),
    path("<int:pk>/", views.detail, name="detail"),
    path("<int:pk>/delete/", views.delete, name="delete"),
    path("<int:pk>/update/", views.update, name="update"),
    path("<int:pk>/like/", views.like, name="like"),
]

urlpatterns += [
    path("<int:pk>/comments/", views.comment_create, name="comment_create"),
    path(
        "<int:pk>/comments/<int:comment_pk>/delete/",
        views.comment_delete,
        name="comment_delete",
    ),
]

urlpatterns += [
    path("<int:hash_pk>/hashtag/", views.hashtag, name="hashtag"),
]

urlpatterns += [
  path('search/', views.SearchFormView.as_view(), name='search'),
]
```

### INDEX [¶](#index)
- 구현 기능 항목으로 이동