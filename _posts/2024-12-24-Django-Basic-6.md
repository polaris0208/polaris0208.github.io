---
layout: post
title: Django Basic 6
subtitle: TIL Day 93
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic 6

## 해쉬태그 기능 개선

### `models.py`
- `validation` : 정당성 평가 도구 작성
- `^...+$` : 범위 설정
- `re.match(r"^[0-9a-zA-Z_]+$", value)`: 숫자, 알파벳, 공백만 허용

```py
def validation_hashtag(value):
    if not re.match(r"^[0-9a-zA-Z_]+$", value):
        # ^: 시작 / $ : 특정 패턴 끝
        raise ValidationError


class HashTag(models.Model):
    name = models.CharField(max_length=50, unique=True, validators=[validation_hashtag])

    def __str__(self):
        return f"#{self.name}"

class Products(models.Model):
    ...
    hashtags = models.ManyToManyField(HashTag, related_name='products', blank=True)
    ...
```

### `forms.py`
- `hashtags_str = forms.CharField(required=False)` : 문자열을 입력받는 선택적 필드
- `def __init__(self, *args, **kwargs):` : `user` 추출
- `def save(self, commit=True):` : 사용자 정의 저장방식 사용
  - `product = super().save(commit=False)` : **DB**에 바로 반여하지 않고 별도의 처리 진행
  - `hashtags_input` : `hashtags_str` 필드에서 입력된 해시태그 문자열을 가져옴
  - `hashtag_list` : 쉼표나 공백을 기준으로 해시태그 목록
  - `for ht in hashtag_list:` : 각각의 해시태그에 대해 존재하지 않으면 새로 생성
- `product.hashtags.set(new_hashtags)` : 해쉬태그 설정

```py
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

        # 사용자가 제공된 경우, 'user' 값을 'product'의 'user'와 'author' 필드에 할당
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

### `templatetags/make_link.py` 수정
- 각각의 해쉬태그에서 링크 생성
  - 템플릿 : `tag|hashtag_link|safe`
```py
from django import template

register = template.Library()

@register.filter
def hashtag_link(hashtag):
    tag_link = f'<a href="/products/{hashtag.pk}/hashtag/">{hashtag.name}</a>'
    return tag_link
```

## 검색 기능

### `urls.py`

```py
urlpatterns += [
  path('search/', views.SearchFormView.as_view(), name='search'),
]
```

### `views.py`
- **ORM**을 사용하여 검색

```py
from django.views.generic import FormView
from django.db.models import Q

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

## 정렬 기능
- 상품목록 기능을 수정
- `href="?sort=date"`: 상품목록 템플릿에서 쿼리스트링을 추가하여 작동

```py
from django.db.models import Count

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

## Docker-Compose
- 테스트를 위한 세팅 자동화
- 현재 프로젝트를 바탕으로 컨테이너 생성
- 관리자 계정 생성
- 테스트를 위한 **Seeding**

### Dockerfile

```yml
# Python 3.9 slim 이미지를 사용하여 기본 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 8000을 노출
EXPOSE 8000
```

### docker-compose.yml
- `environment`: 관리자 계정에 빌요한 변수를 환경변수에서 가져옴
- `sh -c` : 복수의 명령어 실행
  - `&&` : 연결
  - `--noinput || true` : 입력이 없는 경우 `true`로 간주하고 넘어감

```yml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DJANGO_SUPERUSER_USERNAME: admin
      DJANGO_SUPERUSER_EMAIL: admin@example.com
      DJANGO_SUPERUSER_PASSWORD: password
    command: >
      sh -c "
      python manage.py makemigrations &&
      python manage.py migrate &&
      python manage.py createsuperuser --noinput || true &&
      python manage.py seed products --number=30 --seeder 'Products.author_id' 1 &&
      python manage.py runserver 0.0.0.0:8000
      "
```

### 실행
- `docker-compose up --build`
  - `-d` : 터미널 밖에서 작동