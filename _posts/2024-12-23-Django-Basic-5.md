---
layout: post
title: Django Basic 5
subtitle: TIL Day 92
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django Basic 5

## 언어/시간 설정
- `settings.py`
  - `LANGUAGE_CODE`
  - `TIME_ZONE` 

```py
...
LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'
...
```

## 콜백 함수
- 다른 함수에 인수로 전달되어 나중에 동적으로 호출되는 함수
  - `()`를 붙이면 결과를 전달 : 정적인 호출
- `user_profile_image_path` 함수가 호출되어 저장 경로가 동적으로 생성
  - 함수 객체 자체를 참조 : 동적으로 경로를 생성
  - 필요할 때 **Django**가 직접 호출

```py
def user_profile_image_path(instance, filename):
    return f"profile/{instance.username}/{filename}"


class User(AbstractUser):
    profile_image = models.ImageField(
        upload_to=user_profile_image_path, blank=True, null=True
    )
...
```

## @property
- 계산된 필드
  - 모델의 필드 값을 기반으로 계산된 데이터를 동적으로 반환
  - 데이터베이스에는 저장되지 않고, 객체에만 존재하는 속성
- 읽기 전용 속성
  - 값을 단순히 조회만 가능하게 설정하고 싶을 때 사용

```py
# models.py
class User(AbstractUser):
    ...
    following = models.ManyToManyField(
        "self",
        related_name="followers",
        symmetrical=False,
        blank=True,
    )
    @property
    def follower_counter(self):
        return self.followers.count()  # 역참조 해서 확인 / 나를 팔로우

    @property
    def following_counter(self):
        return self.following.count()  # 정참조 해서 확인 / 내가 팔로우

    def __str__(self):
        return self.username

# views.py
def profile(request, username):
    member = get_object_or_404(get_user_model(), username=username)
    context = {
        "member": member,
        'follower_count': member.follower_counter,
        'following_count': member.following_counter,
        }
    return render(request, "users/profile.html", context)
```

## Hashtag

### urls.py

```py
urlpatterns += [
    path("<int:hash_pk>/hashtag/", views.hashtag, name="hashtag"),
]
```

### models.py
- `ManyToManyField` : **N : M** 관계

```py
class Hashtag(models.Model):
    content = models.TextField(unique=True)  # 중복 방지
    def __str__(self):
        return self.content

class Products(models.Model):
    ...
    hashtags = models.ManyToManyField(Hashtag, blank=True)
    ...
```

### views.py
- `content`를 공백기준 리스트로 변경
- `#` 로 시작하는 요소 선택
- `get_or_create` : `created` 값이 `True`인 경우 새로 생성된 것

```py
@login_required
@require_http_methods(["GET", "POST"])
def create(request):
    ...
            # 해쉬태그 기능 추가
            for word in products.content.split():  
                if word.startswith("#"):  
                    hashtag, created = Hashtag.objects.get_or_create(
                        content=word
                        )
                    products.hashtags.add(hashtag)
    ...  

@login_required
@require_http_methods(["GET", "POST"])
def update(request, pk):
    ...
            # 해쉬태그 기능
            products.hashtags.clear()  # 기존에 있던 hashtags 삭제
            for word in products.content.split():
                if word.startswith('#'):
                    hashtag, created = Hashtag.objects.get_or_create(
                        content=word
                        )
                    products.hashtags.add(hashtag)
    ...

@login_required
def hashtag(request, hash_pk):
    hashtag = get_object_or_404(Hashtag, pk=hash_pk)
    products = hashtag.products_set.order_by("-pk")
    context = {
        "hashtag": hashtag,
        "products": products,
    }
    return render(request, "products/hashtag.html", context)
```

### templatetags
- `load <링크 파일명>` : 템플릿에서 선언 후 호출
  - `<데이터 테이블>|<링크명>|<옵션>`
- `products/templatetags/make_link.py`
- `+ " "` : 비슷한 다른 태그와 구별하기 위해 공백 추가
  - 예 : `#Python, #PythonLibary`

```py
from django import template

register = template.Library()

@register.filter
def hashtag_link(word):
    content = word.content + " "
    hashtags = word.hashtags.all()
    for hashtag in hashtags:
        content = content.replace(
            hashtag.content + " ",
            f'<a href="/products/{hashtag.pk}/hashtag/">{hashtag.content}</a> ',
        )
    return content
```