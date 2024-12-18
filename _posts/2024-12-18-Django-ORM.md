---
layout: post
title: Django ORM
subtitle: TIL Day 87
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django ORM
> **ORM** : 객체 지향 프로그래밍 언어를 사용하여 호환되지 않는 유형의 시스템 간에 데이터를 변환하는 프로그래밍 기술

## 환경 설정
- `pip install django-extensions`
  - `settings.py` : `INSTALLED_APPS`에 등록
- `models.py` : 사용할 데이터 생성
- `python3 manage.py seed products --number=30`
  - 샘플 데이터 30개 생성
- `python3 manage.py shell_plus` : 데이터 베이스 접속

```py 
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Product(models.Model):
    CATEGORY_CHOICES = (
        ("F", "Fruit"),
        ("V", "Vegetable"),
        ("M", "Meat"),
        ("O", "Other"),
    )

    name = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    quantity = models.PositiveIntegerField()
    category = models.CharField(max_length=1, choices=CATEGORY_CHOICES)

    def __str__(self):
        return self.name
```

## ORM 활용

### Q()
- **SQL**의 **WHERE** 절에 해당하는 기능 활용
  - `&` : **AND**
  - `|` : **OR**
  - `~` : **NOT**
- 형태 : `.objects.filter(Q(조건))`

```py
# 가격이 15000보다 크거나 수량이 3000보다 적은 제품
Product.objects.filter(
	Q(price__gt=15000) | Q(quantity__lt=3000)
)
```

### F()
- 필드 값에 의존하는 작업
  - 필드의 값을 가져오거나 업데이트해서 값을 참조
  - **Python** 메모리에서 작업하지 않고 데이터베이스에서 작업을 수행해서 값을 가져옴

```py
# 모든 제품의 가격을 1000원 인상
Product.objects.update(price = F('price') + 1000)
```

### annotate()
- 각 객체별로 계산된 값을 추가

```py
# 제품별 total_price
products = Product.objects.annotate(
    total_price=F('price') * F('quantity')
)
```

### aggregate()
- 전체 쿼리 집계
  - **Avg, Sum, Count** 등 집계함수 사용

```py
# 디폴트 명칭 '__' 구조
Product.objects.aggregate(Avg('price'))
# {'price__avg': 14627.76}

Product.objects.aggregate(my_avg = Avg('price'))
# {'my_avg': 14627.76}
```

### Group By 적용
- 그룹화를 할 기준이 되는 값을 먼저 호출
- 그룹에 들어갈 값을 호출

```py
Product.objects.values('category')
#
<QuerySet [{'category': 'M'}, {'category': 'O'}, 
{'category': 'V'}, {'category': 'M'}, ...
{'category': 'F'}, '...(remaining elements truncated)...']>

Product.objects.values('category').annotate(category_count = Count('category'))
# 
{'category': 'F', 'category__count': 15}, 
{'category': 'M', 'category__count': 15}, 
{'category': 'O', 'category__count': 15}, 
{'category': 'V', 'category__count': 5}
```

### Raw()
- 직접 **SQL**문 입력

```py
categories_count = Product.objects.raw(
'''
SELECT "id", "category", COUNT("category") AS "category_count" 
FROM "products_product" 
GROUP BY "category"
'''
)
# ORM 입력으로는 까다로운 그룹화를 직접 실행

for each in categories_count:
	print(each.category_count, each.category)

# 15 F 15 M 15 O 5 V
```

### Database Connection
- 데이터베이스에 직접 연결한 뒤 쿼리를 전송

```py
from django.db import connection

sql_query = '''
SELECT "category", COUNT("category") AS "category_count" 
FROM "products_product" 
GROUP BY "category"
'''
# 사용할 쿼리

cursor = connection.cursor() 
# 데이터베이스 연결
cursor.execute(sql_query)
# 쿼리 실행
result = cursor.fetchall()
# 실행 결과 모두 가져오기
print(result)

# [('F', 15), ('M', 15), ('O', 15), ('V', 5)]
```

## ORM 최적화

### 필요성
- **Lazy Loading**
  - 지연로딩 : 작성 즉시가 아닌 데이터가 실제로 사용될 때 쿼리를 진행
    - 불필요한 데이터베이스 쿼리를 방지
    - 필요한 경우에만 쿼리 : 메모리 절약
- **N+1 Problem**
  - 필요한 데이터를 한번에 가져오지 않음
  - 관련된 객체를 조회하기 위해 **N**개의 추가 쿼리가 발생하고 실행

```py
comments = Comment.objects.all()
# 쿼리가 발생하지 않음
for comment in comments:
# 쿼리 발생 comments 조회
	print(f"{comment.id}의 글제목")
  # 위에서 조회된 데이터 사용
	print(f"{comment.article.title}")
  # article을 조회하기 위해 comments 개수만큼 쿼리 발생
```

### Eager Loading
- 즉시로딩
  - 데이터를 로드할 때 필요하다고 판단되는 연관된 데이터 객체들을 한 번에 가져옴
  - 너무 많은 데이터를 가져오면 오히려 성능 저하
- `select_related`
  - **one-to-many** 또는 **one-to-one** 관계
  - **SQL**의 **JOIN**을 이용해서 관련된 객체들을 한 번에 로드
- `prefetch_related`
  - **many-to-many** 또는 역참조 관계
  - 첫번째 쿼리는 원래 객체 조회
  - 번째 쿼리는 연관된 객체 조회

```py
# select_related
@api_view(["GET"])
def check_sql(request):
    from django.db import connection

    comments = Comment.objects.all().select_related("article")
    # comments 와 article 의 join된 데이터를 모두 가져옴
    for comment in comments:
        print(comment.article.title)

    print("-" * 30)
    print(connection.queries)

    return Response()

# prefetch_related
articles = Article.objects.all().prefetch_related("comments")
for article in articles:
  # 첫 번째 쿼리
  comments = article.comments.all()
  for comment in comments:
  # 두 번째 쿼리
      print(comment.id)
```

### django-silk
- 요청에 대한 다양한 실시간 정보 제공
  - 로직을 분석
  - 내부적으로 사용하는 쿼리 확인
- `pip install django-silk`

#### `settings.py`
- `MIDDLEWARE` : 요청을 중간에 가로채서 사용

```py
MIDDLEWARE = [
    ...
    'silk.middleware.SilkyMiddleware',
    ...
]

INSTALLED_APPS = (
    ...
    "silk",
)
```

#### `urls.py`
- 프로젝트 디렉토리의 `urls.py`에 추가

```py
urlpatterns += [path('silk/', include('silk.urls', namespace='silk'))]
```