---
layout: post
title: DRF Spectacular
subtitle: TIL Day 98
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# DRF-Spectacular
> **OpenAPI** 3 기반 **API** 문서 자동화 라이브러리

## 설정

### 라이브러리 설치

```bash
pip install drf-spectacular
```

### `settings.py` 설정

#### 기본 설정

```py
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'drf_spectacular',
    # ...
]

REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}
```

#### 문서 설명 기입

```py
SPECTACULAR_SETTINGS = {
    'TITLE': 'API Title',
    'DESCRIPTION': 'API description',
    'VERSION': '1.0.0',
}
```

#### 연락처

```py
SPECTACULAR_SETTINGS = {
    'CONTACT': {
        'name': 'example',
        'email': 'examplev@example.com',
        'url': 'https://www.example.com',
    }
}
```

#### 라이선스

```py
SPECTACULAR_SETTINGS = {
    'LICENSE': {
        'name': 'MIT License',
        'url': 'https://opensource.org/licenses/MIT',
    }
}
```

#### 파일 업로드 허용
- 웹 인터페이스에서 테스트시

```py
SPECTACULAR_SETTINGS = {
    'COMPONENT_SPLIT_REQUEST': True,
}
```

### JWT 인증 설정

```py
SPECTACULAR_SETTINGS = {
    ...
    'SECURITY': [{
        'BearerAuth': {
            'type': 'http',
            'scheme': 'bearer',
            'bearerFormat': 'JWT',
        }
    }]
    ...
}
출처: https://devspoon.tistory.com/256 [devspoon 오픈소스 개발자 번뇌 일지:티스토리]
```

### `urls.py`
- `api/schema/swagger-ui/'`로 접속하여 웹인터페이스 사용

```py
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

urlpatterns = [
    # YOUR PATTERNS
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    # Optional UI:
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
```

### `@extend_schema`
- 문서화 커스텀마이즈
  - `parameters` : `path`를 통해 받을 파라미터
  - `request` : 요청 형태
  - `responses` : 응답 형태
  - `auth` : 인증방법
  - `description`: 설명
  - `summary` : 요약
  - `deprecated` : 
  - `tags` : 분류
  - `exclude` : **API** 문서에서 제외 여부
  - `methods` : `Http method` 목록
  - `examples` : 요청/응답에 대한 예시

#### `parameters`
- `OpenApiParameter`로 커스텀 가능

```py
from drf_spectacular.utils import extend_schema, OpenApiParameter

@extend_schema(
    tags=["사용자"],
    parameters=[
        OpenApiParameter(
            name="email", description="계정", required=True, type=str
        ),
        OpenApiParameter(
            name="password", description="암호", required=True, type=str
        ),
    ],
)
```

#### `request`
- `serializer` 지정
  - `GenericAPIView` 이상부터 가능
  - `APIView`에 사용 불가
- `inline_serializer`
  - `APIView`에서 사용 가능

```py
class TistoryView(GenericAPIView):
    serializer_class = LoginSerializer

    @extend_schema(
        request=LoginSerializer,
    )
    def Login(self, request, *args, **kwargs):
```

```py
from drf_spectacular.utils import extend_schema, inline_serializer
from rest_framework import serializers, views

@extend_schema(request=inline_serializer(
    name='LoginSerializer',
    fields={
        'email': serializers.EmailField(),
        'password': serializers.CharField(),
    }
))
class Login(views.APIView):
```

#### `response`
- `status_code : response`의 딕셔너리 형태
  - `status_code`
    - 응답코드
    - 키값으로 사용
    - 중복 사용 불가능
  - `response`
    - 응답
    - 스키마 또는 직렬화 도구로 응답
  - `OpenApiResponse`로 커스텀 가능

```py
from drf_spectacular.utils import extend_schema, OpenApiResponse
from rest_framework import serializers, views

class ExampleSerializer(serializers.Serializer):
    message = serializers.CharField()

@extend_schema(
    responses={
        200: OpenApiResponse(response=ExampleSerializer, description='성공 응답'),
        400: OpenApiResponse(description='잘못된 요청 처리')
    }
)
class ExampleView(views.APIView):
    pass
```

#### `examples`

- `OpenApiExample`로 커스텀
  - `name`: 예시 이름
  - `summary`: 설명
  - `description`: 자세한 설명
  - `value`: 실제 값 `dict` 주로 사용
  - `request_only`: 요청에만 사용
  - `response_only`: 응답에만 사용
  - `status_code`: 상태 코드를 명시, 응답 예제에 주로 사용

## 파일 업로드
- `settings.py` : `COMPONENT_SPLIT_REQUEST: True` 설정 필요

```py
class FileUploadView(APIView):
    parser_classes = (MultiPartParser,)
    @extend_schema(
        request=inline_serializer(
            name="upload example",
            fields={
                "file": serializers.FileField(),
            },
        )
    )
    def post(self, request, *args, **kwargs):
```

### 파일로 분리해서 사용

#### custom_decorators.py

```py
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes, OpenApiExample

def custom_extend_schema(_func=None, **kwargs):
    def decorator(func):
        return extend_schema(
            tags=["사용자"],
            summary="사용자 정의 데코레터",
            parameters=[
                OpenApiParameter(
                    name="파라미터", 
                    description="쿼리 파라미터", 
                    required=False, 
                    type=str,
                ),
            ],
            **kwargs  #데코레이터 호출 후 추가할 인자
        )(func)
    return decorator if _func is None else decorator(_func)
    # 인자가 있든 없든 작동
```

#### `views.py`

```py
from rest_framework.views import APIView
from rest_framework.response import Response
from .custom_decorators import custom_extend_schema

@custom_extend_schema
class ExampleView(APIView):
    
    def get(self, request, *args, **kwargs):
```