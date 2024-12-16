---
layout: post
title: Http & RESTful
subtitle: TIL Day 85
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Web]
author: polaris0208
---

# HTTP
> **H**yper **T**ext **T**ransfer **P**rotocol

## 개념
>웹 브라우저와 서버 간에 데이터를 주고받는 프로토콜로, 인터넷의 작동 방식의 근간
    
- 요청(Request) : 클라이언트 → 서버
- 응답(Response) : 서버 → 클라이언트
- 효율적이고 확장성 있는 구조
  - **Stateless (무상태)**
    - 서버가 클라이언트의 상태를 보존하지 않음
  - **Connectless (비연결성)**
    - 연결을 유지하지 않고 응답을 주고 받을 때만 연결

### HTTP Message
- 요청과 응답은 비슷한 구조를 가짐

- **요청(Request)**
  - 클라이언트가 서버의 행동 유발
- **응답(Response)**
  - 요청에 대한 서버의 답변

- **Http Message의 구조**
  - **Start Line** - 실행되어야할 요청 / 요청에 대한 성공 또는 실패
  - **HTTP Header** - 요청에 대한 설명 / 본문에 대한 설명
  - **Blank Line** - 메타 정보의 끝
  - **HTTP Body** - 요청과 관련된 내용 / 응답과 관련된 문서

**요청(Request)**

- **Start Line** : **Method, Traget, HTTP Version**
- **Headers** : 요청에 필요한 여러가지 메타 정보
- **Body** : 요청에 필요한 여러가지 데이터

**응답(Response)**

- **Start Line** : **Method, Traget, HTTP Version**
- **Headers** : 요청에 필요한 여러가지 메타 정보
- **Body** : 요청에 필요한 여러가지 데이터

### HTTP Request Methods
- 요청 자원에 대한 **행위**를 나타냅니다.
- **GET, POST, PUT, DELETE**...

### HTTP Status Code
- 요청 결과를 담은 코드값
    - `1XX` : Informational Response
    - `2XX` : Successful Response
      - 200 **OK**
      - 201 **Created**
      - 202 **Accepted**
      - 204 **No Content**     
    - `3XX` : **Redirection Message**
    - `4XX` : **Client Error Response**
      - 400 **Bad Request**
      - 401 **Unauthorized**
      - 403 **Forbidden**
      - 404 **Not Found**
    - `5XX` : **Server Error Response**
      - 500 **Internal Server Error**
      - 503 **Service Unavailable**

## URL
- 자원을 식별하기 위해 **URI(Uniform Resource Identifier)** 사용
  - **URL - URN** 으로 구성
  - **URN**을 사용하는 비중이 낮기때문에 **URI**와 **URL**을 같은 의미로 사용

### URI (Uniform Resource Identifier)
- 통합 자원 식별자입니다.
- 인터넷의 자원을 식별할 수 있는 유일한 문자열

### **URL(Uniform Resource Locator)**
- **통합 자원 위치(Location)**
  - 웹상에 자원이 어디 있는지 나타내기 위한 문자열

### **URN(Uniform Resource Name)**
- **통합 자원 이름(Name)**
- 위치에 독립적인 자원을 위한 유일한 이름: **ISBN**(국제표준도서번호)
    
### URI의 구조


- `https://`
    - **Scheme(Protocol)**
        - 브라우저가 사용하는 프로토콜
        - **http, https, ftp, file**, …
        
- `www.polaris0208.io`
    - **Host(Domain name)**
        - 요청을 처리하는 웹 서버
        - **IP** 주소 or 도메인 이름
        
- `:80`
    - **Port**
        - 리소스에 접근할 때 사용되는 게이트
        
- `/path/to/resource/`
    - **Path**
        - 웹 서버에서의 리소스 경로
        
- `?key=value`
    - **Query(Identifier)**
        - 웹 서버에 제공하는 추가적인 변수
        - `&`로 구분되는 `Key=Value` 형태의 데이터
        
- `#docs`
    - **Fragment(Anchor)**
        - 해당 자원 안에서의 특정 위치
        - **HTML** 문서의 특정 부분

# RESTful API
- **CLI(Command Line Interface)** : 명령줄로 소통
- **GUI(Graphic User Interface)** : 그래픽으로 소통
- **API(Application Programming Interface)** : 프로그래밍으로 어플리케이션과 소통

## Representational State Transfer
- 웹에 대한 소프트웨어 설계 방법

### 활용
- 형식 `GET` `POST` `PUT` `DELETE` + `PATCH`
- `POST` `/articles/` : 생성
- `GET` `/articles/` : 조회
- `GET` `/articles/1` : 상세 조회
- `DELETE` `/articles/1/` : 삭제
- 핵심 규칙
  - 자원 : **URI**
  - 행위 : **HTTP Method**
  - 표현 : **Json**

### JSON(JavaScript Object Notation)
- **Key : Value** 형식
- `.json` 형식으로 사용
-  `"`만 허용
- `true` `false` : 소문자 표기 주의

#### Django Json
- **Json** 형식으로 전달
- `serializers` 사용

```py
# json 형식으로 재구조화
def json_01(request):
    articles = Article.objects.all()
    json_articles = []

    for article in articles:
        json_articles.append(
            {
                "title": article.title,
                "content": article.content,
                "created_at": article.created_at,
                "updated_at": article.updated_at,
            }
        )

    return JsonResponse(json_articles, safe=False)

# serializers
from django.core import serializers

def json_02(request):
    articles = Article.objects.all()
    res_data = serializers.serialize("json", articles)
    return HttpResponse(res_data, content_type="application/json")
```