---
layout: post
title: .env & gitignore 
subtitle: TIL Day 44
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

# .env & .gitignore
> .env파일과 .gitignore를 이용한환경 변수 관리
>[¶ .env](#fastapi--gpt-4o-chat-서비스)<br>
>[¶ .gitignore](#yolo-webcam-이용-객체-탐지)

# 환경 변수
> 프로세스가 컴퓨터에서 동작하는 방식에 영향을 미치는, 동적인 값들의 모임
**API** 키와 같은 민감한 정보를 환경변수에 포함시켜 코드에 직접 작성하지 않고 사용 가능

# dotenv
- 환경변수를 `.env` 에 작성하여 **Python** 내부에서 사용할 수 있게 해주는 라이브러리

## 설치
- `pip install python-dotenv`

## `.env` 파일 작성
- `.env` 을 이름으로 파일 생성
- 환경변수 정의
  - 변수명 = 변수값 구조
  - 띄어쓰기 없이 작성
  - 변수명은 대문자로 작성
  - 따옴표와 작성과 관련없이 string 값으로 호출

```py
TEST_KEY = test_key
TEST_URL = test_url
TEST_VALUE_1 = test
TEST_VALUE_2 = 'test'
TEST_VALUE_3 = 55
```

## 환경변수 호출
- `from dotenv import load_dotenv` : `.env` 파일 호출 모듈
- `import os` : 환경 변수호출 모듈 
- `load_dotenv()` : `.env` 파일 호출
- 변수명으로 선언하여 환경변수 호출

```py
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv('TEST_KEY')
url = os.getenv('TEST_URL')
value_1 = os.getenv('TEST_VALUE_1')
value_2 = os.getenv('TEST_VALUE_2')
value_3 = os.getenv('TEST_VALUE_3')

print(f"TEST_KEY: {key}")
print(f"TEST_URL: {url}")
print(f"TEST_VALUE_1: {value_1}")
print(f"TEST_VALUE_2: {value_2}")
print(f"TEST_VALUE_3: {value_3}")
```

## 호출결과
- 따옴표와 상관없이 호출됨

```py
TEST_KEY: test_key
TEST_URL: test_url
TEST_VALUE_1: test
TEST_VALUE_2: test
TEST_VALUE_3: 55
```

## `load_dotenv()` 함수의 `.env` 탐색 방식
- 코드가 실행되는 현재 작업 경로 **CWD : current working directory** 우선 탐색
- 상위 디렉토리로 순차적으로 이동하며 탐색

```
1. /my_env/study/working_dir/.env
2. /my_env/study/.env
3. /my_env/.env
```

## 활용

### 경로지정 탐색

```py
import dotenv

dotenv.load_dotenv('사용할 .env파일 경로')
```

### `find_dotenv`

```py
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
```

### 전체 조회

```py
import dotenv

dotenv_file = dotenv.find_dotenv()

# 전체 변수 출력
print(dotenv.dotenv_values(dotenv_file))

# 특정 변수 출력
print(dotenv.dotenv_values(dotenv_file)['TEST_KEY'])
```

### 변수값 수정
- 코드 실행 후 `.env` 파일을 확인하면 값이 변경되어 있음

```py
import dotenv

dotenv_file = dotenv.find_dotenv()

dotenv.set_key(dotenv_file, 'TEST_KEY', 'changed_test_key')
```

```py
TEST_KEY='changed_test_key'
TEST_URL = test_url
TEST_VALUE_1 = test
TEST_VALUE_2 = 'test'
TEST_VALUE_3 = 55
```

[¶ Top](#환경-변수)
