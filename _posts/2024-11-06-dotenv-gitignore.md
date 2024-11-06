---
layout: post
title: .env & .gitignore 
subtitle: TIL Day 45
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

# .env & .gitignore
> .env파일과 .gitignore를 이용한환경 변수 관리<br>
>[¶ .env](#fastapi--gpt-4o-chat-서비스)<br>
>[¶ .gitignore](#yolo-webcam-이용-객체-탐지)

# 환경 변수
> 프로세스가 컴퓨터에서 동작하는 방식에 영향을 미치는, 동적인 값들의 모임
**API** 키와 같은 민감한 정보를 환경변수에 포함시켜 코드에 직접 작성하지 않고 사용 가능

# `.env`
- 프로젝트에 필요한 환경변수를 모아 작성한 파일

## dotenv
- 환경변수를 `.env` 에 작성하여 **Python** 내부에서 사용할 수 있게 해주는 라이브러리

## 설치
- `pip install python-dotenv`

## `.env` 파일 작성
- `.env` 을 이름으로 파일 생성
- 환경변수 정의
  - `변수명` = `변수값` 구조
  - 띄어쓰기 없이 작성
  - 변수명은 대문자로 작성함이 관례
  - 따옴표와 작성과 상관없이 `str` 값으로 호출

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
- `os.getenv()` 에 변수명을 입력하여 호출

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

## 주의점
- `.env` 파일이 노출되게 되면 치명적
  - **github** 및 클라우드 서비스 이용 시 주의 필요
- 자주 사용하는 환경변수는 일일이 `.env` 파일을 작성하기보다 터미널에 등록

## 해결방법
- `.gitignore` 파일 작성

# `.gitignore`
> 버전관리에서 제외되어야 할 폴더 및 파일들을 규정하는 문서<br>
> 불필요한 로그 및 시스템 파일, 민감 정보 등을 **Repository** 관리 대상에서 제외

## 생성
- `.gitignore` 파일 생성
  - 최상위 위치 : **root directory** 에 생성
  - 다른 위치에도 생성 가능 : 허용범위가 해당 디렉토리의 하위 디렉토리로 제한

## 문법
- **Globs(Global Patterns) [¶](https://code.visualstudio.com/docs/editor/glob-patterns)** 사용
- `[파일명]` : 특정 파일 무시
- `#` : 주석처리 - 인식되지 않음
- `!` : 예외처리 - 해당 파일을 무시하지 않음
- `*` : 전체
  - `*.[확장자]` : 특정 확장자 파일 무시
    - `*.env` : `.env` 파일 무시
- `/` : 현재 디렉토리
  - `[디렉토리]/` : 특정 디렉토리 내부 파일 무시
  - `[디렉토리]/[파일]` : 특정 디렉토리의 특정 파일
  - `[디렉토리]/**/[파일]` : 하위 디렉토리도 포함
- `-` : 특정 이름으로 탐색
  - `[특정이름]-*[파일]` : 특정이름으로 시작하는 파일 무시
  - `docs-*.pdf` : **doc** 라는 이름으로 시작하는 모든 **pdf** 파일 무시
- `{}` : 여러 파일 
  - `/*.{확장자 1, 확장자 2}`
- `[]` : 범위 설정 - **a-z, A-Z, 0-9** 등
  - `/docs[1-3].pdf` : **docs1.pdf, docs2.pdf, docs3.pdf** 무시

## 주의점
- 이미 버전 관리중인 파일은 무시 불가
  - **Repository** 에 **commit** 으로 올라간 파일
  - **Staging Area** 에 올라간 파일

## 해결방법
- 파일 제거 또는 캐시 제거
- 해당 파일 제거
  - `git rm [파일명]`
  - `git commit -m [메시지]`
- 캐시 제거
  - `git rm -r --cached .`
  - `git add .` 
  - `git commit -m`
  - `git push origin [브렌치]`

## [gitignore.io](https://www.toptal.com/developers/gitignore)
- 옵션을 선택하여 원하는 `.gitignore` 파일 생성
- `.py` 파일을 무시하고 싶은 경우 **python** 입력 후 생성

[¶ Top](#환경-변수)
