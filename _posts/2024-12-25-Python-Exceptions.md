---
layout: post
title: Python Exceptions
subtitle: TIL Day 94
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Exceptions
- 프로그램 실행 도중에 발생하는 오류
- 파이썬 : 유형, 관련 메시지를 출력, 프로그램을 종료
  - `int()`에 문자열 입력
  - `ValueError` 예외를 발생

```python
number = int(input("숫자를 입력하세요: ")) 
print("입력한 숫자:", number)
```

## 예외 처리 구문 (try-except)
- `try` 우선 실행
- `except` 예외가 발생하면 실행

```python
try:
    # 예외가 발생할 수 있는 코드
except 예외타입:
    # 예외 발생 시 실행
```

```python
try:
    number = int(input("숫자를 입력하세요: "))
    print("입력한 숫자:", number)
except ValueError:
    print("올바른 숫자를 입력하지 않았습니다!")
```

## 예외 처리 정의
- 여러 개의 `except` 절을 사용

```python
try:
    f = open("nonexistent_file.txt", "r")
    number = int(input("숫자를 입력하세요: "))
    print("입력한 숫자:", number)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다!")
except ValueError:
    print("올바른 숫자를 입력하세요!")
```

## `Exception`
- 대부분의 예외를 포괄하는 기본 예외 클래스
- 구체적인 예외를 처리한 후 마지막에 남는 예외를 처리하도록 사용

```python
except Exception as e:
    print("예외가 발생했습니다:", e)
```

## else와 finally
- `else` 블록: `try` 블록에서 예외가 발생하지 않았을 때만 실행
- `finally` 블록: 예외 발생 여부와 상관없이 항상 실행
  - 주로 자원 정리(파일 닫기, **DB** 연결 종료 등)에 사용

### 예시

```python
try:
    number = int(input("숫자를 입력하세요: "))
except ValueError:
    print("숫자가 아닌 값을 입력했습니다.")
else:
    # 예외가 발생하지 않았을 때
    print("입력한 숫자는:", number)
finally:
    # 예외 발생 여부 관계 없이 항상 실행
    print("프로그램 실행을 마칩니다.")
```

## 예외 클래스와 상속

- 모든 예외는 `BaseException` 클래스를 상속
- 주로 사용되는 예외들은 `Exception` 클래스를 기반
  - `ValueError`: 값
  - `TypeError`: 자료형
  - `IndexError`: 범위
  - `KeyError`: 키
  - `FileNotFoundError`: 존재하지 않는 파일에 접근할 때
  - `ZeroDivisionError`: 0으로 나누었을 때

## 클래스를 정의

```python
class MyCustomError(Exception):
    pass

try:
    raise MyCustomError("만든 예외")
except MyCustomError as e:
    print("사용자 정의 예외 발생:", e)
```

## 예외 발생시키기 (raise)

- `raise` 특정 조건에서 예외 발생

```python
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("0으로 나눌 수 없습니다.")
    return a / b

try:
    result = divide(10, 0)
    print(result)
except ZeroDivisionError as e:
    print("예외 발생:", e)
```

## 예외 체이닝 (from)
- 예외를 감싸서 새로운 예외를 발생시킬 때 원래 예외 정보를 유지

```python
try:
    number = int("abc")
except ValueError as e:
    raise TypeError("잘못된 형변환 시도")
```