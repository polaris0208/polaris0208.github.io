---
layout: post
title: Python Basic 1
subtitle: TIL Day 69
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python Basic

## print

### pprint
- 복잡한 문자열을 정리해서 출력

```py
from pprint import pprint
```

### `sep`, `end`
- 구분자 설정
- 입력하지 않고 실행할 경우 기본 값 적용

```py
print('hello', 'world', sep = '+', end='!')
# hello+world!
```

### 포매팅
- 출력 형식 지정

```py
a = 1
b = 2
print('{} + {}'.format(a, b))
# 1 + 2
```

- 서식 지정

```py
pi = 3.141592
print("%.2f" % pi)
# 3.14
```

### `sys.stdout.write()`
- 단순 출력, 구분자, 줄바꿈이 없이 출력됨

```py
import sys
sys.stdout.write("hello")
sys.stdout.write("world")
# helloworld
```

## input
- 항상 문자열만 반환
- 다른 형태가 필요한 경우 변환

```py
num_str = input('값은?')
num_int = int(input('값은?'))
num_float = float(input('값은?'))
print(type(num_str))
print(type(num_int))
print(type(num_float))
# <class 'str'>
# <class 'int'>
# <class 'float'>
```

## 변수
- 값을 저장하기 위한 공간
- 문자열, 정수, 실수, 불린(참, 거짓)
- `type()`으로 확인 가능

### 변수명
- 문자로 시작
- 대소문자 구분
- 예약어 사용 금지
  - 파이썬 기능에 사용되고 있는 이름들
  - `class, lamda, True, import...`
  - 사용이 필요한 경우 `_` 붙여 사용
    - `class_`

### 문자열 변수
- `+` 로 붙이기
- `*` 로 반복

### 불린 변수
- 값이 존재하면 `True` 
- 조건문에서 변수명만 입력한 경우 불린 변수로 사용
  - `False_` 는 빈 리스트이기 떄문에 `False` 로 간주
  - 출력되지 않음

```py
True_ = ['True']
False_ = [] 
if True_ : print(True_)
if False_ : print('False')
# True_ = ['True']
```

### 여러 변수 지정

```py
list_ = [1, 2, 3]
a, b, c = list_
print(a, b, c, sep = '\n')

# 
1
2
3
```

### 전역 변수와 지역 변수
- 전역 변수 : 함수 밖에서 선언된 변수
  - 프로그램 전체에서 접근 가능
- 지역 변수 : 함수 내에서 선언된 변수
  - 함수 내에서만 접근 가능

```py
# 전역 변수 지정
num_list = [_ for _ in range(1,11)]

def print_num_list():
  print(num_list)

print_num_list()
print(num_list)

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 지역 변수 지정
def print_ch_list():
  ch = ['a', 'b', 'c', 'd', 'e']
  print(ch)

print_ch_list()
print(ch)

# ['a', 'b', 'c', 'd', 'e']
# name 'ch' is not defined
```