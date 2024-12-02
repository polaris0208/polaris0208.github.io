---
layout: post
title: Python Basic 2
subtitle: TIL Day 71
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

## Raw String
- 이스케이프 문자 `\` 를 무시하고 그대로 문자열에 표시
  - `\n` : 줄바꿈
  - `\t` : 탭
  - `\\` : 백 슬래시
  - `\'` : 작은 따옴표
  - `\"` : 큰 따옴표

```py
print("\n\t")
# 줄바꿈, 공백
print(r"\n\t")
# \n\t
```

## 인덱싱과 슬라이싱

### 인덱싱
- `str[0]` : 첫 번째 문자
- `str[-1]` : 마지막 문자

### 슬라이싱
- `str[start : end +1 : step]`
- `str[start:]` : 끝까지
- `str[:end+1]` : 처음부터
- `str[start : end +1 : -1]` : 뒤에서 부터 1씩
- `str[::-1]` : 문자열 뒤집기

## 문자열 메소드
- `.capitalize()` : 제일 앞글자 대문자
- `.title()` : 각 단어의 앞글자 대문자

```py
str_ = 'hello world'
str_.capitalize() 
# Hello world
str_.title()
# Hello World
```

## `.find()` 와 `.index()`
- 사용법은 동일
- 값이 없을 때
  - `.find()` : -1
  - `.index()` : `ValueError: substring not found`

## `.replace('A', 'B')`
- `A`를 모두 `B`로 바꾼다

```py
str_ = '터키 이스탄불은 과거 동로마와 오스만제국의 수도였으나 정작 터키 수도는 아니다.'
str_.replace('터키', '튀르키에')

# 튀르키에 이스탄불은 과거 동로마와 오스만제국의 수도였으나 정작 튀르키에 수도는 아니다.
```

## 공백 제거
- `strip()`: 문자열 양쪽의 공백을 제거
- `lstrip()`: 문자열 왼쪽의 공백을 제거
- `rstrip()`: 문자열 오른쪽의 공백을 제거

## 문자열 분할과 결합

- `split(separator)`: `separator`를 기준으로 문자열을 분할
- `'separator'.join(iterable)`: `separator` 기준으로 문자열의 `iterable` 요소를 연결

```py
str_ = ['U', 'I']

str_ = '&'.join(str_)
# 'U&I'
str_ = str_.split('&')
# ['U', 'I']
```