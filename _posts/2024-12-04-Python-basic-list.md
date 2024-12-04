---
layout: post
title: Python Basic list
subtitle: TIL Day 73
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python Basic : list
- 변경 가능한 객체의 조합
- 여러 데이터 타입 포함 가능
  - `list_ = [1, '문자열', True']`

## 리스트 생성
- `[]`로 묶기
  - `ch = ['a', 'b', 'c']`
- `list()`함수 사용
  - `list('list')`
  - `['l', 'i', 's', 't']`

## 인덱싱과 슬라이싱
- 인덱싱

```py
ch[0]
# a
ch[1]
# b
```

- 슬라이싱
  - `str_[start : end+1 : step]`

## 변경
- 인덱싱과 슬라이싱 활용

```py
ch[0] = 'A'
ch
# ['A', 'b', 'c']
ch[1:2] = ['B', 'C']
# ['A', 'B', 'C']
```

## 메서드
- `.append(요소)`
- `.insert(위치, 요소)`
- `.extend(iterable)` : **iterable** 모두 추가
- `.remove(요소)`
- `.pop(위치)`
- `.clear()` : 모두 제거
- `.index(요소)` : 요소의 인덱스
- `.count(요소)` : 요소의 개수
- `.sort()` : 오름차순 정렬
- `.reverse():` : 뒤집기

## 연산
- `+` : 연결
- `*` : 반복

## 리스트 반복문
- **for** 루프
- 리스트 컴프리헨션
- **enumerate**

```py
list_ = []
for num in range(1,11):
  list_.append(num)
# [1,2,3,4,5,6,7,8,9,10]
[num for num in range(1,11)]
# [1,2,3,4,5,6,7,8,9,10]
[num for num in range(1,11) if num % 2 == 0]
# [2,4,6,8,10]
```