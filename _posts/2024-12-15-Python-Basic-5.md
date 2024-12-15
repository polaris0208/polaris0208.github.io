---
layout: post
title: Python Basic 5
subtitle: TIL Day 84
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python 정렬
>`list.sort()` 메서드와 `sorted()` 함수를 제공

## 기본 정렬

```py
sorted([5, 2, 3, 1, 4])
# 결과: [1, 2, 3, 4, 5]

a = [5, 2, 3, 1, 4]
a.sort()
a
# 결과: [1, 2, 3, 4, 5]
```

## Key
- 각 요소에 대해 호출할 함수 지정
- `str.casefold` : `.lower`보다 폭넓은 소문자 변환 기능 제공

```py
sorted("This is a test string".split(), key=str.casefold)
# 결과: ['a', 'is', 'string', 'test', 'This']

student_tuples = [
    ('john', 'A', 15),
    ('jane', 'B', 12),
    ('dave', 'B', 10),
]
sorted(student_tuples, key=lambda student: student[2])
# 결과: [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
```

## operator
- `operator` 모듈의 `itemgetter()`와 `attrgetter()` 활용
  - `attrgetter` : 객체 속성을 기준으로 호출

```py
from operator import itemgetter, attrgetter

sorted(student_tuples, key=itemgetter(2))
# 결과: [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

다중 수준 정렬도 가능합니다.

sorted(student_tuples, key=itemgetter(1, 2))
# 결과: [('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]
```

## 내림차순 정렬
- `reverse` 매개변수 사용

```py
sorted(student_tuples, key=itemgetter(2), reverse=True)
# 결과: [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
```

## `.sort()`와 `sorted()`의 차이점
- 작동 방식
  - `.sort()`
    - 리스트 객체에서만 사용 가능
    - 리스트를 제자리에서 정렬
    - 원본 리스트 변경
  - `sorted()`
    - 모든 이터러블에서 작동
    - 정렬된 새로운 리스트를 반환
    - 원본 데이터는 변경되지 않음
- 반환값:
  - `.sort()`
    - `None`을 반환
    - 반환하는 값 없이 원본 리스트 수정
  - `sorted()`
    - 정렬된 새로운 리스트를 반환
- 사용 범위
  - `.sort()`: 리스트 전용
  - `sorted()`: 튜플, 딕셔너리, 문자열 등 이터러블 데이터에 사용 가능

# Set
- 순서가 없고 고유한 값
- **mutable** 객체
- **dictionary**와 비슷하지만, **key**가 없이 값만 존재

## 생성
- `set()` : 이터러블 객체 변환
- 순서, 중복 제거

## 메서드

### in
- `in/not in`

### 원소 추가
- `.add()`

```py
k = {100, 105}
k.add(50)
# {105, 50, 100}
```

### update
- 여러데이터를 한번에 추가할 때 사용

```py
k = {1, 2, 3}
k.update([3, 4, 5])
# {1, 2, 3, 4, 5}
```

### 원소 제거
- `.remove()` : 원소를 제거
  - 없으면 **KeyError** 발생
- `discard()` : 없어도 에러발생하지 않음

```py
k = {1, 2, 3}
k.remove(3)
# {1, 2}

k = {1, 2, 3}
k.discard(3)
# {1, 2}
```

### 연산자

```
| - 합집합 연산자
& : 교집합 연산자
- : 차집합 연산자
^ : 대칭차집합 연산자(합집합 - 교집합)
```

### 연산메소드

- `union` : 합집합
- `intersection` : 교집합
- `difference` : 차집합
- `symmetric_difference` : 대칭차집합 연산자(합집합 - 교집합)
- `issubset` : 부분집합 여부 확인
- `isdisjoint` : 교집합 여부
