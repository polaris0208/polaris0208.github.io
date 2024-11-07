---
layout: post
title: Python Comprehension
subtitle: TIL Day 46
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Comprehension
> **list**, **dictionary**, **set**

## List Comprehension
- 리스트를 생성
- `[새로운 리스트에 들어갈 요소] for [리스트에서 가져올 요소] in [리스트]`

### 기본 for문

```py
numbers = []
for n in range(1, 10+1):
    numbers.append(n)
```
 
### 기본 컴프리헨션
- `[x for x in range(10)]`

### 컴프리헨션 활용

#### 연산 적용
- 각 인자에 두배
- `[2 * x for x in range(10)]`

#### 조건 적용
- 짝수만 포함
- `[x for x in range(10) if x % 2 == 0]`

#### 중복 적용
- 다중 **for**문
- `[ (x, y) for x in ['a', 'b', 'c'] for y in ['1', '2', '3']]`

```
[(a,1), (a,2), (a,3), 
(b,1), (b,2), (b,3), 
(c,1), (c,2), (c,3),]
```

## Dictionary Comprehension

### Set Comprehension
- 리스트 컴프리헨션에 `{}` 사용
- `{ x+y for x in range(10) for y in range(10)}`

```
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
```

### Value 지정

```py
spells = ['a', 'b', 'c', 'd']

{ spell : 0 for spell in spells }
#
{'a': 0, 'b': 0, 'c': 0, 'd': 0}
```

### Value 제거
- 조건문 활용

```py
scores = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
scores = { name: score for name, score in scores.items() if name != 'e'}
```
