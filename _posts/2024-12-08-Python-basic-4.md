---
layout: post
title: Python Basic Tuple & Dictionary
subtitle: TIL Day 77
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python : tuple

## 개념
- 순서가 있는 불변 객체의 집합

```py
tuple_ = (1, 2)
# ()
tuple_ = 1, 2 
# 패킹
list_ = [1, 2]
tuple_ = tuple(list_)
# tuple()
```

## 인덱싱과 슬라이싱
- 리스트와 동일

## 불변성
- 요소 변경, 추가 불가
- 요소를 추가하여 새로 생성은 가능

```py
origin = 1, 2, 3
new = origin + 4
print(new)
# 
(1, 2, 3, 4)
```

## 메서드
- 리스트와 동일

## 활용
- 불변성과 순서를 사용하여 활용

### 딕셔너리 키

```py
dict_ = { (0,0) : 'point zero'}
```

### 복수 반환

```py
def calculate(a, b):
    sum_ = a + b
    diff = a - b
    prod = a * b
    return sum_, diff, prod
```

### 순서가 중요한 데이터 저장
- 계좌 번호, 회원 번호 등 등록 번호
- 날짜 좌표 등

```py
id_ = ('B', 458011)
```

# Python : dictionary

## 개념
- **Key** 와 **Value** 의 쌍을 저장
- 요소의 추가 삭제 변경이 가능
- **Key** 는 고유의 값 사용 : 문자열, 튜플

## 생성

### `{}`

```py
info = {
  'id' : 123,
  'password' : 456,
}
```

### `dict()`

```py
# 키워드 인자 사용
info = dict(model='m1911', brand='colt', year=1911)

# 튜플, 리스트
info = [('Kim', 3), ('Yang', 54)]
info_dict = dict(info)
```

### `fromkeys()`

```py
name = ['Ryu', 'Kim', 'Yang']
position = 'SP'
dict_ = dict.fromkeys(name, position)
```

### 요소 접근

```py
info = {
  'name' : 'TAA'
  'position' : 'RWB'
}
info['name']
info,get('position')
```

### 요소 변경 및 추가
- 변경 : `info['position'] = 'CM'`
- 추가 : `info['homegrown] = True`

### 삭제

```py
info = {
  'name' : 'Heo'
  'id' : 13
  'div' : 'DB'
}
del info['div']
info.pop['div']
info.clear() # 전체 삭제
```

### 메서드

### `keys()`
- 모든 키를 반환

### `values()`
- 모든 값을 반환

### `items()`
- 모든 키-값 쌍을 튜플로 반환

### `update()`
- 다른 키-값쌍으로 수정

```python
info = {
  'name' : 'Heo'
  'id' : 13
  'div' : 'DB'
}
info.update({'div' : 'KW'})
```

### `popitem()`
- 마지막으로 삽입된 키-값 쌍을 삭제하고 반환

### `setdefault()`
- 기존 값이 있으면 그 값을 반환하고, 없으면 키와 입력 값을 딕셔너리에 추가