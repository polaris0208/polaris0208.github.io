---
layout: post
title: Numpy 기본 개념
subtitle: TIL Day 11
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---
# Numpy
## 다차원 데이터의 분석과 고속 배열과 연산
* 파이썬 기본 연산

```python
sample_list = [1, 2, 3, 4, 5]
for num in sample_list:
  num += 10
# 동작은 하지만 리스트 본래의 값은 변경되지 않음

for i, v in enumerate(sample_list): 
# enumerate() 열거기능
  sample_list[i] = v + 10 
  # i번째 값에 10을 더함 = 각 위치의 값에 10을 더함
sample_list 
# [11, 12, 13, 14, 15]
#리스트 본래의 값도 변경 / 그러나 식이 복잡
```

* 넘파이를 이용한 빠른 연산

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
arr
# array([1, 2, 3, 4, 5])

arr = arr + 10 
# 넘파이 어레이를 통해 간단히 계산 가능
arr
# array([11, 12, 13, 14, 15])

arr = arr / 2
arr
# array([5.5, 6. , 6.5, 7. , 7.5])
```

* 복합 대입 연산자는 오류를 초래하니 주의해서 사용

## 배열 변경

```python
arr = np.array([1, 2, 3, 4, 5, 6])
arr
# array([1, 2, 3, 4, 5, 6])
arr.shape
(6,) # (1, 6) 행열인데 1 생략

arr_2 = arr.reshape((2, 3)) 
# 배열 모양의 변경 ((행, 열)) 행렬값 자체가 튜플이기 떄문에 괄호를 두겹 사용
arr_2
# array([[1, 2, 3],
        [4, 5, 6]])
```

## Broadcasting 기능
* 크기가 다른 배열간의 연산을 가능하게 해줌

```python
arr3 =np.array([[1, 2, 3], [4, 5, 6]]) 
# (2,3) 배열
arr4 = np.array([10, 11, 12])
# (1,3) 배열
arr3 + arr4
# array([[11, 13, 15], [14, 16, 18]])
# (2, 3) 배열
```

* 단 규칙에 맞춰 계산

```
# 행렬의 더하기 뺴기와 유사
# 각 차원에서 크기가 동일 
# arr3 은 1차원 3 2차원 3 
# arr4 는 1차원 3 
# 크기가 다른 경우 하나의 배열에서 차원의 크기가 1인 경우
```

## 함수 사용
* 병합

```python
np.add(arr1, arr2)
```

* 빈 리스트 생성

```python
result = np.empty_like(arr1) 
# arr1의 빈공간 리스트 생성
np.add(arr1, arr2, out = result) 
등을 이용해 값을 채울 수 있음
```

* 집계

```python
np.sum(arr1)
```

* 누적 합

```python
np.prod(arr1)
```

* 누적 곱

```python
np.cumsum(arr1)
```

* 평균

```python
np.mean(arr1)
```

* 중앙값(정중앙이 없을 경우 중앙값의 평균)

```python
np.median(arr1)
```

* 표준편차

```python
np.std(arr1)
```

* 최소, 최대 # .min .max
* argument of minimum 최솟값으로 만들어주는 매개변수

```python
np.argmin(arr1)
```

* 최솟값과 최댓값이 차이

```python
np.ptp(arr1)
```

* 수학과 관련된 함수는 대부분 존재 
지수, 로그, 삼각함수, 쌍곡선, 복소수.........

```
.suntract 뺄셈
.multiply 곱셈
# 곱셉은 행렬 곱셈이 아닌 요소 곱셈
.floor_divde 
# 요소별 나눗셈 후 소수점 이하 버림
.mod
# 요소별 나눗셈의 나머지를 반환
```

* 넘파이의 기능은 편리하나 사용법에 주의

```python 
arr1 = np.array([1,2,3,4,5])
arr2 = np.array([10, 12,13,14,15])
arr3 = np.array([20,21,22,23,24])
np.add(arr1,arr2,arr3) 
# 세 어레이를 더하는 것이 아닌
arr3 
# 두 어레이를 더해 세번 쨰 어레이에 저장 

# array([11, 14, 16, 18, 20])
```