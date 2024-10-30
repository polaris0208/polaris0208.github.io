---
layout: post
title: Python 라이브러리 기본 개념
subtitle: TIL Day 10
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---
# Library & Framework

- 라이브러리 = 도구상자
- 프레임워크 = 설계도
- Django : 웹 프레임워크

```
* 보안, 확장성, DRY(don't repeat yourself) 원칙
* 모델 = 데이터베이스 구조화, 데이터와 상호작용
* 뷰 = 사용자가 보는 화면
* 템플릿 = HTML 생성, 데이터 표시
* URL conf URL과 뷰를 연결
* Django 작동 방식
# url 호출 -> 뷰 호출 -> 데이터 처리 및 템플릿 렌더링 => html응답
```

* MVT(Model Veiw Template) 패턴

```
# 모델 = 데이터 처리 ex)ORM - SQL 동작을 Python에서 
# 뷰 = 사용자 요청에 따라 어떤 데이터를 보여줄지 결정
# 템플릿 = 뷰와 상호작용
```

## Pandas
* Python 데이터 분석 Library

```bash
#설치
conda activate 가상환경이름 # 충돌방지를 위해 가상환경 할성화
또는 source 가상환경이름/bin/activate
pip --version # pip verison이 최신인지 확인 
pip install --upgrade pip # 패키지에 따라 최신버전을 요구하는 경우 존재
pip install pandas # pip = Package installer for Python
```

* python 환경에서 적용(import에 시간소요)

```python
import pandas as pd # pandas series 기능 사용을 위해 import
sample = pd.Series(["Alice", "Wonderland", 1], 
		index = ["이름", "주소", "학년"])
# index 설정을 통해 series의 데에터에 쉽게 접근 가능 # ([])
# index 설정을 안하면 기본 위치가 설정됨
print(sample) 
#
이름         Alice
주소    Wonderland
학년             1
dtype: object
#데이터 타입은 시리즈 전체가 공유
#해당 시리즈에서는 문자형과 정수형이 혼재하여 object 

sample["이름"] # 호출 방식 / index 활용
```

* 데이터 타입은 시리즈 내부 데이터를 모두 수용할 수 있는 형식을 택함

```python
sample.apply(type) 
#이름    <class 'str'>
#주소    <class 'str'>
#학년    <class 'int'>
#dtype: object
```

* DataFrame

```python
data = {
  "이름" : ["Xavi", "Iniesta", "Busquets"],
  "출생" : [80, 84, 88],
  "포지션" : ["CM", "AM", "DM"]
} 
#dictionary 형태로 전달 할 때는 list의 길이르 맞춰야함
df = pd.DataFrame(data)
```

```python
df["이름"][2]
# 'Busquets'
# 컬럼명과 index를 이용해 접근 가능

df["포지션"]["Xavi"]
# 포지션 중 Xavi의 포지션 조회 (순서가 중요)
# CM
```

* 데이터프레임의 열이 시리즈로 구성
* 데이터프레임 -2차원(다양한 데이터타입)
* 시리즈 -1차원(하나의 데이터타입, 내부에서는 여러 종류)

```python
df.apply(type)
이름     <class 'pandas.core.series.Series'>
출생     <class 'pandas.core.series.Series'>
포지션    <class 'pandas.core.series.Series'>
dtype: object
```

* 정수형과 문자형이 섞이면 numpy.intenger 값으로 기능이 달라짐

```python
type(df["출생"]["Xavi"])
#numpy.int64
```

## Numpy

```
* numpy-다차원 데이터의 분석
고속 배열 연산
수학 함수
선형 대수(벡터 공간, 벡터, 선형 변환, 행렬, 연립 선형 방정식 등)
통계 함수
```
