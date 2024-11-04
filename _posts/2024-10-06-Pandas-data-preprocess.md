---
layout: post
title: Pandas 데이터 전처리
subtitle: TIL Day 14
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---

# Pandas 데이터 전처리
## 데이터 전처리 개요

### 데이터 전처리(비선형 변환 포함)
1. 데이터 정규화
* 데이터의 범위를 0과 1 사이로 변환 = 서로 다른 범위를 가진 데이터를 동일한 스케일로 맞추어 비교(ex 키:재산)
2. 데이터 표준화
* 평균을 0으로 표준편차를 1로 = 정규분포와 같이 조정하여 비교
3. 경사 하강법 = 손실함수(내가 얼마나 잘못되었나) 최소화
4. 과적합
* 지나치게 학습해서 일반적인 문제를 해결 못하는 경우(같은 문제를 너무 학습하면 답을 외워버림)

## 정규화
### MinMaxScaler()
* pip로 scikit-learn 설치
* min-max 정규화 = 최솟값을 0 최댓값을 1 
* 즉 0과 1사이의 값으로 바꿔 비교가 쉽게 변환

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = {
  '특성1': [10, 20, 30, 40, 50],
  '특성2': [1, 2, 3, 4, 5]
}
df = pd. DataFrame(data)
print(df)
#
   특성1  특성2
0   10    1
1   20    2
2   30    3
3   40    4
4   50    5

scaler = MinMaxScaler()
# 처리 메서드 선택
nomalized_df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
# .fit_transform() 사용
print(nomalized_df)
#
    특성1   특성2
0  0.00  0.00
1  0.25  0.25
2  0.50  0.50
3  0.75  0.75
4  1.00  1.00
# 동일한 스케일에서 비교된 값이 출력
```

## 표준화 
### Z 표준화
* 데이터에서 평균을 뺴고 표준편차로 나눔
* 평균 0 표준편차는 1에 가깝게 = 표준정규분포화

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_df = pd.DataFrame(scaler.fit_transform(df), columns = df. columns)
standardized_df
print(standardized_df.describe())
# 기초 통계 출력
#
            특성1       특성2
count  5.000000  5.000000
mean   0.000000  0.000000
# 평균이 0
std    1.118034  1.118034
# 표준편차가 1에 가까워짐
min   -1.414214 -1.414214
25%   -0.707107 -0.707107
50%    0.000000  0.000000
75%    0.707107  0.707107
max    1.414214  1.414214
```

## 전처리 예시

```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)
age_data = titanic[['Age']]
age_data = age_data.dropna()
# 결측치 제거
scaler = StandardScaler()
age_scaled = scaler.fit_transform(age_data)
age_scaled_df = pd.DataFrame(age_scaled, columns= ['age_scaled'])
# z표준화
print(age_scaled_df.describe())
#
         age_scaled
count  7.140000e+02
mean   2.338621e-16
std    1.000701e+00
min   -2.016979e+00
25%   -6.595416e-01
50%   -1.170488e-01
75%    5.718310e-01
max    3.465126e+00
```

## 비선형 변환
### .log()
* 데이터의 분포를 줄여줌

```python
import numpy as np
df['특성1_log'] = np.log(df['특성1'])
# 넘파이 메소드
print(df.describe())
             특성1       특성2
count   5.000000  5.000000
mean   30.000000  3.000000
std    15.811388  1.581139
# 표준편차가 더 작아짐
min    10.000000  1.000000
25%    20.000000  2.000000
50%    30.000000  3.000000
75%    40.000000  4.000000
max    50.000000  5.000000
```

### .sqrt() 
* 제곱근(square root) 변환
* 데이터는 값이 클 수록 큰 폭으로 변화
* 표준화를 이용해 데이터를 평탄화 

```python
df['특성1_sqrt'] = np.sqrt(df['특성1'])
print(df)
# 값의 차이가 줄어듦
   특성1  특성2  특성1_sqrt
0   10    1  3.162278
1   20    2  4.472136
2   30    3  5.477226
3   40    4  6.324555
4   50    5  7.071068
```

### boxcox()
* 정규분포에 가깝게 변환(양수 데이터만)
* 심화----추가학습 필요 !!

```python
from scipy.stats import boxcox
df['특성1_boxcox'], _ = boxcox(df['특성1'])
print(df)
   특성1  특성2  특성1_sqrt
0   10    1  3.162278
1   20    2  4.472136
2   30    3  5.477226
3   40    4  6.324555
4   50    5  7.071068
```

### RobustScaler()
* 평균은 이상치에 영향을 크게 받음
* 로버스트(강건한)한 통계를 얻기 위해 평균대신 중앙값, 표준편차 대신 중앙값 절대편차 사용

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns = df.columns)
print(scaled_df)
#
   특성1  특성2  특성1_sqrt
0 -1.0 -1.0 -1.249689
1 -0.5 -0.5 -0.542582
2  0.0  0.0  0.000000
3  0.5  0.5  0.457418
4  1.0  1.0  0.860411
# 중앙값인 30과 3을 기준으로 표준화 된 모습
```

## 인코딩 
* 데이터를 다른 시스템에서 사용 할 수 있도록 처리

### LabelEncoder()

```python
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
data = {'과일':['사과', '바나나', '사과', '오렌지', '바나나']}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['과일_인코딩'] = label_encoder.fit_transform(df['과일'])
print(df)
# 데이터 요소마다 값을 부여
    과일  과일_인코딩
0   사과       1
1  바나나       0
2   사과       1
3  오렌지       2
4  바나나       0
# 과일끼리는 값의 차이가 의미x 
# 인코딩 결과에서는 과일마다 다른 크기의 값을 부여 
# 따라서 라벨 인코딩은 데이터에 순서가 있을 때 적합
```

### one-hot 인코딩
* .get_dummies()
* 이진법 데이터로 변환
* 값이 같고 순서가 상관없는 경우
* 각 범주를 독립적으로 처리
* 데이터가 급격히 늘어나는 단점

```python
df_one_hot = pd.get_dummies(df['과일'], prefix = '과일')
print(df_one_hot)
#
   과일_바나나  과일_사과  과일_오렌지
0   False   True   False
1    True  False   False
2   False   True   False
3   False  False    True
4    True  False   False
# 데이터 항목이 더 많아질 경우 행과 열이 크게 늘어날 수 있음
```

### 차원 축소 인코딩
* .value_count()
* count or frequeny encoding
* 빈도 기반 인코딩

```python
df['과일_빈도'] = df['과일'].map(df['과일'].value_counts())
print(df)
    과일  과일_인코딩  과일_빈도
0   사과       1      2
1  바나나       0      2
2   사과       1      2
3  오렌지       2      1
4  바나나       0      2
```

### 등급 인코딩
* ordinal 인코딩(기능이라기보단 사용법)
* 라벨 인코딩의 단점 보완
* 순서 부여

```python
data2 = {'등급' : ['낮음', '중간', '높음', '중간', '높음']}
df2 = pd. DataFrame(data2)
label_encoder = LabelEncoder()
df2 ['등급_인코딩'] = label_encoder.fit_transform(df2['등급'])
print(df2)  
#
   등급  등급_인코딩
0  낮음       0
1  중간       2
2  높음       1
3  중간       2
4  높음       1
# 라벨 인코더만 해주면 낮음 0 중간 2 높음 1 로 부적절하게 라벨링이 이루어짐
order = {
  '낮음' : 1, '중간' : 2, '높음' : 3
}
# 순서 맵핑
df2['등급_인코딩'] = df2['등급'].map(order)
print(df2)
   등급  등급_인코딩
0  낮음       1
1  중간       2
2  높음       3
3  중간       2
4  높음       3
# 등급에 맞게 라벨링이 부여됨
```

## Embedding
* embed 끼워넣다
* 하나의 위상 공간에서 다른 위상 공간으로
* 고차원 데이터를 벡터 변환(이미지 벡터변환 개념)

```python
import sentence_transformers as st
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# 예시 모델
sentence = ['This is and exemple sentence', 'Each sentence is converted']
embeddings = model.encode(sentence)
# 모델 적용
print(embeddings)
# 벡터로 전환된 값들끼리는 유사도를 확인 가능
from sklearn.metrics.pairwise import cosine_similarity
# metrics : 계량분석, pairwise : 쌍으로 
# cosine_similarity : 코사인 유사도: 백터의 각도를 기준으로 유사도를 판단
# 180도 -1, 90도 0, 0도 1, 즉 -1 부터 0까지의 값을 가진다, 1일 경우 일치
```

* https://wikidocs.net/24603 (코사인 유사도 개념, 활용)

```python
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f'Similarity: {similarity}')
[[ 2.93127000e-01  8.37765932e-01  2.98555195e-01  3.05838525e-01
   3.50848883e-01 -2.26002291e-01  9.05976415e-01 -5.65104783e-01
   1.69190481e-01  3.16751271e-01  2.76756465e-01 -7.30728209e-01
                                  ...
   2.36845031e-01  5.73047340e-01  4.85777766e-01  1.85360566e-01
   7.73247704e-02  4.17804569e-01  4.38058347e-01 -6.62089348e-01]]
# Similarity: [[0.46550906]] # 유사도
```
