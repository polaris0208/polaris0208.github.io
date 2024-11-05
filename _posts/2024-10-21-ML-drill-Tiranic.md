---
layout: post
title: 머신러닝 연습, 타이타닉 데이터
subtitle: TIL Day 29
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, ML]
author: polaris0208
---

## 머신러닝 연습
>titanic 데이터셋<br>
>데이터 전처리 & 모델 적용 과정에서 추가 학습한 내용 정리 

### 데이터 전처리
1. 백분위수 지점 

```py
df.quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')
```

- q : 분위수, 소수로 표현 (예 : 75% = 0.75)
- aixs : 분위수의 값을 구할 축(0 : 행, 1: 열)
- numeric_only : 수(소수)만 대상으로할지 여부 False일 경우 datetime 및 imedelta 데이터의 분위수도 계산
- interpolation : 분위수에 값이 없을때 보간하는 방법
  - liner : i + (j - i) x 비율 [분위수 앞, 뒤 수 간격 * 비율]
  - lower : i [분위수 앞, 뒤수 중 작은수]
  - higher : j [분위수 앞, 뒤수 중 큰수]
  - midpoint : (i+j)÷2 [분위수 앞, 뒤수의 중간값]
  - nearest : i or j [분위수 앞, 뒤수중 분위수에 가까운 수]
2. 결측치 처리(최빈값 계산)

```py
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace = True)
```

- .mode() 가 아닌 .mode()[0]인 이유
  - .mode()는 series 형태로 값을 반환
    ```
    _ | 0
    -----
    0 | S
    ```
  - .model()[0]을 입력하여 첫번째 값인 "S"만 취함

3. 데이터 스케일링

```py
scaler = StandardScaler() # 평균이 0 분산 1 크기로 맞춤
X_train = scaler.fit_transform(X_train) # 평균과 편차를 찾아서 정규화
X_test = scaler.transform(X_test) 
# fit 제거 
# X_test의 데이터도 학습되기 때문
```

- fit.transform() 과 transform()의 차이
  - .fit(): 데이터에 모델을 맞추는 것 - 데이터의 평균과 분산을 학습
  - .transfotm(): fit()을 기준으로 얻은 평균과 분산에 맞춰 변형
  - .fit_transform() = .fit() 과 .transform()을 묶은 것
    - 학습된 데이터에서 얻은 평균과 분산을 기준으로 변형
  - X_test 데이터에 사용하면 평가용 데이터도 모델에 학습되는 문제 발생
  
4. **Classification Report**

```py
              precision    recall  f1-score   support

           0       0.82      0.86      0.84       105
           1       0.78      0.73      0.76        74

    accuracy                           0.80       179
   macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179
```

- 정밀도(Precision) : Positive로 예측한 경우 중 실제로 Positive인 비율이다, 즉 예측값이 얼마나 정확한가
- 재현율(Recall) : 실제 Positive인 것 중 올바르게 Positive를 맞춘 것의 비율 이다다, 즉 실제 정답을 얼마나 맞췄느냐
- Support: 각 클래스의 실제 데이터 수
- F1 Score: 정밀도와 재현율의 조화평균
- macro avg: 각 클래스별로 동일한 비중을 둔 평균
- weighted avg: 클래스의 데이터 수(support)를 고려한 평균
