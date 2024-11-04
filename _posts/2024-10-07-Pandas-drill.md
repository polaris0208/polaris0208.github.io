---
layout: post
title: Pandas 데이터 연습
subtitle: TIL Day 15
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---

# Pandas 실제 활용 시 주의할 점 위주로 실습

## Kaggle 데이터 활용

```
데이터 분석 및 머신러닝에 대한 학습 플랫폼
다양한 분석 데이터와 데이터 개요 제공
```

### 데이터 전처리

```
데이터의 개요를 파악하여 데이터 전처리 계획
결측치, 이상치 제거 및 병합 방식 및 순서 등
```

### 데이터 불러오기

```
결측치 이상치를 미리 제거하면 정합성 확인이 용이
```

```py
import pandas as pd
products_df = pd.read_csv('/Users/사용자이름/myenv/study_python/kaggle_mrae/products.csv')
order_items_df = pd.read_csv('/Users/사용자이름/myenv/study_python/kaggle_mrae/order_items.csv')
orders_df = pd.read_csv('/Users/사용자이름/myenv/study_python/kaggle_mrae/orders.csv')
customers_df = pd.read_csv('/Users/사용자이름/myenv/study_python/kaggle_mrae/customers.csv')
payments_df=pd.read_csv('/Users/사용자이름/myenv/study_python/kaggle_mrae/payments.csv')
# 중복치 제거
products_df = products_df.drop_duplicates()
order_items_df = order_items_df.drop_duplicates()
orders_df = orders_df.drop_duplicates()
customers_df = customers_df.drop_duplicates()
payments_df = payments_df.drop_duplicates()
# 결측치 제거
products_df = products_df.dropna()
order_items_df = order_items_df.dropna()
orders_df = orders_df.dropna()
customers_df = customers_df.dropna()
payments_df = payments_df.dropna()
```

### 결측치 확인

```py
products_df.info()
# 개수의 차이가 존재, 결측치 의심
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32951 entries, 0 to 32950
Data columns (total 6 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   product_id             32951 non-null  object 
 1   product_category_name  32781 non-null  object 
 2   product_weight_g       32949 non-null  float64
 3   product_length_cm      32949 non-null  float64
 4   product_height_cm      32949 non-null  float64
 5   product_width_cm       32949 non-null  float64
dtypes: float64(4), object(2)
memory usage: 1.5+ MB

products_df.isnull().sum()
# 결측치 존재 Truo =1, 없음 False = 0 의 합
# 170개 존재

product_id                 0
product_category_name    170
product_weight_g           2
product_length_cm          2
product_height_cm          2
product_width_cm           2
dtype: int64
```

### 결측치 데이터 추가

```py
products_df['missing_count'] = products_df.isnull().sum(axis=1)
# 결측치 개수를 열(axis=1)에 추가
# filtered_df = products_df[products_df['missing_count'] > 0]
# 결측치가 있는 경우만 출력
# filtered_df
```

### 결측치 제거

```py
products_df = products_df.dropna()
```

### 중복치 확인 및 제거

```
중복값을 처리하지 않을 경우 merge를 진행했을 때 중복값이 대폭 증가
```

```py
customers_df.duplicated().sum()
# np.int64(3089) 3089개
customers_df = customers_df.drop_duplicates()
customers_df.duplicated().sum()
# np.int64(0) 0개
```

### 데이터 병합

```
SQL inner join
데이터 개요를 통해 확인한 공통 데이터 기반으로 병합
계층적으로 병합 실행
```

```py
merged = pd.merge(orders_df, payments_df, how='left', on='order_id')
# 첫 번째 병합 실행 후 병합 데이텅 계속 중첩하여 병합 실행
# 각각 병합 후 병합도 가능
merged = pd.merge(merged, customers_df, how='left', on='customer_id')
merged = pd.merge(merged, order_items_df, how='left', on='order_id')
merged = pd.merge(merged, products_df, how='left', on='product_id')
# 병합 후 중복치와 결측치 확인
merged.duplicated().sum()
merged.isnull().sum()
```

### 결측치 제거 주의점

```
결측치를 단순 제거하면 안됨
결측치가 발생한 이유를 확인하고 전처리
예시)order_id 한개에 order_items 여러개, payments 여러개
# 예시 구조
order_id - order_items - payments
         price/shipping - payment_value
   1            1       - price
                        - shipping
                2       - price
                        - shipping
```

### 데이터 전처리(그룹화)
* order_items의 데이터와 paymentsd의 데이터를 order_id 아래로 그룹화

#### order_items 그룹화

```py
grouped_1 = merged.groupby(by = ['order_id', 'order_item_id']).agg({'price' : 'mean', 'shippin g_charges': 'mean'})
# .agg() 다중집계 함수
grouped_1 = grouped_1.groupby(by = ['order_id']).agg({'price':'sum', 'shipping_charges': 'sum'})
grouped_1
```

#### payments 그룹화

```py
grouped_2 = merged.groupby(by = ['order_id', 'payment_sequential']).agg({'payment_value' : 'mean'})
grouped_2 = merged.groupby(by = ['order_id']).agg({'payment_value' : 'sum'})
grouped_2
```

#### 병합하여 order_id로 그룹화된 하나의 데이터 산출

```py
grouped_merge = pd.merge(grouped_1, grouped_2, how = 'left', on = 'order_id')
grouped_merge
```

#### 결과값을 이용해 데이터 정합성 확인

```py
(abs(grouped_merge['price'] + grouped_merge['shipping_charges'] - grouped_merge['payment_value']) > 0.1).sum()
# 가격과 운송료의 합이 결제가격과 같지 않은 경우 (절대값)
```

* 오차범위 벗어나는 데이터의 원인 = 부동소수점

### 데이터 추가(부피 계산)

```
merged['product_volume'] = merged['product_length_cm'] * merged['product_height_cm'] * merged['product_width_cm']
merged.head()
```

### 데이터 타입변환(시간 데이터)

```py
merged['order_delivered_timestamp'] = pd.to_datetime(merged['order_delivered_timestamp'])
merged['order_approved_at'] = pd.to_datetime(merged['order_approved_at'])
merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
merged['order_estimated_delivery_date'] = pd.to_datetime(merged['order_estimated_delivery_date'])
merged.info()
```

* object 타입을 datetime 타입으로 변경

```py
merged["delivered_time"] = merged["order_delivered_timestamp"] - merged['order_purchase_timestamp']
```

### 데이터 표준화
* 타입변환한 시간 데이터는 연산 가능
* 배달시간을 계산하여 데이터 추가
* 표준화 진행

```py
merged["delivered_time"] = merged["order_delivered_timestamp"] - merged['order_purchase_timestamp']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = merged.select_dtypes(include=['number'])
# 숫자형 데이터로 변환
scaled_data = scaler.fit_transform(merged[['delivered_time']])
scaled_data
merged['delivered_time'] = scaled_data
merged['delivered_time']
#
0        -0.428669
1        -0.428669
2        -0.428669
3         0.137950
4        -0.327153
            ...   
115718    1.029655
115719    1.312265
115720    0.488212
115721    0.488212
115722   -0.509499
Name: delivered_time, Length: 115289, dtype: float64
```

### 추가 학습 요소

* IQR 
* 부동소수점
