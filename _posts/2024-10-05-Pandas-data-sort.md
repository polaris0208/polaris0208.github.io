---
layout: post
title: Pandas 데이터 정렬
subtitle: TIL Day 13
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---
# Pandas 데이터 정렬
## .sort_values()

```python
import pandas as pd
data = {
  'name' : ['홍지혜', '이아희', '장나영', '이시연'],
  'age' : [25, 27, 23, 24],
  'position' : ['Q', 'W', 'E', 'R'],
  'instrument' : ['drum', 'bass', 'guitar', 'vocal']
}
df = pd.DataFrame(data)
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
    </tr>
  </tbody>
</table>
</div>

### 나이로 정렬

```python
sorted_df = df.sort_values(by = 'age') 
```

### 이름으로 정렬

```python
sorted_df2 = df.sort_values(by = 'name', ascending = False)
# 기본 오름차순 
# ascending = False 내림차순
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
    </tr>
  </tbody>
</table>
</div>

### 여러 조건을 적용한 정렬

```python
data2 = {
  'name' : ['홍지혜', '이아희', '장나영', '이시연', '바위게'],
  'age' : [25, 27, 23, 24, 27],
  'position' : ['Q', 'W', 'E', 'R', 'Fan'],
  'instrument' : ['drum', 'bass', 'guitar', 'vocal', 'light stick']
}
df2 = pd.DataFrame(data2)
```

```python
sorted_df3 = df2.sort_values(by = ['age', 'name'], ascending = [False, True])
#  나이는 내림차순, 이름은 오름차순(앞에 있는 조건이 우선 반영)
sorted_df3
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>바위게</td>
      <td>27</td>
      <td>Fan</td>
      <td>light stick</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
    </tr>
  </tbody>
</table>
</div>

```
# 결과 = 나이를 기준으로 내림차순 후, 같은 나이인 경우 이름을 기준으로 오름차순으로 정렬
```

### .sort_index()

```python
sorted_df3.sort_index() 
#인덱스를 기준으로 정렬
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>바위게</td>
      <td>27</td>
      <td>Fan</td>
      <td>light stick</td>
    </tr>
  </tbody>
</table>
</div>

### .merge()

```
SQL inner join과 유사, 공통 열을 기준으로 병합
```

```python
data3 = {
  'name' : ['홍지혜', '이아희', '장나영', '이시연'],
  'alias' : ['Chodan', 'Magenta', 'Hina', 'Siyeon']
}
df3 = pd.DataFrame(data3)
merged_df = pd.merge(df2, df3, on ='name')
# 공통되지 않는 데이터('바위게')는 지우고 병합
merged_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
      <th>alias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
      <td>Chodan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
      <td>Magenta</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
      <td>Hina</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
      <td>Siyeon</td>
    </tr>
  </tbody>
</table>
</div>

### .merge() 조건 설정

```python
merge_df2 = pd.merge(df2, df3, on = 'name', how='left')
```

```
# 왼쪽을 데이터 유지, 없는 값은 NaN 으로 표시
# 모든 행을 유지하려면 how = outer, 겹치는 부분만 inner
# sql left join
# right 도 가능
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
      <th>alias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
      <td>Chodan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
      <td>Magenta</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
      <td>Hina</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
      <td>Siyeon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>바위게</td>
      <td>27</td>
      <td>Fan</td>
      <td>light stick</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

#### merge 주의점  = 중복값이 생길 수 있음

### .concat()

```
# 데이터 프레임을 서로 연결
```

```python
data4 = {
  'name' : ['김계란', '차정원', '고윤하', '전소연'],
  'position' : ['producer', 'teacher', 'mentor', 'director']
}
df4 = pd.DataFrame(data4)
```

#### 행으로 연결

```python
concat_df = pd.concat([df, df4], axis = 0) # 행단위
concat_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25.0</td>
      <td>Q</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27.0</td>
      <td>W</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23.0</td>
      <td>E</td>
      <td>guitar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24.0</td>
      <td>R</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>0</th>
      <td>김계란</td>
      <td>NaN</td>
      <td>producer</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차정원</td>
      <td>NaN</td>
      <td>teacher</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>고윤하</td>
      <td>NaN</td>
      <td>mentor</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>전소연</td>
      <td>NaN</td>
      <td>director</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


#### 열로 연결

```python
concat_df2 = pd.concat([df, df4], axis = 1) # 열단위
concat_df2
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>position</th>
      <th>instrument</th>
      <th>name</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>Q</td>
      <td>drum</td>
      <td>김계란</td>
      <td>producer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이아희</td>
      <td>27</td>
      <td>W</td>
      <td>bass</td>
      <td>차정원</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>2</th>
      <td>장나영</td>
      <td>23</td>
      <td>E</td>
      <td>guitar</td>
      <td>고윤하</td>
      <td>mentor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이시연</td>
      <td>24</td>
      <td>R</td>
      <td>vocal</td>
      <td>전소연</td>
      <td>director</td>
    </tr>
  </tbody>
</table>
</div>

### .set_index() & .join()

```
인덱스를 설정한 후 정렬, 해당 인덱스를 기준으로 데이터 병합
```

```python
df = df.set_index('position')
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>instrument</th>
    </tr>
    <tr>
      <th>position</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Q</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>drum</td>
    </tr>
    <tr>
      <th>W</th>
      <td>이아희</td>
      <td>27</td>
      <td>bass</td>
    </tr>
    <tr>
      <th>E</th>
      <td>장나영</td>
      <td>23</td>
      <td>guitar</td>
    </tr>
    <tr>
      <th>R</th>
      <td>이시연</td>
      <td>24</td>
      <td>vocal</td>
    </tr>
  </tbody>
</table>
</div>

```python
df5 = pd.DataFrame({'from' : ['twitch', 'twitch', 'tiktok', 'NMB48']}, 
index = ['Q', 'W', 'E', 'R'])
df.join(df5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>instrument</th>
      <th>from</th>
    </tr>
    <tr>
      <th>position</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Q</th>
      <td>홍지혜</td>
      <td>25</td>
      <td>drum</td>
      <td>twitch</td>
    </tr>
    <tr>
      <th>W</th>
      <td>이아희</td>
      <td>27</td>
      <td>bass</td>
      <td>twitch</td>
    </tr>
    <tr>
      <th>E</th>
      <td>장나영</td>
      <td>23</td>
      <td>guitar</td>
      <td>tiktok</td>
    </tr>
    <tr>
      <th>R</th>
      <td>이시연</td>
      <td>24</td>
      <td>vocal</td>
      <td>NMB48</td>
    </tr>
  </tbody>
</table>
</div>

### .groupby()

```python
import pandas as pd
data = {
  '이름': ['김도영', '로하스', '송성문', '김도영', '로하스', '송성문'],
  '출루' : ['안타', '안타', '안타', '볼넷', '볼넷', '볼넷'],
  '개수' : [189, 188, 179, 66, 88, 64]
}
df = pd.DataFrame(data)
grouped = df.groupby('이름')
# 이름으로 그룹화
grouped['개수'].mean()
# 개수의 평균을 구함
#
이름
김도영    127.5
로하스    138.0
송성문    121.5
Name: 개수, dtype: float64
```

```python
grouped_multi = df.groupby(['이름', '출루'])['개수'].sum()
# 이름과 출루로 그룹화 후 개수의 각 합을 구함
grouped_multi
#
이름   출루
김도영  볼넷     66
     안타    189
로하스  볼넷     88
     안타    188
송성문  볼넷     64
     안타    179
Name: 개수, dtype: int64
```

### .agg()

```
.agg() 다중집계작업 메서드
# aggregate 
.apply()와 비슷하지만 여러개의 함수를 동시에 적용 가능
df.agg(func=None, axis=0, args, kwargs) 형태
```

```python
grouped['개수'].agg(['sum','mean'])
# 합과 평균을 동시에 적용
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>김도영</th>
      <td>255</td>
      <td>127.5</td>
    </tr>
    <tr>
      <th>로하스</th>
      <td>276</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>송성문</th>
      <td>243</td>
      <td>121.5</td>
    </tr>
  </tbody>
</table>
</div>

### .pivot_table()

```python
pivot = pd.pivot_table(df, index = '이름', columns = '출루', values ='개수', aggfunc = ['sum','mean'], fill_value=0, margins = True)
# 행, 열, 열에 들어갈 데이터, 집계방식, 결측치에 넣을 값,  margins 을 통해 합계 표시 여부
pivot #없는 데이터는 NaN
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">sum</th>
      <th colspan="3" halign="left">mean</th>
    </tr>
    <tr>
      <th>출루</th>
      <th>볼넷</th>
      <th>안타</th>
      <th>All</th>
      <th>볼넷</th>
      <th>안타</th>
      <th>All</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>김도영</th>
      <td>66</td>
      <td>189</td>
      <td>255</td>
      <td>66.000000</td>
      <td>189.000000</td>
      <td>127.5</td>
    </tr>
    <tr>
      <th>로하스</th>
      <td>88</td>
      <td>188</td>
      <td>276</td>
      <td>88.000000</td>
      <td>188.000000</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>송성문</th>
      <td>64</td>
      <td>179</td>
      <td>243</td>
      <td>64.000000</td>
      <td>179.000000</td>
      <td>121.5</td>
    </tr>
    <tr>
      <th>All</th>
      <td>218</td>
      <td>556</td>
      <td>774</td>
      <td>72.666667</td>
      <td>185.333333</td>
      <td>129.0</td>
    </tr>
  </tbody>
</table>
</div>

#### 특정 열에 각각 다른 집계 적용

```python
agg_reult = df.groupby('이름').agg({'개수' : ['sum', 'max', 'min']})
# 이름으로 그룹화, 개수에 각각 합, 최댓값, 최솟값을 적용해 조회
agg_reult
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">개수</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>김도영</th>
      <td>255</td>
      <td>189</td>
      <td>66</td>
    </tr>
    <tr>
      <th>로하스</th>
      <td>276</td>
      <td>188</td>
      <td>88</td>
    </tr>
    <tr>
      <th>송성문</th>
      <td>243</td>
      <td>179</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>
  

```python
df.groupby('이름').agg('sum').sort_values(by='개수')
# 이름으로 그룹화 값의 합을 조회하고 개수의 오름차순으로 정렬
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>출루</th>
      <th>개수</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>송성문</th>
      <td>안타볼넷</td>
      <td>243</td>
    </tr>
    <tr>
      <th>김도영</th>
      <td>안타볼넷</td>
      <td>255</td>
    </tr>
    <tr>
      <th>로하스</th>
      <td>안타볼넷</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>

### 결측치, 이상치, 중복값 처리

```
데이터 전처리, 결측치, 이상치 처리, 인코딩 등
ai 발전으로 비정형 데이터 처리 중요성이 커짐
```

```python
import pandas as pd
data = {
  'name' : ['아이유', '이지금', '이지은', '이지동'],
  'age' : [16, 7, 31, None],
  'job' : ['celebrity', 'youtuber', 'celebrity', 'office worker']
  }
df = pd.DataFrame(data)
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>job</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>아이유</td>
      <td>16.0</td>
      <td>celebrity</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이지금</td>
      <td>7.0</td>
      <td>youtuber</td>
    </tr>
    <tr>
      <th>2</th>
      <td>이지은</td>
      <td>31.0</td>
      <td>celebrity</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이지동</td>
      <td>NaN</td>
      <td>office worker</td>
    </tr>
  </tbody>
</table>
</div>

#### 결측치 확인

```python
df.isna().sum() 
# 결측치인 경우의 합
# 결측치인 경우 = True = 1
#
name    0
age     1
job     0
dtype: int64
# age 컬럼에 결측치 1개 존재
```

```python
df.loc[(df['job'] == 'celebrity') & (~df['age'].isna()), 'name']
# 직업이 유명인이고 나이가 결측치가 아닌 사람의 이름을 출력
#
0    아이유
2    이지은
Name: name, dtype: object
```

#### 결측치 제거
```python
df_dropped = df.dropna(axis = 0) # axis = 0 행삭제, 1 열 삭제
df_dropped # 결측치가 있는 행 제거
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>job</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>아이유</td>
      <td>16.0</td>
      <td>celebrity</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이지금</td>
      <td>7.0</td>
      <td>youtuber</td>
    </tr>
    <tr>
      <th>2</th>
      <td>이지은</td>
      <td>31.0</td>
      <td>celebrity</td>
    </tr>
  </tbody>
</table>
</div>

#### 결측치 대체

```python
df_filled = df.fillna('모름')
df_filled
#
df['age'] = df['age'].fillna(df['age'].mean())
# 결측치를 나이의 평균값으로 채움
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>job</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>아이유</td>
      <td>16.0</td>
      <td>celebrity</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이지금</td>
      <td>7.0</td>
      <td>youtuber</td>
    </tr>
    <tr>
      <th>2</th>
      <td>이지은</td>
      <td>31.0</td>
      <td>celebrity</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이지동</td>
      <td>모름</td>
      <td>office worker</td>
    </tr>
  </tbody>
</table>
</div>

#### 결측치 보간 (주변값 참조하여 결측치 보완)

```python
data = {
    '날짜': pd.date_range('2024-01-01', periods=5),
    '온도': [20, 22, None, 24, 25]
}
df2 = pd.DataFrame(data)
df2['온도'] = df2['온도'].interpolate()
# .interplate() 보간법(기본값은 선형보간법)
df2
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>날짜</th>
      <th>온도</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-01-01</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-01-02</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-01-03</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-01-04</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-05</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
data3 = {
  'name' : ['아이유', '이지금', '이지은', '이지동'],
  'age' : [16, 7, 31, None],
  'job' : ['celebrity', 'youtuber', 'celebrity', 'office worker']
  }
df3 =pd.DataFrame(data3)

df2.loc[(df2['job'] == 'office worker') & (df2['age'].isna()), 'age'] = 20
# 직업이 회사원이고 나이가 결측치인 것은 20으로 대체
또는

def fill_nan_age(x): 
  if pd.isna(x):
    return 20
  return x
# 나이 데이터에 결측치를 20으로 채워라
df3['age'] = df3['age'].apply(fill_nan_age)
print(df3)

#
  name   age            job
0  아이유  16.0      celebrity
1  이지금   7.0       youtuber
2  이지은  31.0      celebrity
3  이지동  20.0  office worker
```

>결측치를 채우는 방법은 간단, 어떻게 채우는지가 중요
나이를 단순히 평균으로 취하는 것과 직업(예를 들어 학생)을 참고하여 채우는 것은 차이가 있다 