---
layout: post
title: Pandas 데이터 구조화
subtitle: TIL Day 15
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---

# Pandas 데이터 구조화

## Multi index 생성
* 하나 이상의 인덱스를 사용, 데이터프레임의 행과 열을 구조화

### .set_index([인덱스1, 인덱스2])

```python
import pandas as pd
data = {
    '도시': ['서울', '서울', '부산', '부산'],
    '년도': [2021, 2022, 2021, 2022],
    '인구수': [9700000, 9720000, 3400000, 3450000]
}
df = pd.DataFrame(data)
# .set_index()
df_multi_index = df.set_index(['도시', '년도'])
print(df_multi_index)
```

### pd.MultiIndex.from_tuples()
* 멀티 인덱스를 구성할 각 계층을 튜플로 부터 가져옴, name은 계층의 이름을 지정

```python
index = pd.MultiIndex.from_tuples(
  [('서울', 2021), ('서울', 2022), ('부산', 2021), ('부산', 2022)],
  names = ['도시', '년도'])
df_multi_index2 = pd.DataFrame(
{'인구수' : [9700000, 9720000, 3400000, 3450000]}, index=index)
print(df_multi_index2)
```

```
             인구수
도시 년도           
서울 2021  9700000
   2022  9720000
부산 2021  3400000
   2022  3450000
```

* 보다 상세한 데이터 조회가 가능(조건을 여러개 설정 가능)

```python
df_multi_index.loc['서울', 2021]
# 
인구수    9700000
Name: (서울, 2021), dtype: int64
```

```python
df_multi_index = df_multi_index.sort_index(ascending=True)
# 오류를 방지하기 위해 인덱스 정렬
print(df_multi_index.loc['부산':'부산'])
#
             인구수
도시 년도           
부산 2021  3400000
   2022  3450000
```

```python
# 특정 레벨 또는 레벨을 넘어서 데이터 선택 / (선택 데이터, 레벨 = 제외)
print(df_multi_index.xs('서울', level = '도시'))
#
          인구수
년도           
2021  9700000
2022  9720000
```

## Multi index 재구조화

### stack - unstack

```python
df_unstacked = df_multi_index.unstack(level = '년도')
# unstack() 인덱스를 열로 변환, stack 열을 인덱스로 변환
print(df_unstacked)
# 되돌리기 df_multi_index.stack()
#
        인구수         
년도     2021     2022
도시                  
부산  3400000  3450000
서울  9700000  9720000
```

### 그룹화
* 선정한 인덱스를 기준으로 그룹화 진행

```python
# 그룹화로 멀티인덱스 생성
data = {
    '도시': ['서울', '서울', '부산', '부산', '서울', '부산'],
    '년도': [2021, 2022, 2021, 2022, 2021, 2022],
    '인구수': [9700000, 9720000, 3400000, 3450000, 9800000, 3500000],
    '소득': [60000, 62000, 45000, 46000, 63000, 47000]
}
df = pd.DataFrame(data)
#
grouped_df = df.groupby(['도시', '년도']).mean()
print(grouped_df)
#
               인구수       소득
도시 년도                      
부산 2021  3400000.0  45000.0
   2022  3475000.0  46500.0
서울 2021  9750000.0  61500.0
   2022  9720000.0  62000.0
```

### .pivot()
* 데이터 프레임 재구조화
* 서브 데이터프레임 생성시 자주 활용

```python
data = {
    '날짜': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
    '도시': ['서울', '서울', '부산', '부산'],
    '온도': [2, 3, 6, 7],
    '습도': [55, 60, 80, 85]
}
df = pd.DataFrame(data)
pivot_df = df.pivot(index = '날짜', columns='도시', values = '온도')
# index 행, columns 열 values 값
# values 를 빼면 모든 데이터가 들어감
print(pivot_df)
#
도시          부산  서울
날짜                
2023-01-01   6   2
2023-01-02   7   3
```

## 데이터 구조 해체
* 하나하나 해체하여 긴 데이터로 만들어줌

### .melt()

```python
melted_df = pd.melt(
	df, id_vars=['날짜', '도시'], 
	value_vars =['온도', '습도'], value_name = '값')
# pd.melt(
	가져올 데이터 프레임, id_vars = [식별자-그대로 남겨둘 부분], 
    value_vars = [값으로 녹여낼 부분])
# 식별자는 데이터를 구분하는데 용이하도록 구성
print(melted_df)
#
           날짜  도시 variable   값
0  2023-01-01  서울       온도   2
1  2023-01-02  서울       온도   3
2  2023-01-01  부산       온도   6
3  2023-01-02  부산       온도   7
4  2023-01-01  서울       습도  55
5  2023-01-02  서울       습도  60
6  2023-01-01  부산       습도  80
7  2023-01-02  부산       습도  85
```

## 연습코드
### 데이터 구조화 - 해체 - 재구조화 - 변경
* 데이터 프레임 생성

```python
concert_data = {
  '콘서트' : ['dlwlrma', 'Love,Poem', 'dlwlrma', 'Love,Poem', 'dlwlrma', 'Love,Poem'],
  '도시' : ['서울', '서울', '부산', '부산', '광주', '광주'],
  '관객수' : [23154, 28574, 6091, 6756, 6541, 112926]
}
con_df = pd.DataFrame(concert_data)
print(con_df)
# 단순 나열 
         콘서트  도시     관객수
0    dlwlrma  서울   23154
1  Love,Poem  서울   28574
2    dlwlrma  부산    6091
3  Love,Poem  부산    6756
4    dlwlrma  광주    6541
5  Love,Poem  광주  112926
```

* .pivot() 이용해 구조화

```py
pivot_con = con_df.pivot(
index = '콘서트', columns = '도시', values = '관객수')
print(pivot_con)
#
도시             광주    부산     서울
콘서트                           
Love,Poem  112926  6756  28574
dlwlrma      6541  6091  23154
```

* .melt()로 해체

```py
melted_con_df = pd.melt(
con_df, id_vars = ['콘서트', '도시'], value_vars = ['관객수'])
print(melted_con_df)
#
         콘서트  도시 variable   value
0    dlwlrma  서울      관객수   23154
1  Love,Poem  서울      관객수   28574
2    dlwlrma  부산      관객수    6091
3  Love,Poem  부산      관객수    6756
4    dlwlrma  광주      관객수    6541
5  Love,Poem  광주      관객수  112926
```

* stack - unstack

```
열 데이터였던 도시명들이 도시의 항위 레벨로 이동됨
.unstack() 으로 복귀
중복된 결과가 생길 수 있다면 상호 실행이 불가능 할 수 있기에 주의
```

```py
stacked_df = pivot_con.stack()
stacked_df
#
콘서트        도시
Love,Poem  광주    112926
           부산      6756
           서울     28574
dlwlrma    광주      6541
           부산      6091
           서울     23154
dtype: int64
```

* 데이터 추가

```py
con_df['공연장'] = ['KSPO Dome', 'KSPO Dome',
	'사직 실내 체육관', '사직 실내 체육관', 
	'광주여대 유니버사이드 체육관', '광주여대 유니버사이드 체육관']
print(con_df)
#
         콘서트  도시     관객수              공연장
0    dlwlrma  서울   23154        KSPO Dome
1  Love,Poem  서울   28574        KSPO Dome
2    dlwlrma  부산    6091        사직 실내 체육관
3  Love,Poem  부산    6756        사직 실내 체육관
4    dlwlrma  광주    6541  광주여대 유니버사이드 체육관
5  Love,Poem  광주  112926  광주여대 유니버사이드 체육관
```

* 데이터 삭제(행 과 열)

```py
#열 삭제
df_drop = con_df.drop(columns = '관객수')
print(df_drop)
#
         콘서트  도시              공연장
0    dlwlrma  서울        KSPO Dome
1  Love,Poem  서울        KSPO Dome
2    dlwlrma  부산        사직 실내 체육관
3  Love,Poem  부산        사직 실내 체육관
4    dlwlrma  광주  광주여대 유니버사이드 체육관
5  Love,Poem  광주  광주여대 유니버사이드 체육관

# 행 삭제
drop_df =con_df.drop(index = 0)
print(drop_df)
#
         콘서트  도시     관객수              공연장
1  Love,Poem  서울   28574        KSPO Dome
2    dlwlrma  부산    6091        사직 실내 체육관
3  Love,Poem  부산    6756        사직 실내 체육관
4    dlwlrma  광주    6541  광주여대 유니버사이드 체육관
5  Love,Poem  광주  112926  광주여대 유니버사이드 체육관
```