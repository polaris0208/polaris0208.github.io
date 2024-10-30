---
layout: post
title: Pandas 데이터 관리
subtitle: TIL Day 12
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python_Library]
author: polaris0208
---
# Pandas 데이터 관리

```python
import pandas as pd
df = pd.read_csv("/Users/사용자이름/myenv/scv/lol_users.csv")
df #"파일위치/파일명"
```

## 데이터 조회
* 행 개수 설정 후 조회 (기본값=5)

```pyhton
df.head(3) 
# 위에서 부터 행 3개만큼 조회
```

* 뒤에서 부터 조회 (기본값=5)

```python
df2.tail()
```

* url을 통한 자료 조회(예시 자료 = titanic)

```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df2 = pd.read_csv(url) 
```

* DataFrame 조회

```python
data = {
  "이름" : ["Xavi", "Iniesta", "Busquets"],
  "출생" : [80, 84, 88],
  "포지션" : ["CM", "AM", "DM"]
}
studydataframe = pd.DataFrame(data)
studydataframe
```

* 무작위 조회

```python
df2.sample(5) 
#행 5개 랜덤으로 가져옴
```

* 기본 정보 확인
non-null count 결측치(측정이 되지 않는 값)

```python
df2.info()
``` 

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

* 기초 통계량 조회

```python
df2.describe()
```

* 배열 조회

```python
df2.shape
(891, 12)
#행 열
```

* 열 조회

```python
df2.columns
# 
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
```

* 열의 데이터 조회

```python
# 열의 데이터 조회
for i in df2.columns:
  print(df2[i])
  #[] 사용 

# 열에 대한 통계
for i in df2.columns:
  print(df2[i].describe)
```

* 행 열의 위치를 지정하여 데이터 확인

```python
print(df2.iloc[0, 1]) #숫자로만 지정 
# 0
df2.loc[0, 'Name'] 
# 라벨기반 - 이름을 기준으로 선택 [행, 열] 여러개는 [] - list 사용
# 'Braund, Mr. Owen Harris'
```

* 단일 열

```python
print(df2['Name'])
```

```
0                                Braund, Mr. Owen Harris
1      Cumings, Mrs. John Bradley (Florence Briggs Th...
                             ...# 생략                  
889                                Behr, Mr. Karl Howell
890                                  Dooley, Mr. Patrick
Name: Name, Length: 891, dtype: object
```

* 여러 열

```python
print(df2[['Name', 'Ticket']])
``` 

```                                                       
                                                            Name  \
Ticket                                                                
A/5 21171                                   Braund, Mr. Owen Harris   
PC 17599          Cumings, Mrs. John Bradley (Florence Briggs Th...   
STON/O2. 3101282                             Heikkinen, Miss. Laina   
113803                 Futrelle, Mrs. Jacques Heath (Lily May Peel)   
373450                                     Allen, Mr. William Henry   
...                                                             ...   
211536                                        Montvila, Rev. Juozas   
112053                                 Graham, Miss. Margaret Edith   
W./C. 6607                 Johnston, Miss. Catherine Helen "Carrie"   
111369                                        Behr, Mr. Karl Howell   
370376                                          Dooley, Mr. Patrick   

[891 rows x 2 columns] # 2개 열
```

## index 설정
* 선택한 열-column-을 기준으로 데이터 정리

```python
df2.set_index('Name')
```

* index 설정(재배열)

```python
df2.set_index('Ticket', inplace = True, drop = False)
# inplace 알아서 재배열
# drop 인덱스로 전환되는 열의 데이터를 숨김
```

* 슬라이싱(문자열)

```python
df2.loc['A/5 21171':'373450', 'Name'] 
#시작:마지막 - 마지막도 포함 # 숫자로 되어 있어도 문자형
```

```
Ticket
A/5 21171                                     Braund, Mr. Owen Harris
PC 17599            Cumings, Mrs. John Bradley (Florence Briggs Th...
STON/O2. 3101282                               Heikkinen, Miss. Laina
113803                   Futrelle, Mrs. Jacques Heath (Lily May Peel)
373450                                       Allen, Mr. William Henry
Name: Name, dtype: object
```

* 슬라이싱(위치)

```python
df2.iloc[0:4, 3] 
# 마지막 값 제외
# 문자열과 같이 다섯번째 행까지 지정 -> 결과는 4번째까지 출력
```

```
Ticket
A/5 21171                                     Braund, Mr. Owen Harris
PC 17599            Cumings, Mrs. John Bradley (Florence Briggs Th...
STON/O2. 3101282                               Heikkinen, Miss. Laina
113803                   Futrelle, Mrs. Jacques Heath (Lily May Peel)
Name: Name, dtype: object
```

* 행 조회(슬라이싱에서 열 제외)

```python
df2.loc["A/5 21171"] # 단일 행
```

```
PassengerId                          1
Survived                             0
Pclass                               3
Name           Braund, Mr. Owen Harris
Sex                               male
Age                               22.0
SibSp                                1
Parch                                0
Ticket                       A/5 21171
Fare                              7.25
Cabin                              NaN
Embarked                             S
Name: A/5 21171, dtype: object
```

* 여러 행 조회

```python
df2.loc['A/5 21171':'373450']
```

```python
df2[0:5]
 # 슬라이싱을 통한 행 지정
```

## 데이터 필터링

```python
titanic[
  (titanic["Age"] >= 25) & 
  (titanic["Age"] <= 60) & 
  (titanic["Survived"])
  ].loc[:,"Age"]
# 조건부 데이터 필터링
# 연산자 해석 혼동 방지를 위해 ()로 묶음
# pandas -> and = & , or = |, not - ~
```

```
1      38.0
2      26.0
3      35.0
8      27.0
11     58.0
       ... 
871    47.0
874    28.0
879    56.0
880    25.0
889    26.0
Name: Age, Length: 167, dtype: float64
```

* 특정값을 포함 

```python
titanic[~titanic["Sex"].isin(["male"])] 
# isin() 특정 값 포함
# ~ 사용하여 반댓값 조회
```

## 데이터 타입 변환

```
1. 데이터의 정제 및 일관화
2. 메모리 절약
3. 데이터 처리를 위한 사전 작업
```

* 타입 확인

```python
titanic.dtypes
```

```
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object
```

* 타입 변경

```python
titanic["Age"] = titanic["Age"].astype(int)
titanic["Embarked"] = titanic["Embarked"].astype("category")
```

## 결측치 해결

```python
titanic["Age"].fillna(titanic["Age"].mean(), inplace = True)
# .fillna() 결측치를 적절한 값으로 변환 - 들어갈 값은 설정필요
# .mean() 평균값
# inplace 데이터프레임 원본 변경 여부
```