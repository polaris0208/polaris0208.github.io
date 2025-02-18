---
layout: post
title: Matplot 라이브러리
subtitle: TIL Day 30
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, NLP]
author: polaris0208
---

## Matplotlib
> 시각화 라이브러리

### 기본 개념
- 데이터 시각화 라이브러리
- 다양한 유형의 그래프(선 그래프, 막대 그래프, 산점도 등)
- 사용자 정의가 용이하여, 그래프의 스타일과 요소를 쉽게 수정
- NumPy, Pandas와의 호환성

### 설치
`matplotlib`

```bash
pip install matplotlib
```

- 주로 함께 사용되는 패키지
1. NumPy
2. Pandas
3. SciPy : 과학적 계산을 위한 패키지로, 다양한 수학적 기능을 제공합니다.
4. Jupyter Notebook : Matplotlib의 그래프를 즉시 볼 수 있어 데이터 시각화 실습에 유용
5. Seaborn : 고급 시각화를 위한 패키지로, Matplotlib 기반

`pyplot`
- Matplotlib의 하위 모듈로, 그래프를 그릴 때 자주 사용하는 함수들을 제공

```bash
import matplotlib.pyplot as plt
```

### 기본 사용법
1. 기본적인 그래프 그리기

```py
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)  # 선 그래프 그리기
plt.title("Sample Plot")  # 제목 추가
plt.xlabel("X-axis")  # X축 레이블
plt.ylabel("Y-axis")  # Y축 레이블
plt.show()  # 그래프 출력
```

2. 여러 그래프 그리기
- 여러 데이터 세트를 한 번에

```py
plt.plot(x, y, label='Data 1')
plt.plot(x, [15, 10, 5, 1], label='Data 2')
plt.legend()  # 범례 추가
plt.show()
```

3. 스타일 조정
- 선의 색상, 스타일, 마커 등을 조정

```py
plt.plot(x, y, color='red', linestyle='--', marker='o')
```

4. 그래프 종류

```py
plt.bar(x, y)  # 막대 그래프
plt.scatter(x, y)  # 산점도
plt.hist(y)  # 히스토그램
```

5. 데이터 조건
- 데이터 타입
  - **NumPy** 배열
 `x = np.array([1, 2, 3, 4])`
  - **Pandas** 데이터 프레임
  `plt.plot(df['x'], df['y'])`
  - **List, Tuple**
> **입력** "항상 x, y가 필요한지, 허용하는 개수는 어느정도인지"
  - 특정 그래프에 따라 필요로 하는 데이터의 형식이 다름
    - 히스토그램: x값만 필요
    - 파이 차트: y값이 아닌 각 부분의 비율만 필요
- 데이터의 개수 
  - 여러 개의 데이터 포인트를 사용할 수 있으며, 그래프의 종류에 따라 시각화되는 방식이 달라짐

### 스타일 조정
> **입력** "선, 색상, 스타일, 마커 사용법 상세하게, 그 외 기능들 있다면 추가 설명"
1. 선
`linestyle = " "`

- "-" : 실선
- "--" : 점선
- "-." : 대시 점선
- ":" : 점선
2. 색
`color = "blue"`
- 이름: "red" , "blue"
- **HEXA** 코드 : `#FF5733`
- **RGB** 튜플 : (1.0, 0.0, 0.0)
3. **Marker**
- 모양
`marker = "o"`
  - "o"
  - "s": 사각형
  - "^": 삼각형 마커
  - "d": 다이아몬드 마커
- 크기
`markersize = 10` 또는 `ms = 10` 
4. **Line Width**
`limewith = 2` 또는 `lw = 2`
<br>

5. **Title,, Axis Labels, Grid**
- 제목
`plt.title('그래프 제목', fontsize =14. fontweight = 'bold')`
- 축 레이블
`plt.xlabel('X축 레비블, fontsize=12)`
- 그리드
`plt.grid(Ture)`
  - 그리드 스타일 조정
  `plt.grid(True, color = 'gray', linestyle = '--', linewidth = 0.5)`
  - 축별 그리드 설정
  `plt.grid(axis = 'x')`
    - 둘다 `axis = 'both'
    - `axis` 설정을 하면 그리드 활성화 `True`가 필요 없음
    - `plt.grid(axis = 'both', color='gray', linestyle='--', linewidth=0.5)`

6. **Legend**
- 범례: 여러 데이터 시리즈를 구분하기 위해

```py
plt.plot(x, y, label='데이터 1')
plt.plot(x, [10, 15, 20, 25], label='데이터 2')
plt.legend()  # 범례 추가
```

7. **Style Sheet**
- 그래프의 전반적인 스타일 변경
`plt.style.use('seaborn')`
- 기본 제공 스타일
  - `"ggplot"`: 
  - `"seaborn"`
  - `"classic"`: **Matplotlib** 기본 스타일
  - `"fivethirtyeight"`: **FiveThirtyEight** 블로그 스타일
  - `"dark_background"`: 어두운 배경 스타일

### 그래프 유형
#### `fig` 
- `plt.figure()`
  - `figsize` : 가로, 세로 단위 (인치)
  - `dpi`: 인치당 점의 수 - 해상도
1. **Line Plot**
- 연속적인 데이터 변화를 시각화

```py
  plt.plot(x, y)
  plt.title('Line Plot Example')
  plt.show()
```

2. **Scatter Plot**
- 두 변수 간의 관계 시각화

```py
plt.scatter(x, y)
plt.title('Scatter Plot Example')
plt.show()
```

3. **Bar Plot**
- 범주형 데이터 시각화

```py
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values)
plt.title('Bar Plot Example')
plt.show()
```

4. **Histogram**
- 데이터의 분포를 시각화
- 데이터의 빈도, `bin`(특정구간)에 속하는 데이터의 수

```py
data = np.random.randn(1000)  # 정규분포를 따르는 1000개의 랜덤 데이터
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

5. **BoxPlot**
- `patch_artist=True`: 색상 적용이 가능

```py
import matplotlib.pyplot as plt

# 데이터 준비
data = {
    '1학급': [65, 70, 75, 80, 85, 90, 95],
    '2학급': [55, 60, 65, 70, 75, 80],
    '3학급': [45, 50, 55, 60, 65, 70, 75, 80]
}

# 상자 그림 그리기
box = plt.boxplot(data.values(), labels=data.keys(), patch_artist=True)

# 색상 지정 (for문 없이)
box['boxes'][0].set_facecolor('lightblue')    # 1학급 색상
box['boxes'][1].set_facecolor('lightgreen')   # 2학급 색상
box['boxes'][2].set_facecolor('lightcoral')    # 3학급 색상

# 그래프 제목과 축 레이블 추가
plt.title("학급별 수학 시험 점수 분포")
plt.xlabel("학급")
plt.ylabel("점수")

# 그래프 보여주기
plt.show()
```

6. **Pie Chart**
- `autopct`: 각 조각의 비율을 표시하는 형식
- `%1.1f%%` : 소수점 첫째 자리까지 표시
  - `%1.1f`
  - `%%`: 백분율 기호(단일 %는 특별한 의미가 있으므로 두 개의 %를 사용)

```py
import matplotlib.pyplot as plt

# 데이터 준비
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']

# 원 그래프 그리기
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

# 그래프 제목 추가
plt.title("원 그래프")

# 그래프 보여주기
plt.axis('equal')  # 원을 유지하기 위해
plt.show()
```

7. **Heatmap**
- 2D 데이터의 값을 색으로 표현
- `cmap`: 데이터의 값을 색상으로 변환하는 방법을 정의
  - `viridis`: 파란색에서 노란색으로 변하는 색상 맵 (기본 색상 맵)
  - `plasma`: 보라색에서 노란색으로 변하는 색상 맵
  - `hot`: 검정에서 빨강, 노랑, 흰색으로 변하는 색상 맵
  - `cool`: 파란색에서 핑크색으로 변하는 색상 맵
  - `gray`: 흑백 색상 맵
  
```py
# 데이터 준비
months = ['1월', '2월', '3월', '4월', '5월', '6월', 
          '7월', '8월', '9월', '10월', '11월', '12월']
temperatures = [2, 3, 8, 15, 20, 25, 28, 27, 22, 15, 8, 3]

# 데이터 배열로 변환
data = np.array(temperatures).reshape(1, -1)  # 1행 12열 배열로 변환

# 히트맵 그리기
plt.imshow(data, cmap='hot', interpolation='nearest')

# 축 레이블 설정
plt.title("월별 평균 기온")
plt.xticks(ticks=np.arange(len(months)), labels=months)
plt.yticks([])  # y축 레이블 제거

# 색상 바 추가
plt.colorbar(label='기온 (°C)')

# 그래프 보여주기
plt.show()
```