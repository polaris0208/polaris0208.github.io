---
layout: post
title: Streamlit Basic
subtitle: TIL Day 72
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Web]
author: polaris0208
---

# Streamlit 기본 기능

## 문자열 작성
- 메서드에 따라 출력 방식 변경 가능
- 마크다운, 수식 출력 지원

```py
st.write('hello')
st.title('test')
st.header('this is test')
st.text('test text')

st.markdown('### h3')
st.latex('E = mc^2')
```

## 간단한 위젯 기능
- 사용자의 클릭으로 작동하는 위젯 생성
- 선택한 값으로 변수 선언하는 방식

```py
if st.button('click'):
  st.write('clicked')

agree_box = st.checkbox('agree?')
if agree_box is True : 
  st.write('agree')

volume= st.slider('slider', 0, 10, 50)
st.write("음악 볼륨은 " + str(volume) + "입니다.")

# 라디오 버튼 1개
# 셀렉트 버튼 여러개

gender = st.radio('gender', ['male', 'female', 'etc'])
st.write('gender is ' + str(gender))

score = st.selectbox('score', ['1', '2', '3', '4', '5'])

df = pd.DataFrame({
  'id' : ['b458011', 'b458012', 'b450802'],
  'name' : ['yu', 'ye', 'bk']
})
```

### 데이터 프레임 출력
- 데이터 프레임 선언 후 바로 도표로 출력 가능

```py
st.dataframe(df)
st.empty()
st.table(df)

st.container(border=False, height=20)
chart_data = pd.DataFrame( np.random.randn(20, 3), columns=["a", "b", "c"] ) 
st.line_chart(chart_data)

chart_data = pd.DataFrame({
    "국어": [100, 95, 80],
    "영어": [80, 95, 100],
    "수학": [95, 100, 80]
})
st.line_chart(chart_data)
```

### 사이드바 구성

```py
st.sidebar.title("옵션 선택")
page = st.sidebar.selectbox("페이지를 선택하세요", ["홈", "데이터", "시각화"])

st.write(f"현재 선택된 페이지: {page}")
```

### 컬럼 레이아웃

```py
col1, col2 = st.columns(2)
col1.write("왼쪽 컬럼")
col2.write("오른쪽 컬럼")
```

### 확장 가능 컨텐츠

```py
with st.expander("자세히 보기"):
    st.write("여기에 숨겨진 내용을 작성합니다.")
```

## 파일 업로드
- 파일 직접 선택
- 파일 끌어서 넣기

```py
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:")
    st.dataframe(df)

if uploaded_file:
    st.write("컬럼별 데이터 요약:")
    st.write(df.describe())
```

## 데이터 필터링

```py
if uploaded_file:
    column = st.selectbox("필터링할 컬럼 선택", df.columns)
    value = st.text_input(f"{column}의 값을 입력하세요")
    if value:
        filtered_df = df[df[column].astype(str).str.contains(value)]
        st.write("필터링된 데이터:")
        st.dataframe(filtered_df)
```

## 그래프

```py
if uploaded_file:
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Scatter Plot")
    st.plotly_chart(fig)

if uploaded_file:
    plt.figure(figsize=(10, 5))
    plt.hist(df[df.columns[0]], bins=20)
    st.pyplot(plt)

if uploaded_file:
    plt.figure(figsize=(10, 5))
    plt.hist(df[df.columns[3]], bins=20)  # hist = 히스토그램 = 분포를 나타내는 그림
    st.pyplot(plt)

if uploaded_file:
    chart = alt.Chart(df).mark_bar().encode(
        x=df.columns[0],
        y=df.columns[1]
    )
    st.altair_chart(chart, use_container_width=True)

if uploaded_file:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, ax=ax)
    st.pyplot(fig)
```