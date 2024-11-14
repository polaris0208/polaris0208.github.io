---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 1
subtitle: TIL Day 53
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

> 제작 계획 및 개요를 포함하여 주요 기능 제작 및 테스트까지 진행

# Index

>[¶ 개요](#개요)<br>
>[¶ 문서 로드 기능](#문서-로드-기능)<br>
>[¶ 문서 분할 기능](#문서-분할-기능)<br>
>[¶ 문서 임베딩 기능](#문서-임베딩-기능)

## 개요

### 목표 : LLM과 RAG 기술을 활용해 사용자 질문에 답변하는 챗봇

### 구현 기능
1. **LLM**을 이용한 질문-답변 챗봇 제작
2. **PDF** 형식의 문서를 불러와 정보를 검색하는 **RAG** 구축

### 평가 환경
- **jupyter notebook**

### 사용 데이터
- 박찬준 외, 「초거대 언어모델 연구 동향」, 『정보학회지』, 제41권 제11호(통권 제414호), 한국정보과학회, 2023, 8-24
- 출처 [¶](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11610124)

### 개발환경
> 각 기능들을 제작 한 이후에 모듈화 하여 작동 평가 예정

```
LLM_RAG/
│
├── README.md : 프로젝트 설명
├── requirements.txt : 패키지 목록
├── .gitignore : 버전관리 제외 목록
├── main.ipynb : 평가 환경
├── first_lot : 1차 작동 평가
├── documents/
│   └── 초거대 언어모델 연구 동향.pdf
└── testbed/
    ├── load_documents.py : 문서 로드 기능 테스트
    ├── split_documents.py : 문서 분할 기능 테스트
    ├── vector_store.py : 분할된 문서 임베딩 테스트
    └── faiss_retriever.py : 주어진 쿼리(query)에 대한 검색 기능 테스트
```

[¶ Top](#index)
><br>

## 문서 로드 기능
> PDF 파일을 불러와 Document 객체로 반환하는 기능<br>
> 사용 가능한 패키지 확인 후 테스트를 거쳐 적합한 모델 선정

### 사용가능 패키지

#### Fitz (PyMuPDF) Loader

- 이미지, 주석 등의 정보를 가져오는 데 매우 뛰어난 성능
- 페이지 단위
- PyMuPDF 라이브러리를 기반, 고해상도 이미지 처리 적합

```py
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("example.pdf")
documents = loader.load()
```

#### PyPDFLoader

- PyPDF2 라이브러리를 사용하여 구현
- 경량, 빠르고 간단하게 텍스트 추출
- 구조화된 텍스트 추출
- 파일 크기가 큰 경우에도 효율적으로 처리

```py
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("example.pdf")
documents = loader.load()
```

#### UnstructuredPDFLoader

- Unstructured 라이브러리를 기반, 다양한 파일 형식을 처리
- 텍스트를 효율적으로 추출
- 비정형 데이터 처리에 강력한 성능을 발휘합니다.
- 특정 레이아웃이 없는 PDF 파일에서도 텍스트를 정확히 추출
- 문서의 구조에 덜 의존하는 방식

```py
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("example.pdf")
documents = loader.load()
```

#### PDFPlumberLoader

- 표와 같은 복잡한 구조의 데이터를 처리
- 텍스트, 이미지, 테이블, 필드 등 모두 추출 가능
- 문서의 메타데이터와 텍스트 간의 상호작용을 분석 

### 사용 패키지
- **PyPDFLoader**
  - 빠르고 간단하게 대용량 문서 처리 가능
  - 경량 패키지로 jupyter notebook 구동에도 적합할 것으로 판단

### 패키지 관리
- `pypdf 5.1.0` : pdf 파일 처리를 위해 설치
- `langchain_community-0.3.7` : **PyPDFLoader** 포함 라이브러리

### 코드 작성

```py
from langchain_community.document_loaders import PyPDFLoader

# 가져올 pdf 파일 주소
path = "documents/초거대 언어모델 연구 동향.pdf"

# 사용할 pdf loader 선택
loader = PyPDFLoader(path)

# pdf 파일 불러오기
docs = loader.load()

# 불러올 범위 설정
page = 0
start_point = 0
end_point = 500

# 결과 확인
print(f"페이지 수: {len(docs)}")
print(f"\n[페이지내용]\n{docs[page].page_content[start_point:end_point]}")
print(f"\n[metadata]\n{docs[page].metadata}\n")
```

### 출력 결과
- `page_content` : 인덱스별 출력 결과와 pdf파일 페이지별 내용과 일치
- `metadata` : 파일경로, 페이지 출력
  - `{'source': 'documents/초거대 언어모델 연구 동향 (1).pdf', 'page': 0}`

### 확인된 문제

> 기능 테스트 이후에 성능에 영향을 미친다 판단되면 전처리 진행 예정
- 페이지 제목이 최상단으로 이동함 

```
8 특집원고  초거대 언어모델 연구 동향
초거대 언어모델 연구 동향
업스테이지  박찬준*･이원성･김윤기･김지후･이활석...
```

- 각주 pdf 문서와 다른 위치에 출력됨
  - `ChatGPT1` 의 각주가 서론에도 붙어 출력됨

```
1. 서  론1)
ChatGPT1)와 같은 초거대 언어모델(Large Language 
Model, LLM) 의 등장으로
```

- 미주가 페이지 끝이 아닌 중간에 포함됨

```
이 모든 변화의 중심에는 ‘scaling law’라는 
* 정회원
1) https://openai.com/blog/chatgpt
학문적인 통찰이 있다
```

[¶ Top](#index)
><br>

## 문서 분할 기능
> 사용하는 데이터에서는 결과상 큰 차이가 없어 RecursiveCharacterTextSplitter 사용

### 사용 가능 패키지
- **CharacterTextSplitter**
  - 기본적인 분할 방식
  - 구분자를 기준으로 청크 단위로 분할
- **RecursiveCharacterTextSplitter**
  - 단락-문장-단어 순서로 재귀적으로 분할
  - 여러번의 분할로 작은 덩어리 생성
  - 텍스트가 너무 크거나 복잡할 때 유용

### 사용 파라미터
1. `separator`
- 텍스트 분할을 위한 구분자로 사용되는 문자열
- 타입: 문자열
- `"\n\n"`: 두 개의 개행 문자 (기본값)
- `"\n"`: 한 개의 개행 문자
- `" "`: 공백
- `","`: 쉼표
- `"\t"`: 탭

2. `chunk_size`
- 분할 후 각 덩어리의 최대 크기
  - 기준 : 문자수
- 타입: 정수
3. `chunk_overlap`
- 덩어리 간 겹치는 부분의 크기
  - 문장이 끊겨서 의미를 알 수 없는 경우 보완
- 타입: 정수
4. `length_function`
- 타입: 함수
  - 기본 : len 함수
  - 필요한 함수를 작성하여 적용 가능
5. `is_separator_regex`
- 구분자의 정규 표현식인지 여부
- 타입: 불리언
  - `True`: 구분자를 정규 표현식으로 처리
  - `False`: 구분자를 문자열로 처리 

### 패키지 관리
- `langchain 0.3.7`

### 코드 작성

#### 파라미터 설정
> 분할된 결과를 확인하여 맥락이 유지되는지 여부를 확인 해 가며 조정

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEPERATOR = "\n"
```

#### CharacterTextSplitter

```py
from langchain.text_splitter import CharacterTextSplitter

# 문서 분할기 설정
splitter = CharacterTextSplitter(
    separator=SEPERATOR,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=True,
)

# 문서 분할
c_splits = splitter.split_documents(docs)

# 결과 확인
print(
    f"c_splits\n길이 : {len(c_splits)}\n 결과 확인 : {c_splits[CHUNK_INDEX].page_content}"
)
```

#### RecursiveCharacterTextSplitter

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 분할기 설정
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# 문서 분할
r_splits = recursive_splitter.split_documents(docs)

# 결과 확인
print(
    f"r_splits\n길이 : {len(r_splits)}\n 결과 확인 : {r_splits[CHUNK_INDEX].page_content}"
)
```

#### Sample 테스트

```py
def sample_printer(splits_1, splits_2, n):
    for i in range(n):
        print(f"Sample 생성중...")
        print(f"Sample_1 {i} \n {splits_1[i].page_content}")
        print(f"Sample_2 {i} \n {splits_2[i].page_content}")
```

### 출력결과
> 원본 문서와 대조하여 의미나 맥락이 유지되는 크기로 결정

#### 파라미터 1

```py
CHUNK_INDEX = 1
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
SEPERATOR = "\n\n"
```

- **CharacterTextSplitter**
  - 한 페이지 분량 출력
    - 본문에 `'\n\n'` 구분자로 나뉘는 부분이 없음
- **RecursiveCharacterTextSplitter**
  - 한 문단 분량 출력

#### 파라미터 2
- 구분자 `"\n"` 로 변경
- 결과가 같아짐

```py
CHUNK_INDEX = 1
CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
SEPERATOR = "\n"
```

```py
# RecursiveCharacterTextSplitter
c_splits
길이 : 121

# CharacterTextSplitter
r_splits
길이 : 121
```

- **CharacterTextSplitter**
  - 한 문단 분량 출력
- **RecursiveCharacterTextSplitter**
  - 한 문단 분량 출력

#### 파라미터 3 
- 한 단락의 절반 분량

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10
SEPERATOR = "\n"
```

#### 파라미터 4
- 한 단락 분량
- 중첨 : 2~3 문단으로 맥락이 이어지게 설정
- 해당 파라미터로 진행

```py
CHUNK_INDEX = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEPERATOR = "\n"
```

[¶ Top](#index)
><br>

## 문서 임베딩 기능
> 60% 이상이면 고성능으로 평가되는 것으로 확인, 제작은 002 모델로 진행, 추후 3 모델로 테스트

### 모델 정보 [¶](https://platform.openai.com/docs/guides/embeddings/)
- **MTEB bench** [¶](https://github.com/embeddings-benchmark/mteb)
  - 허깅페이스의 '대량 텍스트 임베딩 벤치마크 리더보드(MTEB)' 텍스트 검색 평가
- 사용모델 : `text-embedding-ada-002`

```
|------ 모델 ----------|-pages/$-|-MTEB-|
text-embedding-3-small	62,500	 62.3%
text-embedding-3-large	9,615	  64.6%
text-embedding-ada-002	12,500	 61.0%
```

### 패키지 관리
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `faiss-cpu-1.9.0`

### API Key 설정
- 환경변수에서 **API Key** 호출

```py
import os
import openai

openai.api_key = os.environ.get("________")
```

### 임베딩 모델 설정
- 모델 지정
- **API Key** 입력

```py
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=openai.api_key
)
```

### FAISS Vectorstore 생성
> 두 방법 모두 테스트 후 비교 예정

#### FAISS 객체 초기화 방식
- 더 많은 설정 가능
- 대용량 문서 처리가 요구되거나 성능을 최대한으로 끌어올려야 할 상황
- 인덱스나, 임베딩이 계산되어 있는 경우 빠르게 진행
- `docstore` :각 벡터와 연관된 문서 데이터를 메모리에 저장
- `index_to_docstore_id`: 특정 벡터가 어떤 문서와 연관되는지를 저장
- `Sample Text` : 임베딩 벡터의 길이를 확인하기 위한 샘플 텍스트
  - 실제 데이터의 임베딩 길이와 동일하게 설정하는 데 도움
  - 임의 문장으로 설정

```py
index = faiss.IndexFlatL2(len(embedding_model.embed_query("Sample Text")))
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

#### `from_documents` 방식
- 간단하고 빠른 방식
- 일반적인 상황에서 많이 사용

```py
vectorstore = FAISS.from_documents(
  documents=splits, embedding=embedding_model
  )
```

#### UUID : Universally Unique Identifier
- 고유한 **ID**를 생성
- 128비트 숫자로, 이를 통해 거의 중복 없이 고유한 식별자를 생성
- 여러 문서를 처리하거나 분석하는 과정에서, 각 문서를 쉽게 추적하고 연결하려면 고유 식별자가 중요
- **metadata** 에 추가하는 방식 : 검색 시 참고
- **UUDI** 인덱스를 만들어 벡터 스토어에 등록하는 방식
  - 벡터 스토어의 고유 식별자로 사용 
  - 검색 시 UUID 기반으로 문서를 추적

```py
from uuid import uuid4

# 메타 데이터에 추가
for split in splits:
    split.metadata["uuid"] = str(uuid4())
vector_store.add_documents(documents=documents)

# UUID 인덱스 추가
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```
