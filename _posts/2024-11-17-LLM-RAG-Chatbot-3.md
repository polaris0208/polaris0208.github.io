---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 3
subtitle: TIL Day 56
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

## 기능 모듈화
> 작성한 코드들을 함수로 정리한 후 패키지로 구조화

### 패키지 구조

```
RAG_Module/
├── RAG_Params.py
├── PDF_Loader.py
├── VectorStore_Utils.py
└── RAG_Chain.py
```

### RAG_Params
- 파라미터를 `dataclass` 객체로 정의
- 매개변수를 최소화하여 기능 사용

#### 확인된 문제
- 기본값이 없는 파라미터가 기본값이 있는 파라미터 뒤에 배치되면 오류 발생

#### 해결
- 기본값이 있는 파라미터 확인 및 명시 후 뒤에 배치

```py
from dataclasses import dataclass

@dataclass
class RAGParams:
    KEY: str           # API Key 환경변수명
    EBD_MODEL: str     # 임베딩 모델명
    LLM_MODEL: str     # LLM 모델명, 기본값 없음
    PDF_PATH: str      # PDF 파일 경로, 기본값 없음
    SAVE_PATH: str = None  # 저장 경로 (옵션)
    IS_SAFE: bool = False  # 안전한 파일 로드 여부 (옵션)
    CHUNK_SIZE: int = 100  # 분할 크기 (기본값: 100)
    CHUNK_OVERLAP: int = 10  # 분할 중첩 크기 (기본값: 10)
```

### PDF_Loader
- `def PDFLoader(PARAMS, **kwargs):`

```
PDF 파일을 입력 받아 Document 객체 반환

PARAMS: RAGParams 객체
PDF_PATH : 사용 pdf 파일 경로
CHUNK_SIZE : 청크당 문자 수 
CHUNK_OVERLAP : 중첩시킬 문자 수
```

### VectorStore_Utils
- **Vector Store** 반환, 저장, 불러오기 기능

`def VectorStoreReturn(SPLITS, PARAMS, **kwargs):`

```
Document 객체를 임베딩하여 Vector Store 반환
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
```

`def VectorStoreSave(SPLITS, PARAMS, **kwargs):`

```
Document 객체를 임베딩하여 Vector Store 저장
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
SAVE_PATH : 저장할 경로
```

`def VectorStoreLoad(PARAMS, **kwargs):`

```
저장된 Vector Store 반환
SPLITS : Document 객체
PARAMS : RAGParams 객체
KEY : 환경변수에서 호출할 API Key 이름
EBD_MODEL : 임베딩 모델명
SAVE_PATH : Vector Store가 저장된 경로
IS_SAFE : 불러올 Vector Store이 안전한 파일인지 확인(불리언)
```

### RAG_Chain
- **RAG** 기법을 이용한 **LLM** 답변 구조 생성

`def RAGChainMake(VECTOR_STORE, PARAMS, **kwargs):`

```
VECTOR_STORE : Retriever가 검색할 벡터 스토어
KEY : 환경변수에서 호출할 API Key 이름
LLM_MODEL : 사용할 LLM 모델명
```

### 결과 확인

#### 파라미터 설정
- `params` 객체 생성 : 함수에 입력하면 필요한 매개변수만 입력됨

```py
from RAG_Module.RAG_Params import RAGParams

params = RAGParams(
    KEY= "_______",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL='gpt-3.5-turbo',
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH = None,
    IS_SAFE=True,
    CHUNK_SIZE = 100,
    CHUNK_OVERLAP = 10
)
```

#### 답변 생성

```py
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain import RAGChainMake

docs = PDFLoader(params)
vector_store = VectorStoreReturn(docs, params)
chatbot_mk1 = RAGChainMake(vector_store, params)

question = 'RAG에 대해서 설명해주세요'
response = chatbot_mk1.invoke(question)

print(response.content)
```

#### 답변 생성 결과
- 각 패키지가 문제 없이 작동
- 최소한의 청크 크기와 `gpt-3.5` 모델 사용으로 간단한 수준의 답변만 생성

```py
Load PDF.....
Load Complete....!
Split docs.....
Split Complete....!

Authenticate API KEY....
Authenticate Complete....!
Set up Embedding model....
Set up Complete....!
Initiate FAISS instance....
Return Vector Store....!

Debug Output: RAG에 대해서 설명해주세요
final response : RAG은 Retrieval Augmented Generation의 약자로, 초거대 언어모델 연구 동향 중 하나이다.
```