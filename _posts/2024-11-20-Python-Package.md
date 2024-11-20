---
layout: post
title: Python 패키지 제작
subtitle: TIL Day 59
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python, Tools]
author: polaris0208
---

> 파이썬 프로젝트를 배포할 경우<br>
> 작성한 코드를 모듈화 하여 사용할 경우

## 기본 구조
- `__init__.py`
  - 해당 디렉터리가 패키지의 일부임을 알려 주는 역할
    - `python 3.3` 부터는 없어도 패키지로 인식
    - 구번전과의 호환성을 고려해 작성
  - 설정이나 초기화 코드를 포함
- `setup.py`
  - 패키지 사용환경을 설정하는 파일
  - 핵심 내용 : 이름, 버전, 포함할 파일
  - 선택 사항 : 의존성 파일(실행에 필요한 패키지, 라이브러리 등)
- `readme.md` : 패키지 설명
- `licence` : 라이센스 정보

```
RAG_Module/
├── setup.py
├── __init__.py
├── RAG_Params.py
├── PDF_Loader.py
├── VectorStore_Utils.py
└── RAG_Chain.py
```

## `setup.py`
- `name` : `pip`에서 인식 할 이름
- `packages` : 패키지에 포함 될 파일
- `install_requires` : 필요한 패키지 및 라이브러리
  - 패키지 설치 시 자동 설치
  - 작성하지 않고 `requirements.txt` 사용 가능
  - 버전은 선택 사항
- 설치
  - 배포된 경우 : `pip install 패키지 이름`
  - 로컬에서 사용 : 패키지 디렉토리로 이동 후 `pip install .`

```py
from setuptools import setup, find_packages

setup(
    name="RAG_Module",
    version="0.1.0",
    description="RAG 작동을 위한 모듈",
    # long_description_content_type="text/markdown",
    # long_description=open('README.md').read(),
    packages=find_packages(include = ['RAG_Module', 'RAG_Module.*']),
    python_requires='>=3.6',
    install_requires=[
        'faiss-cpu==1.9.0',
        'jsonpatch==1.33',
        'jsonpointer==3.0.0',
        'langchain==0.3.7',  
        'numpy==1.26.4',
        'openai==1.54.4',
        'pypdf==5.1.0',
    ],
)
```

## `__init__.py`
- 패키지 초기화 파일
- 패키지 실행시 우선적으로 실행됨

```py
from .PDF_Loader import PDFLoader
from .RAG_Params import RAGParams, PromptParams, TemplateParams
from .VecotorStore_Utils import VectorStoreReturn, VectorStoreSave, VectorStoreLoad
from .RAG_Chain import RAGChainMake, RAG_Coversation, AutoChain
from .Prompt_Engineering import PromptSave, PromptLoad, PromptResult, PromptTemplate

__all__ = ['PDF_Loader', 'RAG_Params', 'VecotorStore_Utils', 'RAG_Chain', 'Prompt_Engineering']

Version = '0.1.0'

print(f'\n Initializing RAG_Module version {Version}.... \n')
```

### `from .PDF_Loader import PDFLoader`
- 하위 모듈에 바로 접근 할 수 있도록 초기화 파일에서 `import`
- `from RAG_Module import *` 입력으로도 같은 효과
  - 차이점 : 다른 모듈까지 모두 불러와짐

```py
# 일반적인 import 방법
from RAG_Module.PDF_Loader import PDFLoader

pdf_loader = PDFLoader()

# 초기화 파일에서 설정한 경우
import RAG_Module

pdf_loader = RAG_Module.PDFLoader()
```

### `all`
- `import *` 에 포함할 모듈 설정

### 패키지 사용 시 실행될 코드

```py
Version = '0.1.0'

print(f'\n Initializing RAG_Module version {Version}.... \n')
```

## 모듈 제작
- 실행할 코드, 사용할 함수 또는 클래스 정의

### 함수

#### 필요한 패키지, 모듈, 라이브러리 불러오기

```py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

#### 함수 선언 및 독스트링 작성
- 독스트링 : 함수 및 필요한 파라미터에 대한 설명

```py

def PDFLoader(PARAMS, **kwargs):
    """
    PDF 파일을 입력 받아 Document 객체 반환\n
    PARAMS: RAGParams 객체\n
    PDF_PATH : 사용 pdf 파일 경로\n
    CHUNK_SIZE : 청크당 문자 수 \n
    CHUNK_OVERLAP : 중첩시킬 문자 수\n
    """
```

#### 함수 코드 작성

```py
    loader = PyPDFLoader(PARAMS.PDF_PATH)
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARAMS.CHUNK_SIZE,
        chunk_overlap=PARAMS.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    print("\nLoad PDF.....")
    docs = loader.load()
    print("Load Complete....!")

    print("Split docs.....")
    splits = recursive_splitter.split_documents(docs)
    print("Split Complete....!\n")

    return splits
```

### 클래스
- `dataclass`의 클래스를 데코레이터로 사용
- 클래스 생성

```py
from dataclasses import dataclass

@dataclass
class RAGParams:
    KEY: str  # API Key 환경변수명
    EBD_MODEL: str  # 임베딩 모델명
    LLM_MODEL: str  # LLM 모델명, 기본값 없음
    PDF_PATH: str  # PDF 파일 경로, 기본값 없음
    SAVE_PATH: str = None  # 저장 경로 (옵션)
    IS_SAFE: bool = False  # 안전한 파일 로드 여부 (옵션)
    CHUNK_SIZE: int = 100  # 분할 크기 (기본값: 100)
    CHUNK_OVERLAP: int = 10  # 분할 중첩 크기 (기본값: 10)
```

## 패키지 활용
> 필요한 기능들을 패키지로 분리<br>
> 간결한 코드로 실행 가능

```py
import RAG_Module

params = RAG_Module.RAGParams(
    KEY= "MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL='gpt-3.5-turbo',
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH = None,
    IS_SAFE=True,
    CHUNK_SIZE = 1200,
    CHUNK_OVERLAP = 300
)

docs = RAG_Module.PDFLoader(params)
vector_store = RAG_Module.VectorStoreReturn(docs, params)
chatbot_mk1 = RAG_Module.RAGChainMake(vector_store, params)

question = "업스테이지의 solar 모델에 대해 설명해줘."
response = chatbot_mk1.invoke(question)

print(response.content)
```