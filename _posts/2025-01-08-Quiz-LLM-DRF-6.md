---
layout: post
title: Quiz LLM 공식문서 RAG 구축
subtitle: TIL Day 107
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

> 공식문서 데이터를 활용하여 벡터 **DB**를 생성하고 리트리버를 제작하여 **RAG** 시스템 구축

## 의존성
- `pickle` : `binary` 데이터 작성, 벡터스토어 저장에 사용
- `FAISS` : `cpu` 버전 사용
- `BM25Retriever` : 현재 `community` 버전만 지원

```py
import os
import openai
import pickle
import sqlite3
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

## 기존 `DB`에서 공식문서 데이터 추출
- `conn = sqlite3.connect()` : **SQLite** 데이터베이스 연결 (파일명: `db.sqlite3`)
- `cur = conn.cursor()` : 커서 생성
- `cur.execute()` : **SQL** 쿼리 실행
- `cur.fetchall()` : 결가 가져오기
- `.close()` : 연결 종료

```py
openai.api_key = os.environ.get("MY_OPENAI_API_KEY")
EMBED = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai.api_key)

data = []
conn = sqlite3.connect(".../db.sqlite3")
cur = conn.cursor()
category_value = "Django"
cur.execute("SELECT * FROM chatbot_documents WHERE title = ?", (category_value,))

filtered_rows = cur.fetchall()
for row in filtered_rows:
    data.append(row[2])

cur.close()
conn.close()
```

## 데이터 분할
- `CHUNK_SIZE = 2000` : 설명, 예시, 참고자료 등 한가지 내용별로 정보가 많은 공식문서의 특징을 고려하여 사이즈를 크게 설정
  - 데이터의 양이 많아 사이즈를 작게 할 경우 벡터 **DB**의 용량도 크게 증가
  - **Github Push** 파일당 제한 용량 100**mb** 초과 
-  `seen_texts = set()` : 이미 처리한 데이터는 `set()`에 넣어 중복 방지

```py
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

docs = []

def split_texts(texts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=texts)]
    return recursive_text_splitter.split_documents(documents)

seen_texts = set()

for doc in data:
    # 이미 같은 내용이 있다면 제외
    if doc not in seen_texts:
        seen_texts.add(doc)
        docs += split_texts(doc)

# Document(metadata={}, page_content='Getting started | Django documentation | Django Django The web framework for perfectionists with deadlines. Toggle theme...resources such as a list of')
```

## 벡터스토어 생성

### FAISS
- `llow_dangerous_deserialization=True` : 역직렬화 허용 설정
  - 피클 파일이 악의적으로 수정될 경우에 대한 경고 제거

```py
vectorstore = FAISS.from_documents(documents=docs, embedding=EMBED)
vectorstore.save_local("...VDB/Django/faiss_index")

# 불러오기 예시
new_vector_store = FAISS.load_local("...VDB/Django/faiss_index", EMBED, allow_dangerous_deserialization=True)
```

### BM25
- 키워드 기반 백터스토어

```py
with open('...VDB/Django/bm25_retriever.pkl', 'wb') as f:
    pickle.dump(retriever, f)

with open('...VDB/Django/bm25_retriever.pkl', 'rb') as f:
    retriever = pickle.load(f)
```

## 리트리버 생성

### 의존성
- `EnsembleRetriever` : 여러 종류의 리트리버를 결합하여 사용

```py
import pickle
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
```

### `EnsembleRetriever` 생성
- 백터스토어를 불러와 결합
- `CATEGORY` : 카테고리별로 분류된 백터스토어를 찾아 불러옴
- `weights` : 리트리버별 비중 정도를 설정

```py 
openai.api_key = os.environ.get("MY_OPENAI_API_KEY")
EMBED = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai.api_key)

def get_retriever(CATEGORY):
    BASE_DIR = "chatbot/VDB/"
    PATH = BASE_DIR + CATEGORY
    faiss_db = FAISS.load_local(PATH + "/faiss_index", EMBED, allow_dangerous_deserialization=True)
    faiss = faiss_db.as_retriever()
    with open(PATH + '/bm25_retriever.pkl', 'rb') as f:
      bm25 = pickle.load(f)
    retriever = EnsembleRetriever(
                retrievers=[faiss, bm25],
                weights=[0.3, 0.7]
            )
    return retriever
```