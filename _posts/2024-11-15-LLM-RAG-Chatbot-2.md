---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 2
subtitle: TIL Day 54
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

# Index

>[¶ Retriever 기능](#retriever)<br>
>[¶ RAG Chain 구성](#rag-chain-구성)

# Retriever

## 패키지 관리
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `faiss-cpu-1.9.0`

## Vector Store 저장
- `vector_store.save_local("testbed/vector_store_index/faiss_index")`
  - `index.faiss`
  - `index.pkl` : 객체를 바이너리 데이터 형태로 저장하는 피클 파일

## Vector Store 불러오기
- 생성 후 바로 사용하면 생략 가능
- 이미 만들어져 로컬에 저장된 **Vector Store**를 사용하는 방법
- `vector_store = FAISS.load_local("--path--", embedding_model)`

### 확인된 문제
- `index.pkl` 파일을 불러오는 과정에서 발생하는 보안 경고로 코드 작동 중지
- **역직렬화 허용** 문제
  - **역직렬화** : `pkl` 피클파이에 저장된 데이터를 다시 **Python** 객체로 복원하는 과정
  - 악의적인 사용자가 파일을 수정하여 `pkl` 파일을 로드할 때 악성 코드가 실행되는 경우 존재
  - 신뢰할 수 있는 출처의 `pkl`만 사용하고 인터넷에서 다운로드한 경우 주의

### 해결
- 직접 생성하거나 안전한 출처의 `pkl`의 경우 역직렬화를 허용하여 경고 무시
- `allow_dangerous_deserialization=True` : 해당 코드를 인자에 추가

## Retriever 생성
- `retriever = vector_store.as_retriever()`
  - `search_type` : 검색 기준
  - `search_kwargs` : `k` 찾을 문서 개수

### 사용 파라미터
- `similarity` : 코사인 유사도 기반 검색
- `similarity_score_threshold` : `score_threshold` 이상만 검색
  - `search_kwargs` 설정
  - `score_threshold` 입력
- `mmr` : **maximum marginal search result** : 다양성을 고려한 검색
  - `search_kwargs` 설정
  - `fetch_k`: 후보 집합을 생성, 후보중에서 최종 `k`개 생성
  - `lambda_mult`: 유사도와 다양성 비중 조절
- `filter` : 메타 데이터를 기준으로 필터링, `search_kwargs`에 입력

## 코드 작성

```py
query = "RAG에 대해 이야기해주세요."

retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)
for result in results:
    print(f"Source: {result.metadata['source']} | Page: {result.metadata['page']}")
    print(f"Content: {result.page_content.replace('\n', ' ')}\n")
```

## 결과
- **RAG** 에 관련된 문장을 적절하게 가져옴
- 사용된 **pdf** 파일의 레이아웃에 맞춰 `\n` 적용되어 있는 형태로 출력

```
Source: documents/초거대 언어모델 연구 동향.pdf | Page: 8
Content: 16 특집원고  초거대 언어모델 연구 동향
Retrieval Augmented Generation (RAG) [95, 96, 97, 
98]이라 한다.
Other Tools L a M D A  [ 4 4 ]는 대화 애플리케이션에 
특화된 LLM으로, 검색 모듈, 계산기 및 번역기 등의 
외부 도구 호출 기능을 가지고 있다. WebGPT [99] 는 
웹 브라우저와의 상호작용을 통해 검색 쿼리에 사실 
기반의 답변과 함께 출처 정보를 제공 한다......
```

[¶ Top](#index)
><br>

# RAG Chain 구성

## 사용 패키지
- `openai-1.54.4`
- `langchain-openai-0.2.8`
- `langchain-core-0.3.18`
- `faiss-cpu-1.9.0`

## LLM Model 설정

```py
llm_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key)
```

## 프롬프트 작성
- `context`를 통해 전달 받은 정보를 참고
- `question` : 사용자의 질문

```py
from langchain_core.prompts import ChatPromptTemplate

contextual_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the following context."),
        ("user", "Context: {context}\\n\\nQuestion: {question}"),
    ]
)
```

## 디버깅 함수
- `invoke` 메서드 사용한 결과를 전달
- 전달한 결과를 그대로 출력

```py
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
```

## `ContextToText` 함수
- 검색한 문서를 텍스트로 변환하여 전달

```py
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        context_text = [doc.page_content.replace('\n', ' ') for doc in inputs["context"]]
        return {"context": context_text, "question": inputs["question"]}
```

## Langchain 구성
- `|` : 파이프라인 사용, 파라미터를 순서대로 전달

```py
rag_chain_debug = (
    {"context": retriever, "question": DebugPassThrough()}
    | DebugPassThrough()
    | ContextToText()
    | contextual_prompt
    | llm_model
)
```

## 결과 확인

```py
response = rag_chain_debug.invoke("RAG에 대해 이야기해주세요.")
print("Final Response:")
print(response.content)
```

### 확인된 문제
- **LLM** 모델이 답변을 제시하지 못하는 문제

### 원인
- `ContextToText` 함수의 `context_text` 생성 코드
  - `'\n'.join()` 메서드 사용 시 문자 단위로 분할되어 `context` 로 활용되지 못함

### 해결
- `join` 메서드 제거 : 답변 생성
- page_content 결과 내부 `'\n'` 제거 : 향상된 답변 생성

```
 \n 제거 전
Debug Output: RAG에 대해 이야기해주세요.

Final Response:
RAG는 Retrieval Augmented Generation의 약자로, 정보 검색 기능을 활용하여 생성 모델의 성능을 향상시키는 접근 방식입니다. 이 방법은 모델이 질문에 대한 답변을 생성할 때, 외부 데이터베이스나 검색 엔진에서 관련 정보를 검색한 후 이를 바탕으로 보다 정확하고 사실적인 응답을 생성할 수 있도록 돕습니다. RAG는 특히 대화형 AI나 질의응답 시스템에서 유용하게 사용됩니다.
```

```
\n 제거
Debug Output: RAG에 대해 이야기해주세요.

Final Response:
RAG, 즉 Retrieval Augmented Generation은 초거대 언어모델 연구의 한 동향으로, 정보 검색과 생성 과정을 결합하여 보다 정확하고 정보에 기반한 응답을 생성하는 방법입니다. 이 접근 방식은 기존의 언어 모델이 가지고 있는 한계점을 극복하고, 외부 데이터 소스에서 정보를 검색하여 그에 기반한 답변을 제공할 수 있도록 합니다. RAG는 특히 다양한 API와의 통합을 통해 계산, 번역, 검색 등의 기능을 활용하여 더 나은 성능을 발휘할 수 있습니다.
```

[¶ Top](#index)
><br>