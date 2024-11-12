---
layout: post
title: Langchain 연습
subtitle: TIL Day 51
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---
# Index
>[¶ 환경 설정](#환경-설정)<br>
>[¶ Humman Message 활용](#humman-message-활용)<br>
>[¶ 프롬프트 템플릿](#프롬프트-템플릿)<br>
>[¶ 파이프라인](#파이프라인)<br>
>[¶ Vector DB 활용](#vector-db-활용)<br>
>[¶ RAG 연결](#rag-연결)

## 환경 설정
- `faiss` : `cpu`, `gpu` 옵션 존재 - 실행결과 `gpu` 옵션 확인 불가
- `numpy` : 1.26.0 이하 버전만 호환

```bash
!pip install langchain langchain-openai faiss-cpu 
!pip install --upgrade numpy==1.26.0
```

## Humman Message 활용
- `HummanMessage`
  - 사람의 메시지를 정의 및 구조화 
  - 사람이 입력한 대화 내용을 구조화 및 모델에 전달
- `meta data` 
  - 특정 맥락을 제공하기 위해 부가적인 정보를 포함
  - 대화의 목적, 위치, 우선순위 등
- **API** 설정
- `openai.api_key` : `openai` 라이브러리가 `api_key`를 활용할 수 있도록 설정

```py
import os
import openai

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage 

openai.api_key = os.environ.get('나의_API_KEY')
# 환경 변수에서 호출

model = ChatOpenAI(model="gpt-4", api_key=openai.api_key)
# 모델에 입력
```

### 메시지 입력
- `invoke` : 데이터 인자를 가지고 기능을 호출

```py
response = model.invoke([HumanMessage(content="안녕하세요, 무엇을 도와드릴까요?")])
print(response.content)
# 
저는 AI입니다. 정보 검색, 일정 관리, 음악 재생 등 다양한 요청을 처리할 수 있습니다. 무엇이든 물어보세요!
```

[¶ Top](#index)
><br>

## 프롬프트 템플릿
> 프롬프트 객체를 생성하여 대화형 형태 형성

### 템플릿 설정
- 핵심 내용을 변수로 받도록 설정

```py
from langchain_core.prompts import ChatPromptTemplate

# 시스템 메시지 설정
system_template = "Translate the following sentence from English to {language}:"

# 사용자 텍스트 입력
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])
```

### 템플릿 생성
- `invoke` 메서드 : 실제 프롬프트 생성 및 `result`에 저장
  - 딕셔너리 값을 템플릿의 변수에 채워줌
- 생성된 템플릿을 모델에 입력하면 설계한 답변을 얻을 수 있음

```py
result = prompt_template.invoke({"language": "French", "text": "How are you?"})

print(result.to_messages())
#
[SystemMessage(content='Translate the following sentence from English to French:', additional_kwargs={}, response_metadata={}), HumanMessage(content='How are you?', additional_kwargs={}, response_metadata={})]
```

[¶ Top](#index)
><br>

## 파이프라인
- 기본적인 **langchain** 사용법
- `parser`추가 : 모델의 출력을 처리하고 문자열 형태로 반환 

```py
from langchain_core.output_parsers import StrOutputParser

# 응답을 파싱하는 파서 초기화
parser = StrOutputParser() 
# 모델의 출력을 처리하고 문자열 형태로 반환 
```

### `chain` 형성
  - `chain = prompt_template | model | parser`

### 실행 결과

```py
# 체인 실행
response = chain.invoke({"language": "Spanish", "text": "Where is the library?"})
print(response)
#
¿Dónde está la biblioteca?
```

[¶ Top](#index)
><br>

## Vector DB 활용

### 추가 모듈 설치
- `pip install langchain-community langchain-core`

### Embedding 설정
- `OpenAI` 임베딩 모델 초기화

```py
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

### `faiss` 설치
- `import faiss` 
  - 고차원 벡터 검색 라이브러리로, 벡터 간의 유사성을 빠르게 계산하고 검색
- `from langchain_community.vectorstores import FAISS` 
  - 벡터 검색 도구
- `from langchain_community.docstore.in_memory import InMemoryDocstore` 
  - 문서들을 메모리에 저장할 수 있도록 해주는 클래스

### FAISS 인덱스 생성
- `index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))`
  - **L2** 거리(유클리드 거리)를 사용하여 벡터 간의 유사도를 계산하는 인덱스
- `embed_query("hello world")`
  - 텍스트 `"hello world"`를 임베딩 모델`embeddings`에 전달하여 변환
- `len()`: 벡터의 크기를 구함

### `vector_stroe` 생성
- `docstore=InMemoryDocstore()`
  - 각 벡터에 해당하는 문서의 메타데이터나 내용을 메모리에 저장
- `vector_store` : 저장 및 로드
  - `vector_store.save_local("faiss_index")` : 저장
  - `new_vector_store = FAISS.load_local("faiss_index", embeddings)` : 로드
- 병합
  - `db1 = FAISS.from_texts(["문서 1 내용"], embeddings)` : `db1` 생성
  - `db2 = FAISS.from_texts(["문서 2 내용"], embeddings)` : `db2` 생성
  - `db1.merge_from(db2)` : 병합

```py
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    # 각 벡터에 해당하는 문서의 메타데이터나 내용을 메모리에 저장
    index_to_docstore_id={}
)
```

### `documents` 생성
- `from langchain_core.documents import Document`
- **LangChain**에서 문서의 내용을 표현하는 클래스
- `page_content`: 문서의 실제 내용
- `metadata`: 문서에 대한 추가 정보나 메타데이터를 딕셔너리 형태로 저장
- `documents` [전체 목록](#documents)

```py
documents = [
    # 트위터 게시물
    Document(page_content="LangChain은 데이터 파이프라인을 만들 때 매우 유용한 라이브러리입니다.", metadata={"source": "tweet"}),
 ...
    Document(page_content="이번 여름 방학에는 제주도로 여행을 갈 예정입니다. 바다와 자연을 만끽하고 싶어요.", metadata={"source": "travel"})
]
```

### `uuids` 생성
- `from uuid import uuid4`
- 고유 **ID** 생성 및 문서 추가
- 고유한 **UUID(Universally Unique Identifier)** 를 생성하는 함수

```py
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```

### 유사성 검색
- `.similarity_search()` : 기본 유사성 검색
  - `k = 개수` : 설정한 개수의 결과를 반환
  - 메타 데이터를 필터로 설정

```py
results = vector_store.similarity_search("내일 날씨는 어떨까요?", k=2, filter={"source": "news"})
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
#
* 오늘은 날씨가 맑고 기온은 23도입니다. 적당히 따뜻한 날씨가 이어질 것으로 보입니다. [{'source': 'news'}]
* 올해 여름은 예년보다 더 무더운 날씨가 지속될 것으로 보입니다. [{'source': 'news'}]
```

- `.similarity_search_with_score()` : 점수와 함께 유사성 검색

```py
results_with_scores = vector_store.similarity_search_with_score("LangChain에 대해 이야기해주세요.", k=2, filter={"source": "tweet"})
for res, score in results_with_scores:
    print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")
#
* [SIM=0.205] LangChain은 데이터 파이프라인을 만들 때 매우 유용한 라이브러리입니다. [{'source': 'tweet'}]
* [SIM=0.479] 이번 주말에 할 일 목록을 작성해봤어요. 너무 많지만 다 해내고 싶네요. [{'source': 'tweet'}]
```

[¶ Top](#index)
><br>

## RAG 연결

### `retriever` 설정
- `retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})`
-  주어진 쿼리에 대해 벡터 저장소에서 유사한 문서를 검색하는 역할
- `search_type`은 검색 방식에 대한 설정
  - `similarity`는 유사도 기반 검색
- `search_kwargs`는 검색에 사용할 추가적인 인자들을 설정
- `k=1`: 유사도가 가장 높은 1개의 문서를 검색하도록 설정

### 프롬프트 템플릿 정의
- `from langchain_core.prompts import ChatPromptTemplate`
- `system` : 시스템 메시지, 모델에게 주어진 컨텍스트만 사용하여 질문에 답하도록 지시
- `user` : 사용자 메시지, 질문과 함께 `context`를 입력받음. `{context}`와 `{question}`은 동적으로 채워질 자리

```py
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])
```

### `RunnablePassthrough`
- `from langchain_core.runnables import RunnablePassthrough`
- 입력을 그대로 출력하는 기능 제공, 디버깅을 위해서 출력값을 확인

### `class DebugPassThrough(RunnablePassthrough):`
- `invoke` 메서드를 재정의하여 각 단계에서 처리되는 출력값을 출력
- `*args`와 `**kwargs`는 메서드에 전달된 모든 인수를 부모 클래스의 `invoke` 메서드에 그대로 전달

```py
class DebugPassThrough(RunnablePassthrough):

    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)

        print("Debug Output:", output)
        return output
```

### `class ContextToText(RunnablePassthrough):`
- 문서 리스트를 텍스트로 변환하는 단계
- `context`의 각 문서를 문자열로 결합
- `inputs["context"]`는 문서들의 리스트
  - 여기서 각 문서의 `page_content`를 추출하여 결합하는 부분

```py
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가

        context_text = "\n".join([doc.page_content for doc in inputs["context"]])

        return {"context": context_text, "question": inputs["question"]}
```

### RAG Chain 생성
- `DebugPassThrough()`: 사용자 질문을 그대로 출력하는 디버깅 단계
  - 각 단계마다 `DebugPassThrough` 추가
- `ContextToText()`: 검색된 문서들을 텍스트로 결합하는 단계
  - `retriever`: 컨텍스트를 가져오는 단계
  - 주어진 질문에 대해 가장 관련 있는 문서를 검색
- `contextual_prompt`: 최종적으로 context와 question을 템플릿에 맞춰 포맷팅하는 단계
- `model`:모델이 최종 응답을 생성하는 단계

```py
rag_chain_debug = {
    "context": retriever,
    "question": DebugPassThrough()        
    # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
} | DebugPassThrough() | ContextToText()|   contextual_prompt | model
```

### 질문 실행 및 각 단계 출력 확인

```py
response = rag_chain_debug.invoke("앞으로의 날씨 전망")
print("Final Response:")
print(response.content)
#
Debug Output: 앞으로의 날씨 전망
Debug Output: {'context': [Document(metadata={'source': 'news'}, page_content='오늘은 날씨가 맑고 기온은 23도입니다. 적당히 따뜻한 날씨가 이어질 것으로 보입니다.')], 'question': '앞으로의 날씨 전망'}
Final Response:
앞으로의 날씨는 적당히 따뜻한 날씨가 이어질 것으로 보입니다.
```

[¶ Top](#index)
><br>

# `documents`
[돌아가기](#documents-생성)

```py
documents = [
    # 트위터 게시물
    Document(page_content="LangChain은 데이터 파이프라인을 만들 때 매우 유용한 라이브러리입니다.", metadata={"source": "tweet"}),
    Document(page_content="오늘 하루는 정말 바빴어요! 내일은 조금 여유롭기를 바라요.", metadata={"source": "tweet"}),
    Document(page_content="파이썬 3.10에서 개선된 기능들이 많아서 기대됩니다. 업데이트는 언제 할까요?", metadata={"source": "tweet"}),
    Document(page_content="이번 주말에 할 일 목록을 작성해봤어요. 너무 많지만 다 해내고 싶네요.", metadata={"source": "tweet"}),

    # 뉴스 기사
    Document(page_content="오늘은 날씨가 맑고 기온은 23도입니다. 적당히 따뜻한 날씨가 이어질 것으로 보입니다.", metadata={"source": "news"}),
    Document(page_content="최근 전 세계적으로 물가 상승률이 급격히 증가하고 있습니다. 경제 전문가들은 경기가 불안정하다고 경고하고 있습니다.", metadata={"source": "news"}),
    Document(page_content="내일 서울에는 비가 올 예정입니다. 우산을 챙기세요.", metadata={"source": "news"}),
    Document(page_content="올해 여름은 예년보다 더 무더운 날씨가 지속될 것으로 보입니다.", metadata={"source": "news"}),

    # 개인 블로그
    Document(page_content="저는 요즘 파이썬을 배우고 있습니다. 오늘은 클래스와 함수에 대해 공부했어요.", metadata={"source": "blog"}),
    Document(page_content="최근에 읽은 책 'Clean Code'는 정말 유익했어요. 코드의 품질을 높이기 위한 좋은 팁이 많았어요.", metadata={"source": "blog"}),
    Document(page_content="웹 개발에 대한 경험을 나누고 싶어요. 초보자부터 고급 개발자까지 모두가 배울 수 있는 내용이 담겨 있어요.", metadata={"source": "blog"}),
    Document(page_content="어제는 친구와 함께 소풍을 갔어요. 기분 좋은 하루였습니다.", metadata={"source": "blog"}),

    # 책 내용
    Document(page_content="프로그래밍에서 가장 중요한 것은 문제를 해결하는 능력이다. 문제를 해결하려면 어떻게 접근할지 고민해야 한다.", metadata={"source": "book"}),
    Document(page_content="알고리즘은 단순히 코드 작성법을 넘어서, 문제를 풀 수 있는 방법을 찾는 과정이다.", metadata={"source": "book"}),
    Document(page_content="객체 지향 프로그래밍에서는 객체 간의 상호작용이 중요하다. 이를 통해 프로그램의 확장성과 유지보수성이 높아진다.", metadata={"source": "book"}),
    Document(page_content="기술적 부채를 해결하는 과정은 시간과 비용이 많이 들지만, 장기적으로 보면 매우 중요한 일이다.", metadata={"source": "book"}),

    # 유머 또는 농담
    Document(page_content="어제 컴퓨터가 고장 나서 정말 힘들었어요. 그런데 오늘은 그냥 다리도 아프고 몸도 안 좋아요.", metadata={"source": "humor"}),
    Document(page_content="왜 파이썬 개발자가 바다에서 수영을 못할까요? 너무 많은 예외를 처리하느라.", metadata={"source": "humor"}),
    Document(page_content="내가 제일 좋아하는 요리는 피자. 왜냐하면 코드처럼 토핑을 올릴 수 있기 때문이에요.", metadata={"source": "humor"}),
    Document(page_content="자바 개발자는 왜 컴퓨터가 느려지는지 이해하지 못할까요? 무한 루프에 빠져서.", metadata={"source": "humor"}),

    # 과학 기사
    Document(page_content="과학자들은 최근 블랙홀에 대한 새로운 발견을 발표했습니다. 이는 우리가 우주를 이해하는 데 중요한 단서를 제공합니다.", metadata={"source": "science"}),
    Document(page_content="다양한 실험을 통해 인간의 뇌는 놀라운 능력을 가지고 있다는 사실이 밝혀졌습니다.", metadata={"source": "science"}),
    Document(page_content="기후 변화는 이미 우리에게 실질적인 영향을 미치고 있습니다. 이를 해결하기 위한 국제적인 노력들이 필요합니다.", metadata={"source": "science"}),
    Document(page_content="지구에서 가장 오래된 생명체는 3.5억 년 전의 것으로 추정되는 미세조류입니다.", metadata={"source": "science"}),

    # 제품 리뷰
    Document(page_content="이 스마트폰은 카메라 성능이 뛰어나며, 배터리 수명이 길어서 매우 만족스럽습니다.", metadata={"source": "review"}),
    Document(page_content="이 커피 머신은 매우 간편하게 사용할 수 있으며, 커피 맛도 매우 좋아요. 아침에 한 잔 마시면 하루가 시작됩니다.", metadata={"source": "review"}),
    Document(page_content="이 책은 매우 유익하고 읽을 만한 가치가 있습니다. 문체도 쉽게 읽히고, 내용도 풍부합니다.", metadata={"source": "review"}),
    Document(page_content="이 헤드폰은 노이즈 캔슬링 기능이 뛰어나며, 음질도 매우 훌륭합니다. 비싼 가격이 아깝지 않네요.", metadata={"source": "review"}),

    # 음식 레시피
    Document(page_content="오늘의 레시피는 치킨 카레입니다. 준비할 재료는 치킨, 카레 가루, 야채 등이 필요합니다.", metadata={"source": "recipe"}),
    Document(page_content="빠르고 간단한 스파게티 레시피를 소개합니다. 재료는 스파게티 면, 토마토 소스, 치즈가 필요합니다.", metadata={"source": "recipe"}),
    Document(page_content="바나나 브레드를 만들 때는 바나나가 잘 익었을 때 사용하면 더 맛있습니다.", metadata={"source": "recipe"}),
    Document(page_content="초콜릿 케이크는 부드럽고 달콤한 맛이 특징입니다. 주재료로는 밀가루, 초콜릿, 설탕을 사용합니다.", metadata={"source": "recipe"}),

    # 여행 정보
    Document(page_content="이번 여름에 일본으로 여행을 가기로 했습니다. 도쿄에서의 멋진 여행을 기대하고 있습니다.", metadata={"source": "travel"}),
    Document(page_content="내년에 유럽을 여행하려고 계획 중입니다. 프랑스, 독일, 이탈리아를 방문할 예정입니다.", metadata={"source": "travel"}),
    Document(page_content="뉴욕에서의 여행은 매우 흥미로웠습니다. 센트럴 파크에서의 산책이 기억에 남아요.", metadata={"source": "travel"}),
    Document(page_content="이번 여름 방학에는 제주도로 여행을 갈 예정입니다. 바다와 자연을 만끽하고 싶어요.", metadata={"source": "travel"})
]
```