---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 6
subtitle: TIL Day 61
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---
# Chatbot 조정
> >친구가 이야기 해주듯이 스토리를 요약해서 전달해주는 **Chatbot**

## 구상

### 기능
- 특정 사건, 이야기를 친구가 이야기 해주듯 친근하게 전달

### 방식
- 주제에 관해 흥미를 유발하는 질문으로 시작
- 비유나 상황극을 통해 이야기 전달
- 사용자의 질문이나 호응을 유도

### 고려 사항
- 사건, 이야기 데이터 가공 및 활용
- 상호작용 하는 대화 방식의 구현
- **LLM** 의 답변 통제 : 주제에 벗어난 이야기 등

## 모델 개요
- 모델이 한번에 처리할 수 있는 토큰 수 고려
- 모델이 이해할 수 있는 형태의 데이터 구조 고려
- 단계별로 가공하여 **LLM** 에게 전달

```
[데이터셋] -> [모델 1] -> [모델 2] -> [LLM] -[사용자]
[단순자료] -> [요약집] -> [스크립트] -> [스토리 텔링]
```

## 모델 초안 작성

### 패키지 및 API 설정
- `import warnings` : `langchain` 특정 패키지 지원 종료 예고 메시지 무시 설정
- `numpy` : 2.0.0 이상 사용 시 `langchain` 과 충돌

```py
import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import warnings

openai.api_key = os.environ.get('NBCAMP_01')
# LangChainDeprecationWarning 경고를 무시
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
```

### 문서 분할
- 요약이 목적이기 떄문에 청크사이즈는 전체 문서의 1/10 분량 설정

```py
# 문서 로딩 및 전처리 (모델 1)
def load_and_process_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # 문서 분할:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits
```

### 문서 요약
- 분리된 문서들을 청크단위로 불러와 요약 후 하나의 문서로 통합
  - 원본 자료를 모델에 입력 가능한 사이즈로 분할
  - 요약 후 조합하여 전체 내용을 담은 모델에 입력 가능한 사이즈의 문서 생성

```py
def summarize_documents(docs):
    summaries = []  

    for doc in docs:
        doc_text = doc.page_content
        
        # 요약을 요청하는 프롬프트 생성
        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 모델 선택
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},  # 시스템 메시지
                    {"role": "user", "content": f"이 문서를 요약해주세요: {doc_text}"}  # 요약 요청
                ],
                max_tokens=200,  # 요약된 내용의 최대 토큰 길이
                temperature=0.5  # 생성의 창의성 조정 (0은 정해진 답변, 1은 더 창의적)
            )

            # 요약된 텍스트 추출
            summary = response.choices[0].message.content  # 응답에서 요약된 텍스트 가져오기
            summaries.append(summary)  # 요약된 문서 리스트에 추가

        except Exception as e:
            # 오류 처리: 만약 API 호출 중에 문제가 발생하면 오류 메시지 추가
            print(f"Error summarizing document: {e}")
            summaries.append(f"Error summarizing document: {e}")
    
    return ''.join(summaries)
```

### 스크립트 생성
- 요약된 문서를 바탕으로 하나의 스토리 생성
- 원본 문서가 사실의 나열로 이루어진 위키 형식의 문서일 경우 필요한 과정
- 사실을 조합하여 스토리 라인으로 정리

```py
def generate_script_from_summary(summarized_text):
    # 대본 작성을 위한 템플릿 생성
    script_prompt = f"""
    persona = 대본 작가
    language = 한국어로만 답합니다.
    
    <rule>
    개조식으로 작성
    회차는 10회
    회차는 시간 순서대로 진행
    </rule>

    <sample>
    대본 제목 : 
    회차 :
    제목 :
    배경 :
    사건 :
    인물 :
    중요 장면 :
    </sample>

    <output>
    1회부터 10회까지의 대본
    </output>
    """

    # OpenAI를 통해 대본 생성
    client = OpenAI(api_key=openai.api_key)
    script = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{script_prompt}"},  # 시스템 메시지
            {"role": "user", "content": f"{summarized_text}"}],
        max_tokens=3000
    )
    return script.choices[0].message.content
```

#### 결과

```
회차 2:
    제목 : 위조지폐 검거
    배경 : 국가의 화폐 교환 혼란을 초래한 위조지폐 사건
    사건 : 국가정보원의 노력에도 수사가 어려운 사건
    인물 : 국가정보원 요원들, 검거된 범인
    중요 장면 : 국가정보원과 경찰의 합동 수사로 범인을 검거하는 장면
```

### 대화 구성

#### 조건
- 이야기를 끊어서 전달
- 내용 확인이나 흥미를 유발하는 질문
- 대화의 맥락 유지

#### 일반 대화 구성
- 프롬프트-모델-파서 체인 구성

```py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"당신은 스토리 텔러, 대본에 맞게 이야기를 해주세요, 한 회차씩 이야기 해주세요, 한 회차가 끝나면 흥미를 유발하는 질문을 하세요. 대본 : {script}",
        ),
        # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
    ]
)

# LLM 모델을 생성합니다.
llm = ChatOpenAI(model = 'gpt-3.5-turbo', api_key=openai.api_key, temperature=0)

# 일반 Chain 생성
chain = prompt | llm | StrOutputParser()
```

#### 대화 맥락 유지
- `ChatMessageHistory()` 객체를  `session_ids` 기준으로 저장 및 불러오기

```py
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

# 세션 ID를 기반으로 기록 조회
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)
```

#### 대화 싫행
- 사용자 입력과 함께 `session_ids` 입력

```py 
chain_with_history.invoke(
    # 질문 입력
    {"question": "위조지폐 사건에 대해서 설명해줘."},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "abc123"}},
)
```

```
[대화 세션ID]: abc123
한 개인이 2005년부터 2013년까지 8년간 5천 원권 위조지폐를 대량으로 유통하여 국가의 화폐 교환 혼란을 초래한 사건이 있었습니다. 이 사건은 대부김이라는 인물이 위조지폐를 만들어 대규모 유통을 시작한 것이 시작이었습니다. 국가정보원과 경찰이 합동으로 수사를 진행하여 범인을 검거하는 등의 노력을 기울였습니다. 이 사건을 통해 국가는 오천 원권 신권을 도입하고 추가적인 수사를 진행하며 범인을 최종적으로 체포하는 등의 과정을 거쳤습니다. 이 사건은 사회에 미친 영향과 방지 방법에 대한 교훈을 남기며 사건의 결론을 이끌었습니다. 

#Question:
이러한 위조지폐 사건이 어떻게 국가와 국민에 영향을 미쳤을까요?
```

### 추가 과제

#### 챗봇 구성 
- 사용자 입력 기반으로 대화가 진행되도록 구성

#### RAG 연결 
- 상세한 내용 답변을 위해 RAG 연결

#### 프롬프트 엔지니어링
- 말투, 이야기 구성 수정
- 주제에서 벗어나는 경우 통제

#### 데이터 엔지니어링
- 고성능 모델은 대화에만 사용
- 데이터 생성을 위한 무료 모델 필요