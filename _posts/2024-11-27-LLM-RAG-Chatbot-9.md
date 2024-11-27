---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 9
subtitle: TIL Day 66
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---
# Chatbot 고도화 - 조건화
> 사용자 입력에 따라 다른 결과 반환

## 답변 검증 LLM
- 사용자 입력과 **RAG** 검색결과를 비교해 연관성 검사
  - 검색결과에서 주제를 추론
  - 사용자 입력과 대조
  - 정수형 점수 반환

### 검증 LLM 코드

```py
def evaluator(query, db):
    """
    db에서 찾아온 스크립트가 적절한지 판단하는 함수

    Parameters:
        query : 사용자 입력
        db : 스크립트가 저장된 db
    Returns:
        연관 정도 점수
        스크립트 : 연관 정도가 적절한 경우
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai.api_key,
        max_tokens=100,
        temperature=0.0,
    )
    script_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    script = script_retriever.invoke(query)[0].page_content
    prompt = ChatPromptTemplate.from_template(
    """
    persona : relavence check machine
    **return only integer score**
    1. extract subject of script
    2. check relavence between query and subject
    3. calculate elaborate score 
    4. maximum '100', minimum '0', 
    5. increas by '5'
    6. sample is about conversation
    <sample>
    script : 'title : 강다니엘 이모 사건, content : 나 아는사람 강다니엘 닮은 이모가 다시보게되는게 다시 그때처럼 안닮게 엄마보면 느껴지는걸수도 있는거임?'

    query : '사건'
    ai : '10'

    query : '이모'
    ai : '25'

    query : '이모 사건'
    ai : '80'

    query : '강다니엘 사건'
    ai : '85'

    query : '강다니엘 이모'
    ai : '95'
    </sample>

    <query>
    {query}
    </query>

    <script>
    {script}
    </script>
    """
    )
    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"query": query, 'script' : script})
    if not score : return [0, 'N/A']
    return [int(score), script]
```

### 검증결과를 대화 LLM에 연결
- 주어진 점수에 따라 다른 답변

```py
relavence = evaluator(query, script_db)
    print(relavence[0])
    if relavence[0] < 80: 
         print('모르는 이야기 입니다.', '종료 : exit', '다시 물어보기 : return', '생성하기 : create')
         user_input = input('입력하세요.')
         if user_input.lower() == "exit":
            print("대화를 종료합니다.")
            query = False
            break
         elif user_input.lower() == "return":
            continue
         
         elif user_input.lower() == "create":
            text_input = input('URL 또는 텍스트를 입력해주세요.')
            new_script = script_maker(text_input)
            script_documents = [
                Document(page_content=new_script),
             ]
            script_db.add_documents(script_documents)
            script_db.persist()
            print('생성이 완료되었습니다.', '다시 답변해주세요.')
            continue
         
    elif relavence[0] < 95 and relavence[0] >= 80:
        print("더 자세히 이야기 해주세요")
        continue
    elif relavence[0] >= 95:
        script = relavence[1]
        break
```

### 확인된 문제
- `score : 100` 처럼 문자열로 반환
- 점수를 0 또는 100으로 단순하게 부여하는 문제
- 단순 비교로 `사건`만 입력해도 100을 반환

### 해결
- 프롬프트 조정
  - 오직 정수만 반환하도록 유도
  - 점수를 5점 단위로 분할하도록 지시
  - 채점 기준 예시 제시

## 사용자 정의 스크립트 제작 기능
- 찾는 이야기가 **DB**에 없는 경우
- 사용자가 입력한 정보를 바탕으로 새로운 스크립트 생성
  - 입력 형태 : **URL**, 텍스트
- 스크립트 생성 후 **DB**에 반영

### 코드

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def script_maker(INPUT : str):
  print("다소 시간이 소요될 수 있습니다.")
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
        )
  if INPUT.startswith("http"):
        url = INPUT
        web_docs = WebBaseLoader(url).load()
        if web_docs[0].metadata['title'] : title = web_docs[0].metadata['title']
        else : title = ''
        docs = f"title : {title} \n\n" + web_docs[0].page_content
  else:
        docs= str(INPUT)
  documents = [Document(page_content=docs)]
  SPLITS = text_splitter.split_documents(documents)
  refined = documents_filter(SPLITS)
  return generate_script(refined)
```

## 대화 LLM 오류 수정

### 기존 코드
- 사용자 입력이 주어질 때마다 스크립트 검색
- 대부분의 경우 메모리로 맥락이 유지되어 하나의 스크립트로 대화
- 일정 확률로 맥락이 끊기고 다른 스크립트 진행

```py
    chain = (
        {
            "script" : itemgetter("question") | script_retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
```

### 개선한 코드
- `RunnableMap` : `chain`의 인자를 더 자유롭게 설정 가능
- 검증 **LLM**으로 검증된 스크립트를 고정하여 대화 진행

```py
    chain = RunnableMap(
        {
            "script": lambda inputs: script,  # script는 고정값으로 전달
            "question": itemgetter("question"),  # 입력에서 question 추출
            "chat_history": itemgetter("chat_history"),  # 입력에서 chat_history 추출
        }
    ) | prompt | llm | StrOutputParser()
```

## 이야기 무작위 선택 기능
- **DB**에 저장된 데이터셋의 제목만 리스트화
- 무작위로 선택된 제목으로 검색 : 검증 **LLM** 판독 결과 : 100

### 코드

```py
import random
while True:
    print("================================")
    path = './db/script_db'
    db_name = 'script_db'
    script_db = load_vector_store(db_name, path)
    query = input("어떤 이야기가 듣고 싶으신가요?")
    print(query)
    if query.lower() == "exit":
        print("대화를 종료합니다.")
        query = False
        break
    elif query is None or  "아무거나" in query.strip():
        print("재미난 이야기를 가져오는 중...")
        choice = random.choice(sample_titles)
        query = choice
        print(choice)
        break
```

## 추가할 기능
- 다국어 기능

## 추가 과제 
- 코드 정리, 모듈화