---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 8
subtitle: TIL Day 65
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

# Chatbot 고도화 - 데이터셋 황용
> >친구가 이야기 해주듯이 스토리를 요약해서 전달해주는 **Chatbot**

## 테스트 사용 모델 변경
- `gpt-4o-mini`
- 입력 토큰 128k
- 출력 토큰 16k

## Json 형태 데이터셋을 불러와 활용
- 데이터셋 종류 : 위키 형식 문서
  -  한가지 문서에 다양한 정보를 담고 있음
    - 특정 주제에 관련된 정보
    - 유사한 주제
    - 정보 출처

```py
import json

def process_json_data(json_files):
    """
    여러 JSON 파일을 읽고 데이터를 통합한 후 특정 형식의 문자열 리스트로 반환

    Parameters:
        json_files (list): JSON 파일 경로 리스트

    Returns:
        list: 파일 데이터에서 'title'과 'content'를 읽어 특정 형식으로 변환한 리스트
    """
    all_json_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_json_data.extend(data)
        except FileNotFoundError:
            print(f"Error: 파일을 찾을 수 없습니다 - {file_path}")
        except json.JSONDecodeError:
            print(f"Error: JSON 파일 형식이 잘못되었습니다 - {file_path}")

    # 'title'과 'content'를 읽어 특정 형식으로 반환
    return [
        f"Title: {item.get('title', 'N/A')}\nContent: {item.get('content', 'N/A')}"
        for item in all_json_data
    ]
```

### 확인된 문제점
- 데이터 사이사이에 헤더 또는 버튼, 플레이스 홀더 등의 텍스트가 포함
- 하이퍼 링크, 문서 출처 등 불필요한 정보 다수 포함

```
 '제기했으나 유야무야 묻혀져 아직까지의문사로 남아있다.\n'
 '경과[편집]\n'
 '경과\n'
 '1970년3월 17일밤 11시경
 ...
 3살 된 아들이 1명 있는 것으로 확인되었다.[3][4]아이의 아버지에 대해서는 당시 정부의 한
 ...
```

### 해결 방안
- 필터링 코드 작성
- 주제에 맞는 내용만 남기고 제거
  - 문서를 분할 후 순차적으로 필터링
- `ConversationSummaryMemory` : 전체 내용을 요약하여 맥락정보를 제공
  - 사용하지 않는 경우 : 분할 된 문서가 각각의 기준에 맞춰 필터링됨
  - 사용하는 경우 : 맥락을 공유하며 이전 필터링된 문서와 이어지게 필터링
- 입력 토큰의 여유가 있는 고성능 모델을 사용하면, 한번의 입력으로 필터링 가능
- 필터링 결과 : 20000자에서 14000자로 축소, 불필요한 기호, 내용 제거
- 프롬프트 수정으로 불건전 하거나 민감한 정보 필터링에도 활용 가능

```py
from langchain.memory import ConversationSummaryMemory

def documents_filter(SPLITS):
    """
    분할된 데이터에서 불필요한 데이터를 제거하고 하나로 결합
    ConversationSummaryMemory에 이전 내용을 요약하여 저장
    아전 내용과 대조해서 불필요한 데이터 구분

    Parameters:
        SPLITS: 분할된 텍스트 데이터 : Document

    Returns:
        텍스트 데이터
    """
    llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai.api_key,
                max_tokens=1000,
                temperature=0.0,
            )
    summaries = []
    memory = ConversationSummaryMemory(
        llm=llm, return_messages=True)
    
    count = 0
    for SPLIT in SPLITS:
        SPLIT = SPLIT.page_content

        try:
            context = memory.load_memory_variables({})["history"]
            prompt = ChatPromptTemplate.from_template(
                """
                persona : documents filter
                language : only in korean
                extract the parts related to the context and ignore the rest,
                write blanck if it's not relevant,
                
                <context>
                {context}
                </context>
                
                <docs>
                {SPLIT}
                </docs>
                """
            )
            chain = prompt | llm | StrOutputParser()
            summary = chain.invoke({"SPLIT": SPLIT, 'context' : context})
            memory.save_context({"input": f'summary # {count}'}, {"output": summary})
            summaries.append(summary)
            count+=1

        except Exception as e:
            # 오류 처리: 만약 API 호출 중에 문제가 발생하면 오류 메시지 추가
            print(f"Error summarizing document: {e}")
            summaries.append(f"Error summarizing document: {e}")

    return "".join(summaries)
```

## 스트립트 생성 프롬프트 고도화
- 소설의 구성 방법 참고해 적용
- 불필요한 내용은 피해서 작성하도록 유도
- 과도하게 요약하여 내용이 빈약한 경우 방지 : 최소 3000토큰 지정
- 할루시네이션 방지

```
   persona = script writer
    language = only in korean
    least 3000 tokens
    use input,
    refer to sample,
    write about time, character, event,
    write only fact
    ignore the mere listing of facts and write N/A
 
    <sample>
    # title : title of script
    # prologue 1 : song, movie, book, show about subject
    - coontent :
    # prologue 2 : explain about subject
    - coontent :
    # prologue 3 : explain about character
    - coontent :
    # exposition 1 : historical background of subject
    - coontent :
    # exposition 2 : history of character
    - coontent :
    # exposition 3 : beginning of event
    - coontent :
    # development 1 : situation, action, static of character
    - coontent :
    # development 2 : influence of event
    - coontent :
    # development 3 : reaction of people
    - coontent :
    # climax 1 : event and effect bigger
    - coontent :
    # climax 2 : dramatic action, conflict
    - coontent :
    # climax 3 : falling Action
    - coontent :
    # denouement : resolution
    - coontent :
    # epilogue : message, remaining
    - coontent :
    </sample>

    <input>
    {summaries}
    </input>
```

## Chatbot 프롬프트 고도화
- 혼자서 진행하지않고 사용자의 답변에 맞춰 진행하도록 유도
- 스크립트에 적힌 헤더 등 기호가 노출되던 문제 수정
- 친근한 말투로 수정

```
persona : story teller
    language : only korean
    tell dramatic story like talking to friend,
    speak informally,
    progress chapter by chapter,
    **hide header like '###'**,
    start chapter with interesting question,
    wait user answer
    give reaction to answer,
    do not use same reaction
    
    # script
    {script}

    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 
```

### 추가할 기능
- 답변 검증 기능 개선 후 추가
- 무작위 답변 기능
- 찾는 정보가 없을 경우 자료를 받아 생성하는 기능
- 다국어 지원
