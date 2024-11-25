---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 7
subtitle: TIL Day 64
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

# Chatbot 고도화
> >친구가 이야기 해주듯이 스토리를 요약해서 전달해주는 **Chatbot**

## 연관성 검사 단계 추가
- 질문에 맞는 문서를 검색
- 질문과 어느정도의 연관성을 갖는지 검사
  - 1-100 사이로 수치와
- 연관이 적거나 없는 경우
  - 더 자세한 질문 요구
  - 자료 요구

```py
prompt = f"""
    persona : you are script search system
    find script about query
    <query>
    {querry}
    </query>

    calculate relavence score
    between query and script
    socre = [1-100]
    fail = 0

    <script>
    {script}
    </script>

    return only score
    """
```

```py
while True:
    print("================================")
    querry = input("어떤 이야기가 듣고 싶으신가요?")
    if querry.lower() == "exit":
        break
    relavence = script_finder(querry, sdb)
    if relavence[0] < 80:
        print("모르는 이야기 입니다.")
        user_input = input("이야기를 생성하려면 텍스트 또는 URL을 입력하세요: ")
        new_script = script_maker(user_input)
        script_documents = [
                Document(page_content=new_script, metadata={"source": "script"}),
            ]
        sdb.add_documents(script_documents)
        print("이야기가 생성되었습니다.")
        continue
    elif relavence[0] >= 80 and relavence[0] < 90:
        print("더 자세히 이야기 해주세요")
        continue
    elif relavence[0] >= 95:
        script = relavence[1]
        print(script)
        break
```

### 확인된 문제
- 불규칙적으로 `Nonetype error` 발생

## 사용자 입력 기반 대화
- 사용자 입력으로 질문
- 대화 내용을 메모라에 저장하여 맥락을 유지한 채로 대화 진행

```py
def history_chain(chain, memory_store : dict):

    def get_session_history(session_ids):
        print(f"[대화 세션ID]: {session_ids}")
        if session_ids not in memory_store:  # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            memory_store[session_ids] = ChatMessageHistory()
        return memory_store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history
```

```py
while True:
      print("========================")
      query = input("질문을 입력하세요 : ")
      if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
      response = h_chain.invoke(
    # 질문 입력
    {"question": query},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "test6"}},
)
      print(query)
      print("\n답변:")
      print(response)  
```

### 확인된 문제
- 대화가 진행되지 않고 같은 답변은 반복하는 경우 발생

## 프롬프트 고도화
- 스크립트 생성을 위한 프롬프트 고도화
- 완전 자율 구성에서 통제된 구성으로 변경

```
persona = 대본 작가
    language = 한국어로만 답합니다.

    <rule>
    input을 바탕으로 대본 작성,
    모든 항목을 작성
    시간, 인물, 지역 등을 자세하게 기록,
    사실에 근거해 작성
    </rule>

    <sample>
    # 도입 1 : 주제와 관련된 영화나 노래, 사건
    - 내용 :
    # 도입 2 : 주제에 관한 설명
    - 내용 :
    # 도입 3 : 주요 인물에 관한 설명
    - 내용 :
    # 발단 1 : 사건의 배경, 시대 상황
    - 내용 :
    # 발단 2 : 주요인물의 과거
    - 내용 :
    # 발단 3 : 사건의 발단, 초기 상황
    - 내용 :
    # 전개 1 : 사건 발생 초기 상황, 과정, 주요 기술, 방법, 전술 등
    - 내용 :
    # 전개 2 : 사건의 영향, 파급 효과
    - 내용 :
    # 전개 3 : 사람들의 반응, 대응
    - 내용 :
    # 절정 1 : 사건의 발전, 커지는 영향
    - 내용 :
    # 절정 2 : 사건으로 인한 갈등, 피해 또는 극적인 상황
    - 내용 :
    # 절정 3 : 충돌, 사건의 해결 과정
    - 내용 :
    # 결말 1 : 사건의 해결, 교훈
    - 내용 :
    # 결말 2 : 사건이 남긴 것
    - 내용 :
    </sample>

    <input>
    {summaries}
    </input>

    <start>
```

### 확인된 문제
- 생성된 스크립트에 맞춰 대화를 유도하는 것이 잘 이루어지지 않음
- 전체 내용을 요약하여 전달하거나, 같은 내용을 반복해서 설명

## 추가 과제
- 대화 프롬프트 고도화
- 함수 정리 및 모듈화