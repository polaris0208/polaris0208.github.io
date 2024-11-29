---
layout: post
title: Streamlit drill
subtitle: TIL Day 68
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Web]
author: polaris0208
---

# Streamlit drill
> 제작한 **LLM_RAG_Chatbot**을 스트림릿 환경에 맞게 리팩토링하면서 활용법 연습

## DB 및 데이터 세팅
- 지속적으로 업데이트 되어야 하기 때문에 `session.state`에 저장하지 않음
- **App**이 새로고침 될 때마다 로컬 **DB**에서 정보를 받아 데이터를 업데이트

```py
json_files = [
    "./llm_chatbot/documents/filtered_unsolved_cases.json",
    "./llm_chatbot/documents/korea_crime.json",
]
titles = title_json_data(json_files)
sample_titles = titles[0:11]
path = "./llm_chatbot/db/script_db"
db_name = "script_db"
script_db = load_vstore(db_name, path)
```

## 세션 세팅
- 최초 상태를 설정 : 리스트, 딕셔너리 또는 기본값
- 앱이 동작하는 동안 유지될 정보들 보관
  - 세션 **ID** : 현재 및 저장된 **ID**
  - 제목 리스트 토글 상태 : 사이드 바에 위치하기 떄문에, 열리고 닫힌 상태가 유지되어야 함
  - 대화 맥락 : 세션 **ID**를 **KEY**로 대화 내용, 사용 스크립트 저장

```py
if "session_list" not in st.session_state:
    st.session_state["session_list"] = []
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = "no session id"
if "title_list_expanded" not in st.session_state:
    st.session_state["title_list_expanded"] = False
if "conversation" not in st.session_state:
    st.session_state["conversation"] = {}
```

## 사이드 바
- 제목 리스트 : **DB**에 존재하는 스크립트 리스트
  - 내용이 많기 떄문에 토글로 열고 닫아 관리
- 세션 리스트 : 생성된 세션 리스트
  - 최상단에는 현재 세션 표시
  - 세션 **ID**가 등록되면 자동으로 추가
  - 클릭하면 세션 **ID** 의 대화 페이지로 이동

```py
if main_option == "제목 리스트":

    toggle_button = st.sidebar.button("제목 리스트 토글")
    if toggle_button:
        st.session_state["title_list_expanded"] = not st.session_state[
            "title_list_expanded"
        ]

    if st.session_state["title_list_expanded"]:
        st.sidebar.subheader("제목 리스트")
        for title in sample_titles:
            st.sidebar.write(title)
    else:
        st.sidebar.write("제목 리스트를 펼치려면 버튼을 클릭하세요.")

elif main_option == "세션 리스트":
    st.sidebar.subheader(f"**현재 세션 ID:** {st.session_state['current_session_id']}")
    st.sidebar.subheader("세션 리스트")
    session_list = st.session_state.get("session_list")
    for session in session_list:
        if st.sidebar.button(f"[{session['id']}]"):
            st.session_state.session_id = session["id"]
            st.session_state["current_session_id"] = session["id"]
            st.session_state.page = "session_page"
            st.rerun()
```

## 세팅 페이지
- 함수로 페이지 구분
- 세션 **ID** 및 언어를 설정
  - 대화 종료시 까지 유지
  - 저장하면 사이드 바에 반영
  - 저장하지 않으면 다음 단께로 넘어가지 않도록 조치

```py
def setting_page():
    st.title("유저 설정 화면")

    # 세션 ID와 언어 설정
    session_id = st.text_input("세션 ID를 입력하세요", placeholder="예: session123")
    language = st.selectbox(
        "사용할 언어를 선택하세요", options=["한국어", "English", "日本語"]
    )

    if st.button("저장 후 넘어가기"):
        if session_id:
            # 세션 정보 저장
            st.session_state.session_id = session_id
            st.session_state["current_session_id"] = session_id
            st.session_state["session_list"].append({"id": session_id})
            st.session_state.LANG = language
            st.session_state.current_session = session_id
            st.session_state.page = "check"  
            st.rerun()
        else:
            st.warning("세션 ID를 입력해주세요!")

```

## 체크 페이지
- 기존 `While` 반복문으로 작성된 코드를 각 페이지로 분기시킴
- 답변 검증 및 분기 기능
- 사용자 입력과 검색된 스크립트의 연관성을 분석하여 페이지 분기
  - 종료 : 세팅 페이지로 이동
  - 돌아가기 : 추가적인 설명 요구 후 초기화
  - 생성하기 : 생성 페이지로 이동

```py
MIN_SCORE = 80
MAX_SCORE = 85
NEXT_SCORE = 95

# 초기 점수 설정
if "score" not in st.session_state:
    st.session_state.score = 0

def check_page():
    st.title("체크 페이지")
    query = st.text_input("어떤 이야기가 듣고 싶으신가요?", placeholder="예 : 아무거나")

    if query:
        with st.status("답변을 확인 중...", expanded=True) as status:
            if query is None or "아무거나" in query.strip():
                st.write("재미난 이야기를 가져오는 중...")
                choice = random.choice(sample_titles)
                query = choice
            relavence = evaluator(query, script_db)
            st.write(f"관련도 점수: {relavence[0]}")
            st.session_state.score += relavence[0]
        status.update(label="확인이 끝났습니다!", state="complete", expanded=False)

        if relavence[0] < 80:
            query = st.selectbox("모르는 이야기입니다.", options= ['종료, 돌아가기, 생성하기', "exit", "retry", "create"])
            if query.lower() == "exit":
                st.session_state.page = "settings"
                st.rerun()
            elif query.lower() == "retry":
                query = st.text_input("더 자세히 설명해 주세요.", placeholder="예 : 강다니엘 이모 사건")
                if query:
                    st.rerun()
            elif query.lower() == "create":
                st.session_state.page = "create"
                st.rerun()

        elif relavence[0] < 95 and relavence[0] >= 80:
            st.write("더 자세히 이야기 해주세요.")
            st.rerun()

        elif relavence[0] >= 95:
            script = relavence[1]
            st.session_state.page = "chat"
            st.session_state.query = query
            st.session_state.script = script
            time.sleep(1)
            st.rerun()
```

## 생성 페이지
- 사용자가 제공한 데이터로 새로운 스크립트 형성
- **DB**에 반영 여부 묻기
- 스크립트가 생성되는 동안 안내 메시지 출력

```py
def create_page():
    st.title("생성 페이지")
    st.write("새로운 스크립트를 입력할 수 있습니다.")

    if st.button("돌아가기"):
        st.session_state.page = "check"
        time.sleep(1)
        st.rerun()

    text_input = st.text_area("URL 또는 텍스트를 입력해주세요.")
    if text_input:
        with st.status("스크립트를 생성중입니다...", expanded=True) as status:
            st.write("작가가 문서를 읽어보는 중...")
            time.sleep(2)
            st.write("스크립트를 작성하는 중...")
            time.sleep(2)
            st.write("창작의 고통을 느끼는 중...")
            new_script = script_maker(text_input)
            st.write(f"생성된 스크립트: {new_script}")
            time.sleep(2)
            status.update(
                label="작업이 종료되었습니다.", state="complete", expanded=False
            )
        user_input = st.selectbox("DB에 저장하시겠습니까?", options= ['아니오', '예'])
        if user_input == '예':
            script_db = load_vstore("script_db", "./llm_chatbot/db/script_db")
            add_to_vstore(new_script, script_db)
```

## 채팅 페이지
- 기존 메모리 :`langchain` 의 `ChatMessageHistory`
- 변경 메모리 : `st.session_state`를 두개로 나눠 저장
  - `messages` : 대화가 진행되는 동안만 맥락을 저장
  - `conversation` : 대화가 끝난 뒤 세션 **ID** 를 키로 대화내용, 사용 스크립트 저장

```py
def chat_page(script):
    st.title("채팅 페이지")
    st.write("이제 채팅을 시작할 수 있습니다.")
    ID = st.session_state.get("session_id")
    LANG = st.session_state.get("LANG")
    QUERY = st.session_state.get("query")
    if "messages" not in st.session_state:
        init_history = [{"role": "assistant", "content": "no history yet"}]
        chain = streamlit_chain(script, init_history, LANG)
        init_response = chain.invoke(
            {"question": QUERY},
            config={"configurable": {"session_id": ID}},
        )
        st.session_state["messages"] = [{"role": "assistant", "content": init_response}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if prompt.lower() == "exit":
            st.write("대화를 종료합니다.")
            st.session_state["conversation"][f"{ID}_history"] = st.session_state["messages"]
            st.session_state["conversation"][f"{ID}_script"] = script
            del st.session_state["messages"]
            st.session_state.page = "settings"
            st.rerun()

        # elif LANG_CODE:
        #     prompt = translator(prompt, LANG_CODE)

        st.session_state.messages.append({"role": "user", "content": prompt})
        prompt = stream_data(prompt)
        st.chat_message("user").write_stream(prompt)

        history = st.session_state["messages"]
        chain = streamlit_chain(script, history, LANG)
        msg = chain.invoke(
            {"question": prompt},
            config={"configurable": {"session_id": ID}},
        )
        st.session_state.messages.append({"role": "assistant", "content": msg})
        msg = stream_data(msg)
        st.chat_message("assistant").write_stream(msg)

    if st.button("대화 종료"):
        st.session_state.page = "messages"
        st.session_state["conversation"][f"{ID}_history"] = st.session_state["messages"]
        st.session_state["conversation"][f"{ID}_script"] = script
        del st.session_state["messages"]
        st.session_state.page = "settings"
        time.sleep(1)
        st.rerun()
```

## 세션 페이지
- 세션 **ID** 를 키로 대화 내용과 스크립트를 호출
- 대화를 이어가거나, 이전 대화 내용을 확인 가능
- 저장된 내용이 없는 경우 체크 페이지로 하도록 바이패스 설정

```py
def session_page():
    ID = st.session_state["session_id"]
    LANG = st.session_state.get("LANG")
    st.title(f"{ID}페이지")
    st.write("다시 채팅을 시작할 수 있습니다.")

    if f"{ID}_history" not in st.session_state["conversation"]:
        st.session_state.page = "check"
        st.rerun()

    st.session_state["messages"] = st.session_state["conversation"][f"{ID}_history"]
    script = st.session_state["conversation"][f"{ID}_script"]

    chain = streamlit_chain(script, st.session_state["messages"], LANG)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if prompt.lower() == "exit":
            st.write("대화를 종료합니다.")
            st.session_state.page = "settings"
            st.rerun()

        # elif LANG_CODE:
        #     prompt = translator(prompt, LANG_CODE)

        st.session_state.messages.append({"role": "user", "content": prompt})
        prompt = stream_data(prompt)
        st.chat_message("user").write_stream(prompt)

        msg = chain.invoke(
            {"question": prompt},
            config={"configurable": {"session_id": ID}},
        )
        st.session_state.messages.append({"role": "assistant", "content": msg})
        msg = stream_data(msg)
        st.chat_message("assistant").write_stream(msg)

    if st.button("대화 종료"):
        st.session_state.page = "messages"
        st.session_state["conversation"][f"{ID}_history"] = st.session_state["messages"]
        st.session_state["conversation"][f"{ID}_script"] = script
        del st.session_state["messages"]
        st.session_state.page = "settings"
        time.sleep(1)
        st.rerun()
```

## 페이지 설정
- 페이지 세션을 변경하여 페이지를 호출하도록 설정
- 기본 페이지는 세팅 페이지로 이동하도록 설정

```py
if "page" not in st.session_state:
    st.session_state.page = "settings"

if st.session_state.page == "settings":
    setting_page()
elif st.session_state.page == "check":
    check_page()
elif st.session_state.page == "create":
    create_page()
elif st.session_state.page == "chat":
    chat_page(st.session_state.script)
elif st.session_state.page == "session_page":
    session_page()
```