---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 10
subtitle: TIL Day 67
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

# Chatbot 고도화 - 모듈화, 부가기능 추가
> 각 기능을 패키지로 만들어 모듈화하고 부가기능을 추가

## 모듈화
- 함수들을 개별 파일로 분리하여 모듈화
- 실행 공간에서 코드를 최소화

```
llm_module
│
├── db_utils
├── docs_utils
├── llm_utils
├── script_utils
└── translator_module
```

### `db_utils`
- **Chroma DB** : 생성, 저장, 불러오기 기능 모듈
- 로컬환경과 연결하여 유기적으로 데이터 추가 및 불러오기 가능

```
def create_vstore(DOCS, DB_NAME: str, DB_PATH):
    return Chroma.from_documents(
        documents=DOCS,
        collection_name=DB_NAME,
        persist_directory=DB_PATH,
        embedding=EMBED,
    )


def load_vstore(DB_NAME: str, DB_PATH):
    return Chroma(
        collection_name=DB_NAME,
        persist_directory=DB_PATH,
        embedding_function=EMBED,
    )


def add_to_vstore(SCRIPT, DB):
    script_documents = [
        Document(page_content=SCRIPT),
    ]
    DB.add_documents(script_documents)
    DB.persist()
```

### `script_utils` 
- 스크립트 생성 기능
- 사용자 정의 스크립트 개선
  - 스크립트 생성에 다소 시간 소요
  - 생성하는 동안 랜덤으로 메시지 출력

```py
def script_maker(INPUT: str):
    print("다소 시간이 소요될 수 있습니다.")
    messages = [
        "처리 중...",
        "곧 완료됩니다.",
        "끝나가고 있어요.",
        "계속 진행 중입니다.",
        "잠시만 기다려주세요.",
        "창작의 고통을 느끼는 중..."
    ]

    # 메시지를 랜덤으로 출력하는 함수 (백그라운드에서 실행)
    def print_random_messages(stop_event):
        while not stop_event.is_set():  # stop_event가 set될 때까지 메시지를 계속 출력
            print(random.choice(messages))
            time.sleep(15)  

    stop_event = threading.Event()
    message_thread = threading.Thread(
        target=print_random_messages, args=(stop_event,), daemon=True
    )
    message_thread.start()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    if INPUT.startswith("http"):
        url = INPUT
        web_docs = WebBaseLoader(url).load()
        if web_docs[0].metadata["title"]:
            title = web_docs[0].metadata["title"]
        else:
            title = ""
        docs = f"title : {title} \n\n" + web_docs[0].page_content
    else:
        docs = str(INPUT)
    documents = [Document(page_content=docs)]
    SPLITS = text_splitter.split_documents(documents)
    refined = documents_filter(SPLITS)
    new_script = generate_script(refined)

    stop_event.set()
    return new_script
  ```

### `tanslator_module`
  - 다국어 지원, 사용자 입력을 번역
  - 한국어, 영어, 일본어, 스페인어 지원

```py
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)


def translator(TEXT, LANG_CODE='kor_Hang'):
    """
    LANG_CODE: >-
    eng_Latn,
    jpn_Jpan,
    kor_Hang,
    spa_Latn
    """
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(TEXT, return_tensors="pt")
    generated_tokens = model.generate(
        inputs.input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODE]
    )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text
```

## 부가 기능 추가

### Stream 기능 추가
- 답변을 생성되는 대로 쪼개서 출력
- 답변 생성을 기다리는 시간 단축

```py
if query:
    chain = chain_maker(script, LANG)
    h_chain = history_chain(chain, store)
    response = h_chain.stream(
        # 질문 입력
        {"question": query},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": ID}},
    )
    for chunk in response:
        print(chunk, end="", flush=True)

    while True:
        print("\n========================\n")
        query = input("반응을 입력하세요.")
        if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
        response = h_chain.stream(
            # 질문 입력
            {"question": query},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": ID}},
        )
        if LANG_CODE:
            print(translator(query, LANG_CODE))
        else:
            print(query)
        print("\n답변:\n")
        for chunk in response:
            print(chunk, end="", flush=True)
            
print(f"[대화 세션ID]: {ID}")
```

### 이전 대화 이어서 하기
- `session_id` 를 기준으로 대화의 맥락이 저장됨
- `session_id` 와 사용된 스크립트를 별도의 메모리에 저장
- 스크립트 선택 과정을 생략하고 대화 내역과 스크립트를 호출하여 대화 진행

```py
conversation_session[ID] = script

chain = chain_maker(conversation_session['test40'], LANG)
h_chain = history_chain(chain, store)
while True:
        print(f"[대화 세션ID]: {ID}")
        print("\n========================\n")
        query = input("반응을 입력하세요.")
        if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
        response = h_chain.stream(
            # 질문 입력
            {"question": query},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": ID}},
        )
        if LANG_CODE:
            print(f'사용자 : {translator(query, LANG_CODE)}')
        else:
            print(f'사용자 : {query}')
        print("\n답변:")
        for chunk in response:
            print(chunk, end="", flush=True)
print(f"[대화 세션ID]: {ID}")
```

```
[대화 세션ID]: test40

========================

사용자 : 어디까지 이야기 했지?

답변:
[대화 세션ID]: test40
우리가 이야기한 건 김모 씨라는 남자의 배경과 그가 위조지폐를 만들기로 결심한 이유였어. 그리고 2005년부터 본격적으로 위조지폐를 유통시키기 시작한 상황까지였지. 그가 CCTV가 없는 상점에서 저렴한 물건을 사서 잔돈으로 생계를 이어갔다는 것도 이야기했어. 

그런데, 이렇게 쉽게 위조지폐가 유통될 수 있었던 이유는 뭘까? 너는 어떻게 생각해?[대화 세션ID]: test40

========================
```