---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 정리
subtitle: TIL Day 70
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

# Chatbot 기능 정리
> 챗봇 가능 및 코드 정리

## 패키지 버전 관리

```bash
torch==2.5.1
numpy==1.26.4
openai==1.55.2
pydantic==2.10.2
langchain==0.3.9
streamlit==1.40.2
transformers==4.37.0
langchain-core==0.3.21
beautifulsoup4==4.12.3
langchain-chroma==0.1.4
langchain-ollama==0.2.0
langchain-openai==0.2.10
langchain-community==0.3.8
```

## `docs_utils`
- 문서 전처리 기능

### 패키지 관리

```py
import os
import json
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### `def process_json_data(json_files):`
- 여러 JSON 파일을 읽고 데이터를 통합한 후 특정 형식의 문자열 리스트로 반환
- **Parameters**: `json_files` `(list)`: `JSON` 파일 경로 리스트
- **Returns**: 파일 데이터에서 `title`과 `content`를 읽어 특정 형식으로 변환한 리스트
  - `title` 이 본문에 포함되어야 **RAG** 검색에 용이

```py
all_json_data = []
for file_path in json_files:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            all_json_data.extend(data)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다 - {file_path}")
    except json.JSONDecodeError:
        print(f"Error: JSON 파일 형식이 잘못되었습니다 - {file_path}")

return [
    f"Title: {item.get('title', 'N/A')}\nContent: {item.get('content', 'N/A')}"
    for item in all_json_data
]
```

### `def title_json_data(json_files):`
- 여러 JSON 파일을 읽고 데이터를 통합한 후 특정 형식의 문자열 리스트로 반환
- **Parameters**: json_files (list): JSON 파일 경로 리스트
- **Returns**: 파일 데이터에서 `title`만 읽어 반환
  - 데이터 인덱스로 사용될 제목 목록 추출

### `def documents_filter(SPLITS):`
- 분할된 데이터에서 불필요한 데이터를 제거하고 하나로 결합
- `ConversationSummaryMemory`에 이전 내용을 요약하여 저장
- 아전 내용과 대조해서 불필요한 데이터 구분
- **Parameters**: 분할된 텍스트 데이터
- **Returns**: 텍스트 데이터

```py
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai.api_key,
    max_tokens=1000,
    temperature=0.0,
)
summaries = []
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

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
        summary = chain.invoke({"SPLIT": SPLIT, "context": context})
        memory.save_context({"input": f"summary # {count}"}, {"output": summary})
        summaries.append(summary)
        count += 1

     except Exception as e:
        # 오류 처리: 만약 API 호출 중에 문제가 발생하면 오류 메시지 추가
        print(f"Error summarizing document: {e}")
        summaries.append(f"Error summarizing document: {e}")

return "".join(summaries)
```

## `db_utils`
- **Chroma DB** 관련 기능
- 생성, 불러오기, 저장

### 패키지 관리

```py
import os
import openai
import warnings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
```

### 임베딩 모델
- `EMBED = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai.api_key)`

### `def create_vstore(DOCS, DB_NAME: str, DB_PATH):`
- **Chroma DB** 생성
- **Parameters**
  - `DOCS` : 포함할 문서 ; `documents` 객체
  - `DB_NAME` : **DB** 생성시 설정한 이름
  - `DB_PATH` : **DB** 경로
- **Returns**:
  - **Chroma** 객체

```py
return Chroma.from_documents(
    documents=DOCS,
    collection_name=DB_NAME,
    persist_directory=DB_PATH,
    embedding=EMBED,
    )
```

### `def load_vstore(DB_NAME: str, DB_PATH):`
- 로컬에 저장된 **Chroma DB** 를 불러옴
- **Parameters**
  - `DB_NAME` : **DB** 생성시 설정한 이름
  - `DB_PATH` : **DB** 경로
- **Returns**:
  - **Chroma** 객체

```py
return Chroma.from_documents(
    collection_name=DB_NAME,
    persist_directory=DB_PATH,
    embedding=EMBED,
    )
```

### `def add_to_vstore(SCRIPT, DB):`
- 새로 생성된 스크립트를 **documets** 객체로 변경 후 **Chroma DB**에 저장
- **Parameters**
  - `SCRIPT` : 추가할 스크립트  ; `documents` 객체
  - `DB` : 저장할 **DB**
- **Returns** : **Chroma DB**에 데이터 추가

## `script_utils`
- **LLM**이 대화에 사용할 스크립트 생성

### 패키지 관리

```py
import os
import time
import openai
import random
import threading
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### `def generate_script(summaries):`
- **LLM**이 전달할 이야기의 대본 생성
- **Parameters**: 필터링된 텍스트 데이터
- **Returns**: 텍스트 데이터

#### 프롬프트
- 역할 : 스크립트 작가
- 문서에 나타난 사실에 근거
  - 시간, 인물, 사건 중심으로 작성
  - 없는 내용은 창작하지 않고 빈칸으로 남김
- **shot** : 소설의 구성 방법

```
    persona = script writer
    language = only in korean
    least 3000 tokens
    use input,
    refer to sample,
    write about time, character, event,
    write only fact,
    ignore the mere listing of facts and write N/A,
    if input is None write '문서가 비어있습니다. URL을 확인해주세요.'
 
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

#### 코드

```py
def generate_script(summaries):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai.api_key,
        max_tokens=5000,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_template(
  
    )
    chain = prompt | llm | StrOutputParser()
    script = chain.invoke({"summaries": summaries})
    return script
```

### `def script_maker(INPUT: str):`
- 사용자가 입력한 데이터로 **LLM**이 전달할 이야기의 대본 생성
- 입력 텍스트의 형태로 **URL** 과 텍스트를 구분
- **Parameters**: **URL** 또는 텍스트 데이터
- **Returns**: 텍스트 데이터

```py
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    if INPUT.startswith("http"):
        url = INPUT
        web_docs = WebBaseLoader(url).load()
        if web_docs[0].metadata["title"]:
            title = web_docs[0].metadata["title"]
        else:
            title = ''
        docs = f"title : {title} \n\n" + web_docs[0].page_content
    else:
        docs = str(INPUT)
    documents = [Document(page_content=docs)]
    SPLITS = text_splitter.split_documents(documents)
    refined = documents_filter(SPLITS)
    new_script = generate_script(refined)
    return new_script
```

## `llm_utils`
- **llm**이 사용자와 대화하기 위해 필요한 기능

### 패키지 관리

```py
import os
import time
import openai
import streamlit as st
from openai import OpenAI
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
```

### `def evaluator(query, db):`
- **DB**에서 찾아온 스크립트가 적절한지 판단하는 함수
- **Parameters**
  - `query` : 사용자 입력
  - `db` : 스크립트가 저장된 **DB**
- **Returns**
  - 연관 정도 점수
  - 스크립트 : 연관 정도가 적절한 경우

#### 프롬프트
- 역할 : 연관성 검증
- 조건 : 1-100 사이의 점수만 출력, 5단위 증감
- **shot** : 점수 책정예시

```
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
```

#### 코드

```py
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
    """
    )
    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"query": query, "script": script})
    if not score:
        return [0, "N/A"]
    return [int(score), script]
```

### `def streamlit_chain(script, history, language='korean'):`
- **streamlit** 환경에서 대화 맥락을 유지하여 이야기를 이어나가는 **chain** 생성
- **Parameters**
  - `scrip` : 선택된 스크립트
  - `history` 
    - `st.session.state['history'] = [{'role' : 'role_value', 'content' : 'content_value'}]` 형식
  - `language` : 사용할 언어
- **Returns**:
  - `llm chain`


#### 프롬프트
- 역할 : 스토리 텔러
- 조건 
  - 스크립트의 각 단계별로 끊어서 이야기 진행
  - 친구같은 말투 사용
  - 흥미를 유발하는 질문
  - 사용자의 답변에 맞장구
- 사용변수
  - `script` : 사용할 스크립트
  - `chat_history` : 대화 내역 / 대화 맥락과 스크립트 진행정도를 파악
  - `question` : 사용자 응답

```py
    persona : story teller
    language : only {language}
    tell dramatic story like talking to friend,
    speak informally,
    progress chapter by chapter,
    do not repeat same chapter,
    **hide header like '###'**,
    at first chapter give hook question like movie or tv show, 
    finish chapter with interesting rhetorical question,
    wait user answer,
    give reaction to answer,
    do not use same reaction or same question,
    end of the script give no question and wrap up the story,
    notice if story finished and give message '종료하려면 exit'

    # script
    {script}

    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 
```

#### 코드

```py
    message = "\n".join([f"{item['role']}: {item['content']}" for item in history])
    prompt = PromptTemplate.from_template(
        """
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.3)
    chain = (
        RunnableMap(
            {
                "language": lambda inputs: language,  # language는 고정값으로 전달
                "script": lambda inputs: script,  # script는 고정값으로 전달
                "chat_history": lambda inputs: message,  # 입력에서 chat_history 추출
                "question": itemgetter("question"),  # 입력에서 question 추출
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### `def stream_data(text):`
- **strealit** 환경 `st.write_stream()` 메서드와 사용
- 텍스트 출력을 스트림 형식으로 변환하는 함수
- **Parameters**
  - `text` : 스트림 형식으로 출력할 텍스트
-  **Returns**
  - 스트림 형식 출력

```py
    for word in text.split(" "):  # 공백 기준으로 문장을 단어 단위로 나누기
        yield word + " "
        time.sleep(0.2)
```

## `translator_module`
- 번역 기능 제공

### 패키지 관리
- `lang_code_to_id` 기능을 사용을 위해
  - `transformers==4.37.0` 이하 버전 설치

```py
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

### `def translator(TEXT, LANG_CODE='kor_Hang'):`
- `os.environ["TOKENIZERS_PARALLELISM"] = "false"` : 병렬 처리 제거
- `warnings.filterwarnings("ignore", category=FutureWarning)` : 경고 무시
- 텍스트 다국어 번역
- **Parameters**
  - `TEXT` : 번역할 텍스트
  - `LANG_CODE` : 사용할 언어의 코드
    - `eng_Latn, jpn_Jpan, kor_Hang, spa_Latn`
- **Returns** : 번역된 텍스트

```py
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