---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 5
subtitle: TIL Day 58
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

## 최종 작동 평가

### 평가 환경
- **Jupyter Notebook**
- **VSCode**

### 패키지 관리
- `ipywidgets-8.1.5` : **Jupyter Notebook** 에서 입력 기능 설정

### RAG Chain 코드 개선

#### 전달 함수 수정
- `context` 전달할 때 줄바꿈 제거

```py
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs


class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # 문서 내용을 텍스트로 변환
        if isinstance(inputs, list):
            context_text = [doc.page_content.replace("\n", " ") for doc in inputs]
            
        else:
            context_text = inputs

        # 프롬프트 템플릿에 적용
        formatted_prompt = self.prompt_template.format_messages(
            context=context_text, question=inputs.get("question", "")
        )
        return formatted_prompt


# Retriever를 invoke() 메서드로 래핑하는 클래스 정의
class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        if isinstance(inputs, dict):
            query = inputs.get("question", "")
        else:
            query = inputs
        # 검색 수행
        response_docs = self.retriever.invoke(query)
        return response_docs
```

### Lang Chain 생성 수정
- 외부에서도 프롬프트와 참고문서를 입력 받을 수 있게 수정

```py
from langchain.chains import LLMChain

PROMPT_BASELINE = "Answer the question using only the following context."
REFERENCE_BASELINE = "check user qestion"

def RAGChainMake(VECTOR_STORE, PARAMS, PROMPT=PROMPT_BASELINE, REFERENCE=REFERENCE_BASELINE, **kwargs):
    """
    RAG 기법을 이용한 대화형 LLM 답변 체인 생성 (히스토리 기억 및 동적 대화 기능 포함)

    VECTOR_STORE : Retriever가 검색할 벡터 스토어
    PARAMS       : API Key 및 LLM 모델명 등의 환경 변수 포함
    PROMPT       : 시스템 초기 프롬프트 (기본값 설정)
    REFERENCE    : 추가 문맥 정보 (선택 사항)
    """
    # 벡터 스토어에서 유사한 문맥 검색
    retriever = VECTOR_STORE.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )

    # API 키 설정
    openai.api_key = os.environ.get(PARAMS.KEY)
    llm_model = ChatOpenAI(
        model=PARAMS.LLM_MODEL,
        api_key=openai.api_key,
    )

    # 대화형 프롬프트 생성
    contextual_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f'{PROMPT} \n\n reference : {REFERENCE}'),
            ("user", "Context: {context}\n\nQuestion: {question}"),
        ]
    )

    # RAG 체인 설정
    rag_chain_debug = {
        "context": RetrieverWrapper(retriever),
        "prompt": ContextToPrompt(contextual_prompt),
        "llm": llm_model,
    }

    return rag_chain_debug
```
### 대화형 답변 생성 기능
- 사용자의 입력을 받아 답변 진행
- 대화 내용을 기록
- 대화가 종료되면 저장 여부를 확인하여 저장

```py
from RAG_Module.Prompt_Engineering import *

def RAG_Coversation(CHAIN, PARAMS, **kwargs):
    """
    사용자로부터 질문을 받아 RAG 체인 기반으로 답변을 생성하는 대화형 함수
    전체 대화 결과를 리스트에 저장
    PARMS에 프롬프트 사용 및 결과 저장을 위한 PromptParams 입력
    """
    print("대화를 시작합니다. 종료하려면 'exit'를 입력하세요.\n")

    conversation_history = []  # 대화 기록을 저장할 리스트

    while True:
        print("========================")
        query = input("질문을 입력하세요 : ")

        if query.lower() == "exit":
            print("대화를 종료합니다.")
            break

        # 1. Retriever로 관련 문서 검색
        response_docs = CHAIN["context"].invoke({"question": query})

        # 2. 문서를 프롬프트로 변환
        prompt_messages = CHAIN["prompt"].invoke(
            {"context": response_docs, "question": query}
        )

        # 3. LLM으로 응답 생성
        response = CHAIN["llm"].invoke(prompt_messages)

        print("\n답변:")
        print(response.content)

        conversation_history.append({"question": query, "answer": response.content})

    while True:
        save_result = input("\n결과를 저장하시겠습니까? (y/n): ").strip().lower()

        if save_result == "y":
            PromptResult(conversation_history, PARAMS, **kwargs)
            print("결과가 저장되었습니다.")
            break  
        elif save_result == "n":
            print("결과가 저장되지 않았습니다. 대화를 종료합니다.")
            break  
        else:
            print(
                "잘못된 입력입니다. 다시 입력해주세요."
            ) 
```

### Chatbot 작동 테스트

```py
from RAG_Module.RAG_Params import *
from RAG_Module.PDF_Loader import PDFLoader
from RAG_Module.VecotorStore_Utils import VectorStoreReturn
from RAG_Module.RAG_Chain import *
from RAG_Module.Prompt_Engineering import *

# RAG 구성을 위한 파라미터
rag_setting = RAGParams(
    KEY="MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL="gpt-3.5-turbo",
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH=None,
    IS_SAFE=True,
    CHUNK_SIZE=500,
    CHUNK_OVERLAP=100,
)

# 프롬프트 작성 및 사용을 위한 파라미터
prompt_setting = PromptParams(
    KEY="MY_OPENAI_API_KEY",
    LLM_MODEL="gpt-3.5-turbo",
    PROMPT_PATH="Prompts/",
    PROMPT_NAME="test_prompt",
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)

# 문서 분할 및 벡터 스토어 생성
docs = PDFLoader(rag_setting)
vector_store = VectorStoreReturn(docs, rag_setting)

# 프롬프트 템플릿 작성
template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    answer is about large language model
    prioritize the context of the question
    specify if there are any new information
    If you can identify the original source of the document you referenced, write in APA format

    """,
    REFERENCE= "context given in the question",
)

# 프롬프트 생성
prompt = PromptTemplate(template_setting)

# 챗봇 생성
## 벡터 스토어, RAG구성 파라미터, 프롬프트 입력
chatbot_mk3 = RAGChainMake(vector_store, rag_setting, prompt)

# 챗봇과 대화 시작
RAG_Coversation(chatbot_mk3, prompt_setting)
```

### 답변 비교
- 질문 : 업스테이지의 solar 모델에 대해 설명해줘.
- 확인 사항 : solar 모델은 2023년에 개발된 업스테이지의 최신 모델

#### GPT-3.5-turbo
- 최신 정보를 반영하지 못하고 할루시네이션 발생

```
업스테이지의 Solar 모델은 한국의 여성 가수이자 댄서인 솔라(Solar)의 이름을 딴 제품입니다. Solar 모델은 업스테이지의 다양한 제품 중 하나로, 솔라의 이미지와 스타일을 반영하여 디자인되었습니다. 이 제품은 솔라의 개성을 살린 컬러나 디자인으로 구성되어 있으며, 솔라의 음악과 무대 퍼포먼스와 어울리는 메이크업을 연출할 수 있도록 제작되었습니다. Solar 모델은 솔라의 팬이나 메이크업을 즐기는 이들에게 특별한 즐거움을 주는 제품으로 소개되고 있습니다
```

#### GPT-4o-mini
- 최신 정보를 반영하여 solar 모델에 대한 정보를 제공
- 상세한 정보를 반영하고 있지 않고 대략적인 정보만 제공

```
업스테이지(Upstage)의 Solar 모델은 자연어 처리(NLP)와 관련된 AI 모델 중 하나로, 주로 대화형 AI 시스템이나 문서 이해, 질의응답 시스템 등 다양한 언어 기반 작업을 수행할 수 있는 고성능 언어 모델입니다. Solar 모델은 대규모 데이터셋을 기반으로 학습된 트랜스포머 아키텍처를 사용하며, 한국어와 영어를 포함한 다양한 언어에서 뛰어난 성능을 발휘할 수 있도록 최적화되어 있습니다.

Solar 모델의 특징:

대규모 언어 모델: 대량의 텍스트 데이터를 학습하여, 다양한 언어적 상황에서 의미를 파악하고 이해할 수 있습니다.
트랜스포머 기반: Transformer 아키텍처를 기반으로 하기 때문에 문맥을 잘 이해하고, 긴 텍스트나 복잡한 문장 구조도 효과적으로 처리할 수 있습니다.
다양한 활용 가능성: 문서 요약, 감정 분석, 기계 번역, 대화형 AI, 질의응답 시스템 등 여러 가지 언어적 작업에서 활용될 수 있습니다.
한국어 최적화: 업스테이지는 특히 한국어에 최적화된 모델을 제공하고 있어, 한국어 처리 성능이 뛰어나고, 한국어를 사용하는 다양한 실시간 응용 프로그램에 효과적입니다.
이 모델은 업스테이지의 다른 AI 시스템들과 함께 사용되며, 기업 고객이나 연구자들이 효율적으로 텍스트 기반의 문제를 해결하는 데 도움을 줍니다
```

#### RAG + GPT-3.5-turbo
- GPT-3.5-turbo에 없는 정보를 RAG 검색을 이용하여 답변
- 업스테이지에서 작성한 문서르 참고했기 떄문에 모델명 등의 상세한 정보를 제공

```
- 업스테이지의 Solar 모델은 Llama2를 파인튜닝하여 개발된 Solar-0-70b 모델입니다.
- Solar 모델은 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다.
- Solar 모델은 한국어와 영어를 모두 지원하고 있습니다."
```

### 결론
> RAG를 이용하면 찾는 정보가 LLM의 데이터에 없는 경우나 있더라도 상세한 내용을 설명하지 못하는 경우에 보다 자세하고 신뢰할 수 있는 답변을 생성 할 수 있다.

### 복수의 프롬프트 결과 비교

#### 파라미터 설정

```py
rag_setting = RAGParams(
    KEY="MY_OPENAI_API_KEY",
    EBD_MODEL="text-embedding-ada-002",
    LLM_MODEL="gpt-3.5-turbo",
    PDF_PATH="documents/초거대 언어모델 연구 동향.pdf",
    SAVE_PATH=None,
    IS_SAFE=True,
    CHUNK_SIZE=500,
    CHUNK_OVERLAP=100,
)

prompt_setting = PromptParams(
    KEY="MY_OPENAI_API_KEY",
    LLM_MODEL="gpt-4o-mini",
    PROMPT_PATH="Prompts/",
    PROMPT_NAME=None,
    PROMPT_EXTENSION="txt",
    RESULT_PATH="Results/",
    RESULT_EXTENSION="txt",
)
```

#### 벡터 스토어 생성 

```py
# 문서 불러오기 및 분할
docs = PDFLoader(rag_setting)

# 벡터 스토어 생성
vector_store = VectorStoreReturn(docs, rag_setting)
```

#### 프롬프트 1 작성 및 저장

```
prompt_1 = "Answer the question using only the following context."
PromptSave(prompt_1, prompt_setting, PROMPT_NAME='prompt_1')
```

#### 프롬프트 2 작성 및 저장

```
prompt_2 = """

you are specialist of large language model
answer question
refer to context qiven in question

"""
PromptSave(prompt_2, prompt_setting, PROMPT_NAME='prompt_2')
```

#### 프롬프트 3 작성 및 저장

- shot 기법 사용을 위한 shot 제작용 프롬프트 생성

```
shot_template = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    if you canspecify the date standard of the information
    if you can identify the original source of the document you referenced, write in APA format
    """,
    REFERENCE="only the latest information")
shot_prompt = PromptTemplate(shot_template)
```

- shot 생성 - 상위 모델인 gpt-4o-mini르 사용, 답변 방식을 참고하도록 유도

```
question = "gpt-4에 대해서 설명해줘"
shot = LLMSupport(question, prompt_setting, shot_prompt)
```

- 프롬프트 3 작성 및 저장

```
template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="only in korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    prioritize the context in question
    specify if there are any new information

    if you can identify the original source of the document you referenced, write in APA format
    """,
    REFERENCE=f"""

    <answer format sample>
    {shot}
    </answer format>
    
    refer to context given in the question",
    """
)
prompt_3 = PromptTemplate(template_setting)
PromptSave(prompt_3, prompt_setting, PROMPT_NAME='prompt_3')
```

#### 저장된 프롬프트 1, 2, 3 불러오기 및 결과 저장
- `QUESTION` : 리스트 형태로 입력

```py
def AutoChain(PARAMS, VECTOR_STORE, QESTION, **kwargs):
    """
    미리 설정된 질문을 바탕으로 대화 없이 결과를 저장하는 함수
    폴더 내 모든 프롬프트 파일에 대해 실행하고 결과를 저장
    QUESTION 은 리스트 형태로 주어져야 함
    """

    # 1. 프롬프트 폴더 내의 모든 프롬프트 파일을 불러오기
    prompt_files = [f for f in os.listdir(PARAMS.PROMPT_PATH) if f.endswith(f".{PARAMS.PROMPT_EXTENSION}")]
    
    if not prompt_files:
        print("프롬프트 파일이 없습니다. 종료합니다.")
        return

    # 각 프롬프트 파일에 대해 처리
    for prompt_file in prompt_files:
        print(f"\n{prompt_file} 로 시작합니다.")

        # 2. 프롬프트 불러오기
        prompt_name = prompt_file.split('.')[0]  # 확장자를 제외한 파일 이름
        prompt = PromptLoad(PARAMS, PROMPT_NAME = prompt_name)
        if not prompt:
            print(f"{prompt_file} 로드 실패. 스킵합니다.")
            continue  # 실패하면 해당 프롬프트를 건너뜁니다.

        # 3. RAG 체인 만들기
        chain = RAGChainMake(VECTOR_STORE, PARAMS, PROMPT=prompt, **kwargs)

        conversation_history = []  # 대화 기록을 저장할 리스트

        for query in QESTION:
            print(f"질문: {query}")

            # 1. Retriever로 관련 문서 검색
            response_docs = chain["context"].invoke({"question": query})

            # 2. 문서를 프롬프트로 변환
            prompt_messages = chain["prompt"].invoke(
                {"context": response_docs, "question": query}
            )

            # 3. LLM으로 응답 생성
            response = chain["llm"].invoke(prompt_messages)

            print("답변:", response.content)

            conversation_history.append({"question": query, "answer": response.content})

        # 결과 저장
        PromptResult(conversation_history, PARAMS, PROMPT_NAME = prompt_name, **kwargs)
        print(f"{prompt_file}에 대한 결과 저장 완료.")
    
    print("모든 결과가 저장되었습니다.")
```

### 최종 결과
> RAG로 정보를 제공하고 상대적으로 고성능 모델의 답변 방식을 예시로 제공하면<br>
>저성능 모델에서도 의미있는 답변을 얻을 수 있음을 확인

#### question
```
 업스테이지의 solar 모델에 대해 설명해줘
```

#### prompt 1
- prompt

```
Answer the question using only the following context.
```

- answer

```
"업스테이지는 Llama2를 파인튜닝하여 Solar-0-70b 모델을 개발하였으며, 이 모델은 한국어와 영어 모두 지원하는 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다."
```

#### prompt 2
- prompt

```
you are specialist of large language model
answer question
refer to context qiven in question
```

- answer

```
"업스테이지의 Solar-0-70b 모델은 Llama2를 파인튜닝하여 개발된 한국어 LLM입니다. 이 모델은 한국어와 영어를 모두 지원하며, 글로벌 LLM 플랫폼 중 하나인 Poe.com에서 서비스되고 있습니다."
```

#### prompt 3
- prompt

```
    persona : specialist of large language model
    language : only in korean
    tone : professional
    purpose : study large language model
    how to write : itemization
    condition : 

    <must obey>
    answer is about large language model
    answer that you do not know what you do not know
    </must obey>

    prioritize the context in question
    specify if there are any new information

    if you can identify the original source of the document you referenced, write in APA format
    
    reference : 

    <answer format sample>
    GPT-4에 대한 설명은 다음과 같습니다:

   1. **모델 개요**:
      - GPT-4는 OpenAI에서 개발한 대규모 언어 모델로, 자연어 처리(NLP) 작업을 수행하는 데 사용됩니다.
      - 이전 버전인 GPT-3에 비해 더 많은 파라미터와 개선된 알고리즘을 통해 성능이 향상되었습니다.

   2. **기능**:
      - 텍스트 생성: 주어진 프롬프트에 따라 자연스러운 문장을 생성할 수 있습니다.
      - 질문 응답: 사용자의 질문에 대해 관련 정보를 바탕으로 답변을 제공합니다.
      - 번역: 여러 언어 간의 번역 작업을 수행할 수 있습니다.
      - 요약: 긴 텍스트를 간결하게 요약하는 기능을 갖추고 있습니다.

   3. **훈련 데이터**:
      - GPT-4는 다양한 출처의 대규모 텍스트 데이터로 훈련되었습니다. 이는 웹사이트, 책, 논문 등 다양한 형식의 데이터를 포함합니다.
      - 훈련 데이터는 2023년 10월까지의 정보로 제한되어 있습니다.

   4. **응용 분야**:
      - 고객 서비스: 챗봇 및 가상 비서로 활용됩니다.
      - 콘텐츠 생성: 블로그, 기사, 소설 등 다양한 콘텐츠를 생성하는 데 사용됩니다.
      - 교육: 학습 도구로 활용되어 학생들에게 정보를 제공합니다.

   5. **한계**:
      - GPT-4는 훈련 데이터에 기반하여 작동하므로, 최신 정보나 특정 전문 분야에 대한 깊이 있는 지식이 부족할 수 있습니다.
      - 모델이 생성하는 정보의 정확성이나 신뢰성은 항상 보장되지 않으며, 사용자는 결과를 비판적으로 평가해야 합니다.

   6. **미래 전망**:
      - GPT-4와 같은 모델은 지속적으로 발전하고 있으며, 향후 더 나은 성능과 다양한 기능을 갖춘 모델이 개발될 것으로 예상됩니다.

   이 정보는 OpenAI의 공식 발표 및 연구 자료를 바탕으로 작성되었습니다. (OpenAI, 2023)
    </answer format>
    
    refer to context given in the question",
```

- answer

```
"업스테이지의 Solar 모델에 대한 설명은 다음과 같습니다:
1. **모델 개요**:
   - Solar 모델은 업스테이지에서 개발한 대규모 언어 모델입니다.
   - Llama2를 기반으로 파인튜닝되어 생성되었습니다.
      
2. **지원 언어**:
   - Solar 모델은 한국어와 영어를 모두 지원합니다.

3. **응용 분야**:
   - 해당 모델은 글로벌 LLM 플랫폼인 Poe.com에서 서비스되고 있습니다.
   
4. **훈련 데이터**:
   - 한국어 데이터와 공개된 한국어 데이터, 크롤링 데이터를 활용하여 학습하였습니다.
   
5. **모델의 특징**:
   - 한국어 토큰 비율을 높여 한국어 처리 성능을 개선하는 데 중점을 두고 있습니다.
   
이 정보는 제공된 문서의 내용을 바탕으로 작성되었습니다. (초거대 언어모델 연구 동향.pdf, p. 3)"
```