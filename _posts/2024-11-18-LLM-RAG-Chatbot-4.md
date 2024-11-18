---
layout: post
title: LLM-RAG를 이용한 Chatbot 제작 - 4
subtitle: TIL Day 57
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM]
author: polaris0208
---

## 프롬프트 엔지니어링
> 프롬프트에 필요한 파라미터 및 모듈 제작<br>
> 프롬프트를 고도화 해가며 결과 비교

### 파라미터 설정
- 프롬프트 생성에 필요한 파라미터 관리
- `class PromptParams` : 프롬프트 생성, 저장, 불러오기에 필요한 파라미터
- `class TemplateParams` : 프롬프트를 지정한 양식에 맞춰 생성하기 위한 파라미터

```py
from dataclasses import dataclass

@dataclass
class PromptParams:
    KEY: str  # API Key 환경변수명
    LLM_MODEL: str  # LLM 모델명
    PROMPT_PATH: str  # 프롬프트 파일 경로
    PROMPT_NAME: str  # 프롬프트 파일 이름
    PROMPT_EXTENSION: str  # 프롬프트 파일 확장자
    RESULT_PATH: str  # 결과 파일 경로
    RESULT_EXTENSION: str  # 결과 파일 확장자


@dataclass
class TemplateParams:
    PERSONA: str    # LLM이 수행할 역할 지정
    LANG: str   # 답변 생성 언어
    TONE: str   # 답변의 어조 설정
    PERPOSE: str    # 목적 명시
    HOW_WRITE: str  # 답변 방식 예) 개조식
    CONDITION: str  # 추가할 조건
    REFERENCE: str  # 참조
```

### Prompt_Engineering 모듈
- `def PromptSave(PROMPT, PARAMS, PROMPT_NAME=None, **kwargs):`
  - 주어진 프롬프트를 파일에 저장하는 함수
  - 파일명은 개별 선언할 수 있게하여, 여러파일을 처리할 수 있도록 작성

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def PromptLoad(PARAMS, PROMPT_NAME=None, **kwargs):`
  - 저장된 프롬프트를 불러오는 함수
  - 파일명은 개별 선언할 수 있게하여, 여러파일을 처리할 수 있도록 작성

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def PromptResult(RESPONSE, PARAMS, PROMPT_NAME=None, **kwargs):`
  - 대화 결과를 파일에 저장하는 함수
  - `RESPONSE`가 문자열일 경우 그대로 저장
  - 리스트나 딕셔너리일 경우 `JSON` 형식으로 저장

```
PROMPT_PATH: str  # 프롬프트 파일 경로
PROMPT_NAME: str  # 프롬프트 파일 이름
PROMPT_EXTENSION: str  # 프롬프트 파일 확장자
RESULT_PATH: str  # 결과 파일 경로
RESULT_EXTENSION: str  # 결과 파일 확장자

PROMPT_NAME이 지정되지 않을 때 PARAMS.PROMPT_NAME을 사용.
```

- `def LLMSupport(QUESTION, PARAMS, SUPPORT_PROMPT="answser question", **kwargs):`
  - **shot** 기법 - **shot** 생성을 위한 함수
  - **LLM** 에게 예시를 생성하게 하여 프롬프트에 적용
  - 성능이 높은 모델로 예시 생성 - 성능이 낮은 모델에게 예시로 적용

```
QUESTION : shot 생성을 위한 질문
SUPPORT_PROMPT : shot 생성을 위한 프롬프트(선택)
KEY: str  # API Key 환경변수명
LLM_MODEL: str  # LLM 모델명
```

- `def PromptTemplate(PARAMS, **kwargs):`
  - `class TemplateParams` 객체를 파라미터로 입력
  - 프롬프트 작성에 필요한 요소들 정리
  - 필요 없는 부분은 공백('')으로 작성
  - 영어 사용, 명령어 사용, 리스트, 마크다운 작성 방식으로 성능을 향상 가능

```
PERSONA : LLM이 수행할 역할 지정
LANG : 답변 생성 언어
TONE : 답변의 어조 설정
PURPOSE : 목적 명시
HOW_WRITE : 답변 방식 예) 개조식
CONDITION : 추가할 조건
REFERENCE : 참조
```

### 프롬프트 템플릿 
- 역할, 목적, 답변 방식을 지정하여 전달
- 답변 생성 시 지켜야 할 조건 명시
- 답변에 참고할 내용 명시

```py
question = "RAG에 대해서 설명해주세요"
shot = LLMSupport(question, prompt_setting)


template_setting = TemplateParams(
    PERSONA="specialist of large language model",
    LANG="korean",
    TONE="professional",
    PERPOSE="study large language model",
    HOW_WRITE="itemization",
    CONDITION="""

    answer is about large language model
    prioritize the context of the question
    specify if there are any new information
    If you can identify the original source of the document you referenced, write in APA format

    """,
    REFERENCE=f"{shot} \n\n and context given in the question",
)

prompt = PromptTemplate(template_setting)
chatbot_mk3 = RAGChainMake(vector_store, rag_setting, prompt)
RAG_Coversation(chatbot_mk3, prompt_setting)
```

#### 결과
- 개조식으로 답변
- 전문 용어 사용과 구조화된 답변
- 찾은 여러가지의 지식 중 조건에 해당하는 지식을 구분하여 답변

```
질문을 입력하세요 :  RAG에 대해서 설명해주세요.

답변:
**RAG (Retrieval Augmented Generation)**에 대해 설명드리겠습니다:

- RAG는 위험 평가 격자를 의미하는 약자가 아닌, **Retrieval Augmented Generation**의 약자입니다.
- RAG는 주로 자연어 처리 분야에서 활용되며, 최근 초거대 언어모델 연구 동향에서 주목을 받고 있습니다.
- RAG는 정보 검색 (Retrieval)과 생성 (Generation)을 결합한 모델로, 정보를 검색하여 새로운 내용을 생성하는 기술을 지칭합니다.
- 이 모델은 이전에 생성된 텍스트나 문맥을 활용하여 보다 의미 있는 내용을 생성하고자 하는 데 사용됩니다.
- RAG는 자연어 이해, 생성, 정보 검색 등 다양한 작업에 활용되며, 다양한 응용 분야에서 유망한 기술로 평가되고 있습니다. 

이렇게, RAG는 자연어 처리 분야에서 중요한 기술 중 하나로 발전하고 있으며, 정보 검색과 생성을 결합한 혁신적인 모델로 주목받고 있습니다.
```

### 확인 된 문제
- 프롬프트 결과 저장 중 오류 발생

#### 원인
- 질문과 답변이 두 개 이상 이어지면 결과가 리스트 또는 딕셔너리의 형태로 저장됨
- 리스트 또는 딕셔너리를 `txt` 형태로 저장하기 위해서는 별도의 변환 과정 필요

#### 해결
- 프롬프트 결과 저장 함수를 수정
- 리스트나 딕셔너리 형식은 `json` 확장자로 변경하여 저장

```py
    # RESPONSE가 문자열일 경우
    if isinstance(RESPONSE, str):
        # 문자열을 그대로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(RESPONSE)

    # RESPONSE가 리스트나 딕셔너리일 경우
    elif isinstance(RESPONSE, (list, dict)):
        # 리스트나 딕셔너리를 JSON 형식으로 저장
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(RESPONSE, file, ensure_ascii=False, indent=4)

    else:
        print("지원하지 않는 RESPONSE 형식입니다.")
        return

    print(f"결과가 {file_path}에 {PARAMS.RESULT_EXTENSION} 형식으로 저장되었습니다.\n")
```

### 프롬프트 고도화
- 프롬프트 수정에 따른 답변 변화 확인
- `"question": "RAG에 대해서 설명해주세요"`

#### Prompt 1
- **RAG** 결과를 활용하도록 유도

```
Answer the question using only the following context.
```

```
"answer": "RAG은 Retrieval Augmented Generation의 약자로, 정보 검색을 통해 생성 모델을 보강하는 방법을 가리킨다."
```

#### Prompt 2
- 역할 지정
- **RAG** 결과를 활용하도록 구체적으로 유도

```
you are specialist of large language model
answer question
refer to context qiven in question
```

```
"answer": "RAG은 Retrieval Augmented Generation의 약자로, 정보 검색을 통해 생성 모델을 보강하는 방법론을 가리키는 용어입니다. 이 방법론은 정보 검색 기술을 활용하여 대규모 언어모델을 더욱 효과적으로 학습하고 활용할 수 있도록 설계되었습니다. RAG은 최근 자연어 처리 분야에서 주목을 받고 있으며, 관련 연구들이 활발히 진행되고 있습니다."
```

#### Prompt 3
- 역할, 답변 방식, 조건 구체적으로 명시
- **LLM** 생성한 답변 참고
- 결과
  - 역할에 맞게 전문용어를 사용하여 구조적으로 설명
  - **LLM** 의 답변 형태를 참고하여 답변 생성
  - 검색된 두 종류의 **RAG** 정보 중 **LLM** 과 관련된 정보만 답변에 포함

```
persona : specialist of large language model
language : korean
tone : professional
purpose : study large language model
how to write : itemization
condition : 

answer is about large language model
prioritize the context of the question
specify if there are any new information
If you can identify the original source of the document you referenced, write in APA format

    
reference : RAG는 Risk Assessment Grid의 약자로, 위험 평가 격자를 의미합니다. 이는 프로젝트나 활동을 평가하고 위험을 식별하기 위해 사용되는 도구입니다. RAG는 주로 세 가지 색상으로 표시되며, 각 색상은 다음을 나타냅니다:

- 빨강(Red): 심각한 위험이 있음을 나타냅니다. 이는 프로젝트나 활동이 큰 위험에 직면하고 있음을 의미합니다.
- 주황(Amber): 중간 수준의 위험이 있음을 나타냅니다. 이는 일부 위험이 존재하며 주의가 필요함을 의미합니다.
- 녹색(Green): 낮은 수준의 위험이 있음을 나타냅니다. 이는 프로젝트나 활동이 안정적이며 큰 위험 없이 진행될 수 있음을 의미합니다.

RAG를 사용하여 위험을 시각적으로 파악하고 관리자나 이해관계자에게 위험 상황을 전달하는 데 도움이 됩니다. 이를 통해 조직이 적절한 대응책을 마련하고 위험을 최소화할 수 있습니다. 

 and context given in the question
```

```
"answer": "**RAG(Retrieval Augmented Generation)**에 대해 설명드리겠습니다:

1. **개요**:
   - RAG는 Retrieval Augmented Generation의 약자로, 정보 검색을 강화한 생성 모델을 가리킵니다.

2. **기술적 측면**:
   - RAG는 주로 세 가지 기본 구성 요소로 이루어져 있습니다: 정보 검색(Re-trieval), 생성(Generation), 그리고 조합(Combination).
   - 정보 검색 단계에서는 대규모 텍스트 데이터베이스로부터 관련 정보를 검색합니다.
   - 생성 단계에서는 검색된 정보를 바탕으로 자연스러운 문장 또는 텍스트를 생성합니다.
   - 조합 단계에서는 정보 검색과 생성을 효과적으로 결합하여 높은 품질의 결과물을 얻습니다.
  
3. **적용 분야**:
   - RAG는 자연어 이해, 대화형 인터페이스, 정보 검색 및 요약, 문서 생성 등 다양한 분야에 적용되고 있습니다.
   
4. **장점**:
   - RAG는 대량의 데이터를 기반으로 학습하므로 일반적으로 높은 성능을 보입니다.
   - 정보 검색과 생성을 융합함으로써 상호 보완적인 기능을 제공합니다.

5. **활용**:
   - RAG는 다양한 자연어 처리 작업에 유용하게 활용될 수 있으며, 특히 정보 검색 및 요약에 큰 도움을 줄 수 있습니다.
   
이러한 방식으로 RAG는 정보 검색과 생성을 결합하여 효율적이고 정확한 결과물을 제공하는 초거대 언어모델의 한 형태로 주목받고 있습니다."
```