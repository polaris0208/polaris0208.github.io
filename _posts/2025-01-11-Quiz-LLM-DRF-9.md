---
layout: post
title: Quiz_LLM Function Calling & Structured Outputs
subtitle: TIL Day 110
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

## Function Calling
- 함수 호출 기능
- 미리 정의된 함수를 호출하여서 값을 얻고 답변에 반영하는 기능

### 활용
- **API** : 호출
  - **API**를 호출하여 날씨 등의 데이터를 얻어 답변에 반영
  - 작업 자동화 : 사용자 입력에서 데이터르 추출하여 복잡한 계산이나 정확한 값이 요구되는 작업을 자동으로 수행
  - 정보 추출/검색 : 입력된 문서에서 특정 데이터들을 추출하거나, 사용자 입력에서 인자를 추출하여 **DB**에서 문서 검색

## Structured Outputs
- 사용자가 원하는 형태에 맞춰 답변을 구조화 하여 전달하는 기능

### 방식
- **Function Calling** 을 통한 방식
  - `tools` 파라미터 내에서 `strict: true`를 설정한 뒤에 출력할 데이터 구조를 입력
  - 함수 호출을 지원하는 모델에서 모두 사용 가능
  - 실제 `Structured Outputs` 기능을 사용하는 것은 아님

- **Response Format** 지정 방식
  - 실제 `Structured Outputs` 기능 : 2024년 8월부터 추가된 기능
    - `gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06` 이상의 모델에서 지원
  - `Response Format`에 데이터 구조를 정의
  - 작동 방식
    - 사용자가 정의한 구조를 모델에 문법으로 제공 : 문맥 자유 문법(**Context-Free Grammar**)
    - 해당 문법에 해당하지 않는 토큰을 마스킹하는 방식으로 확률을 0으로 만들어 사용자가 원하는 구조의 답변만 반환

### 활용
- 데이터 추출 및 정형화 : 텍스트에서 필용한 데이터만 추출한 뒤에 원하는 구조로 정리
- **UI** 생성 : 추출한 데이터를 미리 정의한 **UI** 구조에 자동으로 적용하여 생성
- 분석 결과 구조화 : 데이터를 추출한 뒤에 모델을 통해 얻은 분석 결과와 함께 정리

### 장점
- 원하는 구조에 데이터를 얻을 수 있어 **API**에 유용하게 활용 가능
- 예측가능한 형태의 출력을 얻을 수 있어 오류 가능성이 감소
- 별도의 데이터 변환 과정이 불필요
- 프롬프트에 예시 데이터 구조를 넣지 않아도 되기 때문에 추가 컨텍스트 공간 확보
  - 프롬프트에는 추가적인 지시사항 추가 가능, 긴 컨텍스트로 인해 답변의 질적 수준이 하락하는 것 방지

## `Structured Outputs` 활용한 퀴즈 생성 **API** 테스트
- `Pydantic`, `.chat.completions.parse` 사용
  - `Pydantic` : `Python`의데이터 검증과 데이터 파싱을 위한 라이브러리
  - `.chat.completions.parse`
    - 정의된 `Pydantic` 모델이나 다른 구조화된 형식으로 자동 파싱된 응답을 반환
    - 응답을 특정 형식으로 변환하려는 경우에 사용
    - 바로 사용 가능한 구조화된 데이터 반환
  - `.chat.completions.create`
    - 일반적인 대화형 텍스트 응답을 생성, 결과는 문자열 형태로 반환
    - 응답을 그대로 사용하거나 후속 작업을 통해 가공

### 데이터 구조 정의
- **API** 에서 사용하는 `Quiz` 모델의 구조와 동일하게 생성

```py
from pydantic import BaseModel

class QuestionChoice(BaseModel):
    id: int
    content: str
    is_correct: bool

class Question(BaseModel):
    id: int
    content: str
    answer_type: str
    choices: list[QuestionChoice]

class QuizResponse(BaseModel):
    # id: int # DB에서 자동 생성
    title: str
    description: str
    questions: list[Question]
```

### 퀴즈 생성
- `response_format`에 데이터 구조 입력
- `json.dumps` : 응답 결과는 모델형태, `json` 문자열로 변환
- `indent=2` : 들여쓰기 설정

```py
import os
import json
import openai
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 클라이언트 생성
client = OpenAI(api_key=openai.api_key)

# 응답 생성
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "system", "content": "make quiz"},
        {"role": "user", "content": "capital city quiz, 4-multi-choice, 10 question, hard difficulty"},
    ],
    response_format=QuizResponse,  # 응답 포맷 설정, 최상위 모델
)

# 응답 데이터
quiz = completion.choices[0].message.parsed

# JSON 형태로 추출
quiz_json = json.dumps(quiz.model_dump(), indent=2)

# 출력된 JSON 데이터
print(quiz_json)
``` 

### 결과
- 프롬프트에 예시 형태를 제시하지 않아도 의도한 형태의 결과를 일정하게 출력

```py
{
  "id": 1,
  "title": "Capital City Quiz",
  "description": "Test your knowledge of the world's capitals with this challenging quiz!",
  "questions": [
    {
      "id": 1,
      "content": "What is the capital city of Bhutan?",
      "answer_type": "multiple-choice",
      "choices": [
        {
          "id": 1,
          "content": "Thimphu",
          "is_correct": true
        },
        {
          "id": 2,
          "content": "Kathmandu",
          "is_correct": false
        },
        {
          "id": 3,
          "content": "Dhaka",
          "is_correct": false
        },
        {
          "id": 4,
          "content": "New Delhi",
          "is_correct": false
        }
      ]
    },
    ...
  ]
}
```
