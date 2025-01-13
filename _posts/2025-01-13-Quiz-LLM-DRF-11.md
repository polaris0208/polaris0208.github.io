---
layout: post
title: Quiz LLM 개선
subtitle: TIL Day 112
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---
> 구조화된 답변 `Structured Outputs` 설정 및 프롬프트 엔지니어링

## 의존성 설정
- `pydantic` : 출력 형식 설정
- `random` : 정답 난수 설정
  - 문제 생성시 1번 또는 2번에 정답이 몰리는 것 방지

```py
import os
import json
import openai
import random
from openai import OpenAI
from pydantic import BaseModel
openai.api_key = os.getenv("OPENAI_API_KEY")
```
## 출력 형식 지정
- 퀴즈 - 문제 - 선택지 구조
- 들어가야할 요소, 변수명과 변수타입까지 지정

```py
def quizz_chain(content, input):
    # Pydantic 모델 정의
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

## 정답 위치 랜덤 생성
- 각 문제별 정답 번호를 랜덤으로 생성하여 프롬프트에 제시
- 문제 타입에 따라 프롬프트 수정

### 문제점 (진행중)
- 정답 번호는 랜덤으로 생성 완료
  - 프롬프트에 주어진 값 그대로 생성하지 않는 문제 발생

```py
    type = input.get('type', 'ox')
    count = input.get('count', 5)
    difficulty = input.get('difficulty', 'easy')
    correct_answer_distribution = []
    if type == '4_multiple_choice':
        for _ in range(1, int(count) + 1):
                correct_index = random.randint(1, 4)
                suffle = f"""
                qustions_id : {_},
                choices_id : {correct_index},
                is_correct : true
                """
                correct_answer_distribution.append(suffle)
        description = f'create {count}, {difficulty} quiz with 4_multiple_choice. and follow answer_sheet : {correct_answer_distribution}'
    elif type == 'ox':
        for _ in range(1, int(count) + 1):
                correct_index = random.randint(1, 2)
                correct_answer_distribution.append(correct_index)
        description = f'create {count}, {difficulty} quiz with true or false (O/X). and follow answer_sheet : {correct_answer_distribution}'
```

## 프롬프트 수정
- `Structured Outputs` 사용으로 컨텍스트 공간에 여유 확보
- 문제 생성에 대한 상세 지시 추가
  - 난이도별 문제 생성에 대한 상세한 지시 작성
  - 코드 문제에 대한 상세한 조건 추가

```py
    # OpenAI 클라이언트 설정
    client = OpenAI(api_key=openai.api_key)
    prompt = f"""
        must use **korean**
        Create a quiz about context. 

        **{description}**
   
        if 'easy': Problems related to core concepts or key information.
        if 'medium': Problems that could be confusing, involving detailed information.
        if 'hard': Application problems, including coding challenges.

        coding challenge is includes three types of problems:

        Select the expected output based on the given code.
        Choose the appropriate code that matches the given output.
        Fill in the blanks in the code with the correct options.

        make sure that the options include full sentences, not just short answers

        mark the correct answer for each question.

        context : {content}
        """
```

## 답변 생성
- `"gpt-4o-mini-2024-07-18"` : 구조화된 출력 지원 모델
- `response_format` : `pydantic`로 생성한 구조를 지정
- `completion.choices[0].message.parsed` : 지정된 형식으로 출력

```py
    # 퀴즈 데이터를 구조화하여 응답 받기
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.3,
        response_format=QuizResponse,  # 여기에서 QuizResponse 모델을 설정
    )

    # 응답 데이터
    quiz = completion.choices[0].message.parsed

    # JSON 형태로 추출
    quiz_json = json.dumps(quiz.model_dump(), indent=2)
    return quiz_json
```