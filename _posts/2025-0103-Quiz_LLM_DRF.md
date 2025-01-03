# Quiz 생성 LLM 제작
> **LLM**을 이용하여 퀴즈를 생성 후 결과를 **DB**에 반영할 수 있도록 **DRF**의 모델로 정의

## LLM 모델 설계

### 의존성

```bash
langchain==0.3.13
langchain-community==0.3.13
langchain-openai==0.2.14
openai==1.58.1
```

```py
import os
import openai
from openai import OpenAI
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

### Prompt
- `json` 형태로 결과 반환
  - 특정 형식을 취하도록 예졔를 부여
`type` : 문제 형태
`count` : 문제 개수
`difficulty` : 난이도
`level` : 사용자 수준

```py
"""
must use **korean**
    Create a quiz about context.

    If {type} is '4_multiple_choice', create a multiple-choice question with 4 options.
    If {type} is 'ox', create a true or false (O/X) question.

    Include {count} {type} questions and mark the correct answer for each question.
    quiz must be {difficulty} for {level}.

    **return only json**
    **do not include```json```**

    <cotext>
    {content}
    </context>

    <example>
    "id": <int : quizz id>,
    "title": <str: quizz title>,
    "description": <str: quizz description>,
    "questions": [
    <dict : quizz list>
        <dict : quizz>
        "id": 1,
            "content": "",
            "answer_type": "",
            "choices": [
            <dict : choice>
                "id": 1,
                "content": "str",
                "is_correct": true
            </dict : choice>
                ]
        </dict : quizz>
    </dict : quizz list>
    ]
    </example>
"""
```

### Langchain
- 주어진 주제에 맞춰 특정 형식의 문제 출제

```py
...
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.3)
    chain = (
        RunnableMap(
            {
                "content": lambda inputs: content,
                "type": itemgetter("type"),
                "count": itemgetter("count"),
                "difficulty": itemgetter("difficulty"),
                "level": itemgetter("level"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

## DRF 모델 설계

### ERD

#### 퀴즈 모델
- `Quiz` - `Question` : **ForeignKey** 관계
- `Question` - `Choice` : **ForeignKey** 관계
- `Quiz` - `Question` - `Choice` 구조 구성

```py
class Quiz(models.Model):
    title = models.CharField(max_length=100)  # 퀴즈 제목
    description = models.TextField()  # 퀴즈들에 대한 전체적인 설명


class Question(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name="questions")
    number = models.PositiveIntegerField()
    content = models.TextField()  # 문제 설명
    answer_type = models.CharField(max_length=100)  # 문제 형태(객관식, 단답형, ox)


class Choice(models.Model):
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, related_name="choices"
    )
    number = models.PositiveIntegerField()
    content = models.TextField()  # 사용자 답변
    is_correct = models.BooleanField()  # 정답 여부
```

#### 퀴즈 결과 모델
- `QuizResult` - `QuizResultQuestion` : **ForeignKey** 관계
- `QuizResult`
  - 제목
  - 설명
  - 채점 결과
- `QuizResultQuestion`
  - 문제 내용
  - 사용자 답변

```py
class QuizResult(models.Model):
    # user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)  # 퀴즈 제목
    description = models.TextField()  # 퀴즈들에 대한 전체적인 설명
    result = models.JSONField()

class QuizResultQuestion(models.Model):
    quiz_result = models.ForeignKey(QuizResult, on_delete=models.CASCADE, related_name="result_questions")
    question = models.JSONField()
    answer = models.JSONField() 
```