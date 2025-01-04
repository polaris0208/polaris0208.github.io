---
layout: post
title: Quiz 생성 LLM API 제작
subtitle: TIL Day 103
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

> **LLM**을 이용하여 퀴즈를 생성하는 **API** 제작


## 의존성

```py
import json
from . import llm
from .models import *
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
```

## 퀴즈 생성 API
- **Post** 요청으로 퀴즈에 대한 설정값을 보내면 **LLM**으로 부터 결과를 받음
  - 퀴즈 개요 - 문제 - 선택지 구조로 나눠 **DB**에 저장

```py
class QuizRequestView(APIView):
    def post(self, request):
        response = chain.invoke(request.data)
        response_dict = json.loads(response)
        quiz = Quiz.objects.create(
            title=response_dict["title"],
            description=response_dict["description"],
        )
        for question_data in response_dict["questions"]:
            question = Question.objects.create(
                quiz=quiz,
                number=question_data["id"],
                content=question_data["content"],
                answer_type=question_data["answer_type"],
            )
            for choice_data in question_data["choices"]:
                Choice.objects.create(
                    question=question,
                    number=choice_data["id"],
                    content=choice_data["content"],
                    is_correct=choice_data["is_correct"],
                )
        return Response(
            {"detail": "문제가 생성되었습니다."},
            status=status.HTTP_200_OK,
        )
```

## 퀴즈 API
- `class QuizAPIView(APIView):`

### 조회
- 생성된 퀴즈를 조회

```py
    def get(self, request, quiz_id):
        try:
            quiz = Quiz.objects.prefetch_related("questions__choices").get(id=quiz_id)
            questions = [
                {
                    "number": question.number,
                    "content": question.content,
                    "answer_type": question.answer_type,
                    "choices": [
                        {"number": choice.number, "content": choice.content}
                        for choice in question.choices.all()
                    ],
                }
                for question in quiz.questions.all()
            ]
            return Response(
                {
                    "id": quiz.id,
                    "title": quiz.title,
                    "questions": questions,
                },
                status=status.HTTP_200_OK,
            )
        except Quiz.DoesNotExist:
            return Response(
                {"error": "Quiz not found"}, status=status.HTTP_404_NOT_FOUND
            )
```

### 답안 제출/채점
- 답안 제출
  - 문제 번호 - 선택지 번호 제출
- 채점
  - 선택지 데이터의 `is_correct` 옵션을 확인하여 채점
- 결과
  - 전체 문제 중 맞힌 문제의 개수를 계산하여 점수 출력


```py
    def post(self, request, quiz_id):
        try:
            quiz = Quiz.objects.get(id=quiz_id)
            answers = request.data.get("answers", [])
            total_questions = quiz.questions.count()
            correct_answers = 0
            details = []

            for answer in answers:
                question_number = answer.get("q_number")
                choice_number = answer.get("c_number")
                question = Question.objects.get(number=question_number, quiz=quiz)
                selected_choice = Choice.objects.get(
                    number=choice_number, question=question
                )

                is_correct = selected_choice.is_correct
                if is_correct:
                    correct_answers += 1

                details.append(
                    {"question_number": question_number, "is_correct": is_correct}
                )
            score = (correct_answers / total_questions) * 100
            report = {
                "quiz_id": quiz.id,
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "score": score,
                "details": details,
            }

            quiz_result = QuizResult.objects.create(
                title=quiz.title,
                description=quiz.description,
                result=report.pop("details", None),
            )
```

### 결과 저장
- 채점 결과 / 문제 내용-제출한 답안 저장


```py
            for answer in answers:
                question_number = answer.get("q_number")
                choice_number = answer.get("c_number")
                question = Question.objects.get(number=question_number, quiz=quiz)
                selected_choice = Choice.objects.get(
                    number=choice_number, question=question
                )
                questions_json = {
                    "id": question.id,
                    "number": question.number,
                    "content": question.content,
                    "answer_type": question.answer_type,
                    "quiz": question.quiz_id,
                }
                choice_json = {
                    "id": selected_choice.id,
                    "number": selected_choice.number,
                    "content": selected_choice.content,
                    "is_correct": selected_choice.is_correct,
                    "question": selected_choice.question_id,
                }
                QuizResultQuestion.objects.create(
                    quiz_result=quiz_result,
                    question=questions_json,
                    answer=choice_json,
                )
                
            return Response(
                report,
                status=status.HTTP_200_OK,
            )
        except (Quiz.DoesNotExist, Question.DoesNotExist, Choice.DoesNotExist):
            return Response(
                {"error": "Invalid data provided"}, status=status.HTTP_400_BAD_REQUEST
            )
```

### 퀴즈 삭제
- 결과만 저장하고 **DB**부담을 줄이기 위해 퀴즈는 삭제

```py
    def delete(self, request, quiz_id):
        quiz = Quiz.objects.get(id=quiz_id)
        quiz.delete()
        return Response(
            {"detail": "삭제되었습니다."},
            status=status.HTTP_204_NO_CONTENT,
        )
```