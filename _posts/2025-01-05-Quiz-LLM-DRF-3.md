---
layout: post
title: Quiz 생성 Feedback LLM API 제작
subtitle: TIL Day 104
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---
> **LLM**을 이용하여 퀴즈 결과에 대해 피드백하는 **API** 제작

## LLM
- 퀴즈 성적에 대한 피드백 부여
- 퀴즈 문제 및 사용자 답변을 분석하여 피드백 부여
  - 어떤 내용을 알고 모르는지 분석 후 결과 제공

### Prompt

```py
    """
    must use **korean**
    Create feedback about user result.
    include feedback about score, result
    **return only json**
    **remove any space
    **do not include```json```**

    <quiz title>
    {title}
    </quiz title>

    <quiz description>
    {description}
    </quiz description>

    <result>
    {result}
    </result>

    <questions>
    {questions}
    </questions>

    <example>
    "total_feedback" : ""
    </example>
    """
```

### Langchain
- 퀴즈 결과에 대한 내용을 담은 **json** 파일을 입력값으로 받음

```py
def total_feedback_chain():
    prompt = PromptTemplate.from_template(
    ...
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.3)
    chain = (
        RunnableMap(
            {
                "title": itemgetter("title"),
                "description": itemgetter("description"),
                "result": itemgetter("result"),
                "questions": itemgetter("questions"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

## API
- **DB**에서 퀴즈 결과(채점결과) 호출
- 문제 및 사용자 답변 호출
- 결과를 **LLM**에 전달하고 답변을 **API** 응답에 전달

```py
class TotalFeedabckView(APIView):
    def get(self, request, quiz_id):
        try:
            feedback_input = {}
            questions = []
            quiz_result = QuizResult.objects.prefetch_related("result_questions").get(
                id=quiz_id
            )
            result_questions = quiz_result.result_questions.all()

            for result_question in result_questions:
                data = {
                    "question": result_question.question,
                    "choice": result_question.choice,
                    "user_answer": result_question.answer,
                }
                questions.append(data)

            feedback_input["title"] = quiz_result.title
            feedback_input["description"] = quiz_result.description
            feedback_input["result"] = quiz_result.result
            feedback_input["questions"] = questions
            
            chain = llm.total_feedback_chain()
            feedback = json.loads(chain.invoke(feedback_input))

            return Response(
                feedback,
                status=status.HTTP_200_OK,
            )
        except QuizResult.DoesNotExist:
            return Response(
                {"error": "Quiz result not found"}, status=status.HTTP_404_NOT_FOUND
            )
```