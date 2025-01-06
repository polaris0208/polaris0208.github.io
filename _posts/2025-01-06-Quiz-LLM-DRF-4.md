---
layout: post
title: Quiz Feedback LLM & RAG 챗봇 API 제작
subtitle: TIL Day 105
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---
> **LLM**을 이용하여 퀴즈 문제별 결과 피드백 및 **RAG** 챗봇 **API** 제작

## 문제별 피드백 기능

### LLM
- 문제 내용, 선택지, 사용자의 답변을 입력 받음
- 사용자가 선택한 응답에 대한 피드백을 제공
- 틀린 문제의 경우에는 풀이법을 제공

```py
def individual_feedback_chain():
    prompt = PromptTemplate.from_template(
        """
    must use **korean**
    Create feedback about user_answer.
    include feedback about how to solve question
    **return only json**
    **remove any space
    **do not include```json```**

    <question>
    {question}
    </question>

    <choice>
    {choice}
    </choice>

    <user_answer>
    {user_answer}
    </user_answer>

    <example>
    "feedback" : ""
    </example>
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.3)
    chain = (
        RunnableMap(
            {
                "question": itemgetter("question"),
                "choice": itemgetter("choice"),
                "user_answer": itemgetter("user_answer"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### Views
- **DB**에서 문제 정보 및 사용자 퀴즈 결과를 호출
- **LLM**에 정보를 넣고 피드백을 받아 반환

```py
class IndividualFeedabckView(APIView):
    def get(self, request, quiz_id):
        try:
            chain = llm.individual_feedback_chain()
            feedback_output = []
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
                feedback = json.loads(chain.invoke(data))

                output = {}
                output["question"] = data["question"]
                output["feedback"] = feedback
                feedback_output.append(output)

            return Response(
                feedback_output,
                status=status.HTTP_200_OK,
            )
        except QuizResult.DoesNotExist:
            return Response(
                {"error": "Quiz result not found"}, status=status.HTTP_404_NOT_FOUND
            )
```

## RAG 구축
- **Json** 파일 형태로 크롤링된 데이터 사용
- **DB**에 반영하기 위해 **SQL** 파일 형태로 변환
  - `reference` 테이블 생성

```py
import json
import os
# Json 데이터 호출
with open('references.json', 'r') as json_file:
    data = json.load(json_file)

# SQL 쿼리 작성
# sample : 테이블 이름
# AUTO_INCREMENT : MySQL
# SERIAL : PostgreSQL
sql_query = """
CREATE TABLE IF NOT EXISTS reference (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50),
    title VARCHAR(250),
    content TEXT,
    title_no INTEGER
);

INSERT INTO reference (category, title, content, title_no) VALUES
"""

values = []
for entry in data:
    # 작은 따옴표 제거
    category = entry['category'].replace("'", "''")  
    title = entry['title'].replace("'", "''")  
    content = entry['content'].replace("'", "''")
    title_no = entry['title_no']
    values.append(f"('{category}', '{title}', '{content}', {title_no})")

# SQL IN
sql_query += ",\n".join(values) + ";"

# output_dir = '/'
# os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성
# with open(os.path.join(output_dir, 'dataset.sql'), 'w') as file:
#     file.write(sql_query)

with open('dataset.sql', 'w') as file:
    file.write(sql_query)
```

### RAG를 이용한 QnA 기능

#### LLM
- 학습 내용에 대한 의문 해결 및 추가 학습을 위한 기능
- 항상 학습자료를 참조하여 답변
  - 답변에 출처(참고한 부분) 첨부
- 모르는 내용은 모른다고 답변
- 대화 내역을 참조하여 맥락을 유지하면서 대화

```py
def QnA_chain(content):
    prompt = PromptTemplate.from_template(
        """
    **Reply only in Korean**
    You are a Q&A chatbot.
    Answer questions based only on the provided content.
    include the source of the information referenced and specify which part of the content it comes from.
    If a question is not related to the content, alert the user.
    For questions outside the content, reply with "Sorry, I don't know.",
    Maintain the context of the conversation by referencing the chat history.

    <context>
    {content}
    </context>

    <chat history>
    {history}
    </chat history>

    <question>
    {question}
    </question>
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.1)
    chain = (
        RunnableMap(
            {
                "content": lambda inputs: content,
                "history": itemgetter("history"),
                "question": itemgetter("question"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

#### Models
- 학습 자료 저장
  - **LLM**에게 참조 자료 제공
- 사용자 질문과 **LLM** 답변을 저장
  - **LLM**에게 대화 맥락 제공

```py
class ChatHistory(models.Model):
    key = models.CharField(max_length=100)
    conversation = models.JSONField()
    content = models.TextField(null=True, blank=True)
```

#### Views
- 학습자료에 대한 이전 대화내역이 있으면 이어서 대화
- 없다면 새로운 대화 시작
  - 학습 자료에 대한 **Langchain** 생성
  - 대화 내역 생성

```py
class RagChatbotView(APIView):
    def post(self, request, key):
        chat_history = ChatHistory.objects.filter(key=key).first()

        if chat_history:
            memory = chat_history.conversation
            content = chat_history.content
            user_input = request.data["user_input"]
            chain = rag.QnA_chain(content)

        else:
            category = request.data["category"]
            title_no = request.data["title_no"]
            user_input = request.data["user_input"]

            reference = Reference.objects.filter(
                Q(category=category) & Q(title_no=title_no)
            ).distinct()

            content_list = [ref.content for ref in reference]
            content = content_list
            chain = rag.QnA_chain(content)
            memory = []
            memory.append({"SYSTEM": "init conversation"})
            chat_history = ChatHistory.objects.create(
                key=key, conversation=memory, content=content
            )

        response = chain.invoke({"history": memory, "question": user_input})
        memory.append({"USER": user_input})
        memory.append({"AI": response})

        chat_history_obj = ChatHistory.objects.get(key=key)
        chat_history_obj.conversation = memory
        chat_history_obj.save()
        return Response(
            {"CHAT": memory},
            status=status.HTTP_200_OK,
        )
```

### 학습내용 요약 기능

#### LLM
- 학습자료를 바탕으로 학습에 필요한 내용을 정리
  - 핵심 내용/개념/용어
  - 예제/코드스니펫
  - 참고문헌
- 마크다운 형식으로 답변
  - 프론트엔드에서 **Parser**를 통해 출력 
- 사용자 정의 프롬프트를 추가할 수 있도록 입력을 받음

```py 
def summary_chain(content):

    prompt = PromptTemplate.from_template(
        """
    Yor are summary maker.
    Summarize the following content in an easy-to-understand way.  
    Format the summary in **Markdown**.  

    ### Requirements  
    1. **Reply only in Korean**
    2. **Key Concepts**: Briefly explain the main ideas.  
    3. **Important Terms**: Include key terms with simple definitions.  
    4. **Practical Applications**: Provide examples of real-world use cases.
    5. **Code Snippets**: Format all code snippets using ```code``` blocks.  
    6. **References**: Gather all links mentioned in the text and list them at the end as references.
    <user prompt>
    {user_input}
    </user prompt>

    <example>
    ## Summary  
    - **Intro** : one point summary

    ### 1. Key Concepts  
    - **Concept 1**: Explanation of the main idea.  
    - **Concept 2**: Explanation of another key point.  

    ### 2. Important Terms  
    - **Term 1**: Simple definition.  
    - **Term 2**: Simple definition.  

    ### 3. Practical Applications  
    - **Example 1**: A description of a real-world use case.  
    - **Example 2**: Another practical application.  

    ### 4. Conclusion 
    - **Outro** : Additional topics to learn

    ### References
    - Source 1
    - Source 2
    - Source 3
    </example>

    <context>
    {content}
    </context> 
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.1)
    chain = (
        RunnableMap(
            {
                "content": lambda inputs: content,
                "user_input": itemgetter("user_input"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

#### Views
- 학습자료를 **ORM**을 통해 검색 후 **LLM**에게 전달
- 답변을 응답에 담아 전달

```py 
class SummaryView(APIView):
    def get(self, request):
        category = request.data["category"]
        title_no = request.data["title_no"]
        user_input = request.data["user_input"]
        reference = Reference.objects.filter(
            Q(category=category) & Q(title_no=title_no)
        ).distinct()
        content_list = [ref.content for ref in reference]
        content = content_list
        chain = rag.summary_chain(content)
        response = chain.invoke({"user_input": user_input})

        return Response(
            {"result": response},
            status=status.HTTP_200_OK,
        )
```