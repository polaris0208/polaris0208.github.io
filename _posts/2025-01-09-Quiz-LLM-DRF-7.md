---
layout: post
title: 공식문서 RAG 챗봇 구축
subtitle: TIL Day 108
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

> 공식문서 데이터를 활용한 **RAG** 챗봇 구축

## 멀티 쿼리
- 사용자의 질문을 리트리버 검색에 적합한 형태로 변환하여 여러개의 질문 생성
  - 개념 질문
  - 활용과 관련된 질문
  - **code** 정보가 포함된 질문
    - 공식문서 데이터의 `<code_snipet>` 태그를 포함하여 검색 결과에 코드예시가 포함되도록 유도

```py
def multi_query():
    prompt = PromptTemplate.from_template(
        """
    **only in english**
    You are an expert at enhancing search results by rephrasing questions in different ways.
    Based on the question below, create 3 alternative versions of the same question. 
    first question should be related to the concept.
    second should be about application.
    third should be sample code inside <code_snipet></code_snipet>

    <question>
    {question}
    </question>
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai.api_key, temperature=0.1)
    chain = (
        RunnableMap(
            {
                "question": itemgetter("question"),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

## RAG 체인
- 제공된 자료에 근거하여 할루시네이션 방지 및 **LLM**이 학습하지 못한 정보에 대해 답변
  - 제공된 자료에서 벗어난 답변을 하는 경우 명시하도록 설정
  - 모르는 내용에 대해서는 정보를 제공할 수 없다고 답변하도록 설정
  - 되도록 코드에 대한 정보를 제공하도록 유도
- 대화 내역을 제공하여 맥락을 유지한 채 답변

### 공식문서 RAG

```py
def RAG_chain():
    prompt = PromptTemplate.from_template(
        """
    **Reply only in Korean**
    You are an official documentation Q&A chatbot.  
    1. Answer questions based on the provided official documentation.  
    2. Cite the source of the information and specify the relevant section of the documentation.  
    3. If a question is not related to the official documentation, inform the user.  
    4. For questions about unknown knowledge, respond with: "Sorry, I don't know."  
    5. Include code in your answers whenever possible.  
    6. Maintain the context of the conversation by referencing the chat history.  
    7. If you believe the current question has been addressed, ask: "Do you have any additional questions?"  
    8. If there are no further questions, proceed to the next step.
    
    <context>
    {context}
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
                "context": itemgetter("context"),
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

### 교재 RAG
- 단변적인 교재에 근거하여 답변
- 공식문서 **RAG** 보다는 유연하게 답변핟록 설정
  - 교재 내용이 아니더라도 추가정보를 제공
  - 교재에서 벗어난 경우에는 명시
- 되도록 실용적인 예시를 포함하도록 설정

```py
def QnA_chain(content):
    prompt = PromptTemplate.from_template(
        """
    **Reply only in Korean**
    You are a Q&A chatbot.
    Maintain the context of the conversation by referencing the chat history.
    For unknown information, answer by stating that it's unknown.
    Always include sources in the answer. 
    Respond based on the provided content, and if not, issue a warning.
    Include practical examples in the answer.
    For programming-related questions, include code examples.
    
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

## Views
- **RAG** 챗봇 **API**

### 조회
- 사용자 `id`와 대화 세션 `id`를 통해 대화 내용을 조회

```py
class RagChatbotView(APIView):

    def get_chat_history(self, chat_id, user):
        """ChatHistory 조회 유틸리티 함수"""
        return ChatHistory.objects.filter(id=chat_id, user=user).first()

    def generate_ids(self, chat_history):
        """새로운 응답 ID를 생성"""
        if chat_history.last_response_user:
            id_user = chat_history.last_response_user["id_user"] + 1
        else:
            id_user = 1

        if chat_history.last_response_ai:
            id_ai = chat_history.last_response_ai["id_ai"] + 1
        else:
            id_ai = 1

        return id_user, id_ai

    def get(self, request, chat_id):
        try:
            user = request.user
            chat_history = self.get_chat_history(chat_id, user)
            return Response(
                {
                    "id": chat_history.id,
                    "title": chat_history.title,
                    "content_info": chat_history.content_info,
                    "chatlog": chat_history.conversation,
                },
                status=status.HTTP_200_OK,
            )
        except ChatHistory.DoesNotExist:
            return Response(
                {"error": "ChatHistory not found"}, status=status.HTTP_404_NOT_FOUND
            )
```

### 챗봇 대화
- `chat_id` : 대화 세션 `id` 
  - 입력할 경우 : 이전의 대화 정보를 가져와 답변에 필요한 정보를 추출하고 대화를 이어감
  - 입력하지 않을 경우 : 대화 생성에 필요한 정보를 받아 대화 세션 생성
- `category` : **RAG**에 사용할 문서 종류
  - `Official-Docs` : 공식문서 선택시 벡터 **DB**에 기반한 **RAG** 챗봇으로 전환
  - 일반 교재 선택시 : 교재를 **DB**에서 호출하여 전체를 **LLM** 컨텍스트에 전달

```py
    def post(self, request, chat_id=False):
        # 사용자 정보
        user = request.user
        # ChatHistory 조회 또는 생성
        chat_history = self.get_chat_history(chat_id, user) if chat_id else None

        if chat_history:
            # 기존 채팅 기반 처리
            memory = chat_history.conversation
            category = chat_history.content_info["category"]
            user_input = request.data["user_input"]

            if category == "Official-Docs":
                title = chat_history.content_info["title"]
                retriever = rag.get_retriever(title)
                multi_query_chain = rag.multi_query()
                multi_query = multi_query_chain.invoke({"question": user_input})
                context = retriever.invoke(multi_query)
                rag_chain = rag.RAG_chain()
                response = rag_chain.invoke(
                    {"context": context, "question": user_input, "history": memory}
                )
            else:
                content = chat_history.content
                chain = llm.QnA_chain(content)
                response = chain.invoke({"history": memory, "question": user_input})
        else:
            # 새로운 ChatHistory 생성
            category = request.data["category"]
            title_no = request.data["title_no"]
            user_input = request.data["user_input"]

            if category == "Official-Docs":
                documents = Documents.objects.filter(title_no=title_no).first()
                title = documents.title
                retriever = rag.get_retriever(title)
                multi_query_chain = rag.multi_query()
                multi_query = multi_query_chain.invoke({"question": user_input})
                context = retriever.invoke(multi_query)
                rag_chain = rag.RAG_chain()
                response = rag_chain.invoke(
                    {"context": context, "question": user_input, "history": []}
                )
                content_info = {
                    "category": category,
                    "title_no": title_no,
                    "title": title,
                }
                memory = [{"SYSTEM": "init conversation"}]
            else:
                reference = Reference.objects.filter(
                    Q(category=category) & Q(title_no=title_no)
                ).distinct()
                content_list = [ref.content for ref in reference]
                chain = llm.QnA_chain(content_list)
                response = chain.invoke({"history": [], "question": user_input})
                content_info = {"category": category, "title_no": title_no}
                memory = [{"SYSTEM": "init conversation"}]

            chat_history = ChatHistory.objects.create(
                user=user,
                conversation=memory, content_info=content_info
            )
```

### 대화 세션에 기록 저장
- 각각의 질문과 답변을 `id`로 구분하여 저장
  - 사용자 입력
  - 챗봇 답변
- 마지막 질문과 답변은 별도로 저장

```py
        # 응답 ID 생성 및 대화 기록 업데이트
        id_user, id_ai = self.generate_ids(chat_history)

        last_response_user = {"id_user": id_user, "USER": user_input}
        last_response_ai = {"id_ai": id_ai, "AI": response}

        memory.extend([last_response_user, last_response_ai])

        # ChatHistory 저장
        chat_history.title = llm.summarize_title(memory)
        chat_history.conversation = memory
        chat_history.last_response_user = last_response_user
        chat_history.last_response_ai = last_response_ai
        chat_history.save()

        return Response(
            {
                "id": chat_history.id,
                "AI": response,
                # "multi_query": multi_query if category == "Official-Docs" else None,
                # "Retriever": context if category == "Official-Docs" else None,
            },
            status=status.HTTP_200_OK,
        )
```

### 대화 세션 삭제
- `id`를 입력하여 삭제
- 삭제된 `id`는 별도로 지정하지 않는 경우 다시 사용되지 않음

```py
    def delete(self, request, chat_id):
        try:
            # 사용자 정보
            user = request.user
            # 특정 사용자 ID에 해당하는 ChatHistory 삭제
            ChatHistory.objects.filter(id=chat_id, user=user).delete()
            return JsonResponse({"message": f"{chat_id} 삭제"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
```

## 전체 대화 세션 조회
- 전체 대화 세션을 조회
  - 각각의 세션에는 전체 대화 내용을 포함
  - 대화 제목, 사용된 자료에 대한 정보를 포함

```py
class ChatSessionView(APIView):

    def get(self, request):
        try:
            chats = ChatHistory.objects.filter(user=request.user).values(
                "id", "title", "content_info", "conversation"
            )
            return Response(
                {"chatsession": list(chats)},
                status=status.HTTP_200_OK,
            )
        except ChatHistory.DoesNotExist:
            return Response(
                {"error": "ChatHistory not found"}, status=status.HTTP_404_NOT_FOUND
            )
```

## 제목 생성
- 대화 내역을 요약하여 제목생성
- 대화 내역이 갱신될 때마다 새로 생성
  - 일정량 대화가 쌓이지 않으면 변동은 거의 없음

```py
def summarize_title(user_message):
    prompt = f"""
    answer in Korean, except for essential keywords
    Based on the following content, summarize the session title briefly to capture the user's intent clearly.
    {user_message}
    """

    # OpenAI API 호출
    completion = CLIENT.chat.completions.create(
        model="gpt-4o-mini",  # 사용 모델
        messages=[
            {"role": "user", "content": prompt, "type": "text"},
        ],
    )
    return completion.choices[0].message.content
```