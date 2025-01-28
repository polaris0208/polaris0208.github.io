# LLM 메시지 요청에 대한 세분화
- 비동기 함수로 LLM 응답 생성 시간 측정 하여 상태 변경 및 실패처리

## 응답 생성 상태 업데이트 함수
- 응답 생성 개시 ~ 10초 : 응답 생성중
- 10 초 경과 : 응답 지연
- 30초 경과 : 실패 처리

```py
  def update_status(self, chat_history, start_time):
    while True:
      elapsed_time = (now() - start_time).total_seconds()
      if chat_history.status == "completed":
        break
      elif elapsed_time > 30:
        chat_history.status = "failed"
        chat_history.save()
        break
      elif elapsed_time > 10:
        chat_history.status = "delayed"
        chat_history.save()
      else:
        chat_history.status = "generating"
        chat_history.save()

      time.sleep(5)

    chat_history.save()
```

## `views.py`

- 시작 시간 설정
- 상태변경 함수 비동기로 실행
- LLM 응답 생성 비동기로 실행 - 타임아웃 30초 설정
  - 30초 이후에 응답이 없으면 타임아웃 에러(408)

```py
      ...
      chat_history.status = "generating"
      chat_history.save()

      start_time = now()
      threading.Thread(
        target=self.update_status, args=(chat_history, start_time)
      ).start()

      # 타임아웃을 설정하여 응답 생성
      with ThreadPoolExecutor() as executor:
        future = executor.submit(
          self.generate_response,
          request,
          chat_history,
          user_input,
          category,
          title_no,
          start_time,
        )
        try:
          result = future.result(timeout=30) # 30초 타임아웃 설정
        except TimeoutError:
          chat_history.status = "failed"
          chat_history.save()
          return Response(
            {"error": "응답 생성이 30초를 초과하여 실패했습니다."},
            status=status.HTTP_504_GATEWAY_TIMEOUT,
          )
        ...
```