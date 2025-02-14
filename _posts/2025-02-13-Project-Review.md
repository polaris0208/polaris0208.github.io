---
layout: post
title: 프로젝트 경험 정리
subtitle: TIL Day 143
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL]
author: polaris0208
---

### 1. ReadRiddle (최종 프로젝트)

- 프로젝트 소개 : AI 서비스 개발 학습자를 위한 퀴즈 위주 자기주도 학습 서비스
- 사용 스킬 : Python, Django, OpenAPI, Docker, OAuth, Git, Notion
- 소제목 입력하는 곳
    - Docker 컨테이너 환경 구성
        - 개발환경 통일
        - Django - React - PostgreSQL - NGINX - Redis 통합 운용
    - Swagger UI 구성
        - DRF-Spectacular 를 이용한 OpenAPI 3.0 활용
        - API 문서 자동화
    - RAG 구축
        - 공식문서 자료 기반 FAISS VectorDB 생성
        - 앙상블 리트리버 구축
            - FAISS 70%
            - BM25 30%
        - 멀티 쿼리
            - 개념 / 활용 / 코드스니펫 3가지 유형 검색 실행
    - 퀴즈 생성 API 구현
        - 퀴즈 형식과 주제를 입력 받아 퀴즈 생성
            - RAG 를 통해 주제에 맞는 레퍼런스 검색
            - 레퍼런스를 기반으로 형식에 맞는 퀴즈 생성
        - OpenAI GPT-4o
            - “Structured Outputs” 기능을 활용한 Json 형태 응답 출력
                - Pydatic을 이용한 스키마 구조 정의 후 답변 생성 문법으로 제공
        - Redis를 이용한 캐시워밍
            - 1일마다 캐시 업데이트
            - 레퍼런스 검색 시간 단축
    - QnA 챗봇 API 구현
        - 주제에 맞는 QnA 답변 생성
            - RAG를 통해 주제에 맞는 레퍼런스 검색
            - 프롬프트를 통해 레퍼런스를 근거로만 답변하도록 설정
        - 대화 내역을 저장
            - 맥락 유지
            - 세션별 관리
        - 풀링 방식
            - 대화 내역 업데이트 실시간 반영
        - Redis를 이용한 캐싱
            - 대화내역 캐싱 및 우선 조회
            - 응답 속도 30% 단축
        - 응답 상태 관리
            - 비동기로 응답 시간 측정
            - 경과 시간에 따른 상태 변경
                - 10초 : 지연
                - 30초 : 실패 처리
    - 단체 퀴즈 생성 LLM 구현
        - 무작위 주제를 바탕으로 일정시간마다 퀴즈 생성
            - 주제 생성 후 캐시에 저장
            - 퀴즈 생성 요청을 받으면 주제를 조회하여 사용되지 않은 주제를 선택하여 퀴즈 생성
        - 프롬프트
            - few-shot 기법
                - 긍정적 예시 / 부정적 예시 제공
    - JWT 토큰 재발급 API 구현
        - Access 토큰을 디코딩 하여 사용자 인증 후 DB의 Refresh토큰을 이용하여 재발급
    - 소셜 로그인 API 구현
        - OAuth를 이용한 Google 인증 방식 구현

---

### 2.  스토리텔링 챗봇 (그 외 프로젝트)

- 프로젝트 소개 : 사용자와 상호작용하며 이야기를 전달해주는 챗봇
- 사용 스킬 : Python, Streamlit, Langchain, Git
- 소제목 입력하는 곳
    - LLM 입력 토큰을 초과하는 레퍼런스 데이터 필터링 LLM 구현
        - 데이터 분할 후 청크 단위로 필터링
        - Langchain의 ConversationSummaryMemory를 통해 전체 내용의 맥락 유지
    - 이야기 탐색 및 검색 기능 구현
        - CoT 기법을 이용하여 검색 결과 검증 LLM 구현
    - 챗봇 LLM 구현
        - 멀티 에이전트 구성
            - 스크립트 생성 LLM
                - 레퍼런스를 바탕으로 이야기 LLM이 사용할 스크립트 생성
                - Act as/Persona 기법, One-Shot 기법 활용
            - 이야기 LLM
                - 스크립트의 구성에 따라 사용자와 상호작용하며 이야기 전달
                - Act as/Persona 기법 활용
    - 사용자 입력 레퍼런스 생성 기능 구현
        - Langchain의 WebBaseLoader를 통해 레퍼런스 데이터 추출
        - 필터링 LLM을 통해 전처리 후 챗봇에 전달
    - 다국어 기능 구현
        - Meta의 nllb-200를 통해 챗봇의 입출력 번역
    - 텍스트 음성출력 기능 구현
        - Elevenlabs API 의 다국어 모델을 활용하여 챗봇의 답변을 음성으로 출력
    - Streamlit UI 구성