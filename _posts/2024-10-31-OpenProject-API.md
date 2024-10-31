---
layout: post
title: Open Project 와 API 개념
subtitle: TIL Day 39
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

# OpenProject & API
> 라이센스에 주의하며 사용 <br>
> 출처 명시 조건, 코드 변경 조건 등<br>
> [¶ OpneProject](#opneproject)<br>
> [¶ API](#api)<br>

# OpneProject
> 자유롭게 활용하고 기여할 수 있는 프로젝트

## 종류

### **DeepArt - AI 그림** [¶](https://creativitywith.ai/deepartio/)

- 이미지를 예술 작품처럼 변환
  - 활용: 사진을 Monet, Van Gogh 같은 스타일로 변환

### **OpenAI Gym - 강화학습 게임** [¶](https://gymnasium.farama.org)

- 강화학습(적응형 AI 학습)의 연구와 개발을 위한 도구들
  - 활용: 간단한 게임을 만들고, AI가 스스로 게임을 배우고 플레이하도록 학습

### **Mozilla Common Voice - 음성 인식 데이터셋 구축** [¶](https://commonvoice.mozilla.org/ko)

- **Mozilla** 제공
- **AI** 음성 인식을 위한 방대한 데이터셋을 구축
  - 활용: **GitHub**에서 프로젝트를 클론하고, 자신만의 음성 인식 모델을 훈련시켜 보세요. 나만의 음성 비서도 만들어 볼 수 있답니다.

### **Scikit-learn - 머신러닝 라이브러리** [¶](https://scikit-learn.org/stable/)

- 파이썬 기반의 머신러닝 라이브러리로, 다양한 머신러닝 알고리즘을 손쉽게 구현
  - 활용: 고객 데이터 분석, 예측 모델 만들기 등

### **Hugging Face Transformers - 자연어 처리 프로젝트** [¶](https://huggingface.co/docs/transformers/index)

- `BERT`, `GPT-3` 등 최신 `NLP` 모델들을 활용
  - 활용: 텍스트 생성, 번역, 감정 분석 등

### **Magenta - 음악과 예술 창작 AI** [¶](https://magenta.tensorflow.org)
- 허깅페이스는 오픈소스 커뮤니티를 중심
  - 활용: 음악 작곡 AI를 만들거나, 기존 음악에 새로운 스타일 적용

[¶ Top](#openproject--api)

# API
> **Application Programming Interface**<br>
> 프로그램 간에 데이터를 주고받을 수 있게 해주는 인터페이스
>> 서로 다른 프로그램의 대화 통로

## 기본 개념
- 이미 만들어진 다양한 서비스를 활용하여 자신만의 서비스 개발 가능
- 고성능의 모델은 비용이 발생
  - **API Key**를 발급 받아 사용 
  - **Key** 관리가 매우 중요 : 공개될 경우 동시다발적 과금 발생 가능

### 장점
- 손쉬운 사용: 간단한 호출로 사용
- 신속한 개발: 빠르게 프로토 타입 개발 후 새로운 기능을 추가
- 확장성: 다양한 **API** 결합 가능

### 단점
- 비용: 고성능 일수록 고비용
- 제한된 제어: 제공된 기능만 사용 가능, 커스터마이징 제한 가능성
- 의존성: 해당 서비스 중단되면 문제 발생

### Tips
- 문서 읽기: 공식 문서를 자세히 읽어 사용법과 제한 사항 확인
- **API** 키 관리: 신중하게 관리
  - 코드에 포함시키지 않고 환경 변수를 통해 관리
- 무료 할당량 체크: 잘 활용하면 비용없이 충분한 테스트 가능

## 공식 문서 활용법

### **Endpoint**
- **API**가 제공하는 서비스의 주소
- url 형식으로 존재
- 특정 리소스 기능

### **Method**
- **HTTP Method** : **API**의 요청에 의해 서버에서 수행해야 할 동작의 종류
  - **Get** : 리소스 조회
  - **Post**: 데이터 추가, 등록
  - **Delete**
  - **Patch** : 리소스 부분 변경
  - **Head** : **Body** 부분 제외하고 조회
  - **Option** : 서버가 브라우저와 통신하기 위한 통신 옵션
    - **method, header, content-typ** 등

### **Request**
- 클라이언트가 서버에 보내는 메시지
  - 어떤 작업을 수행할지, 어떤 데이터를 사용할지 정하는 것
  - **Url**: 요청이 전달 될 엔드포인트 주소
  - **Header**: 요청에 대한 메타 데이터(인증 데이터 등)
  - **Body**: 서버롤 전달할 데이터

### **Response**
- 서버가 클라이언트 요청에 대해 반환하는 데이터
  - 상태코드: 처리 결과
  - **Header**: 응답에 대한 메타 데이터(데이터 형식 등)
  - **Body**: 요청에 대한 실제 데이터

### 데이타 포맷
- **Jason, XML, YAML** 사용
  - **XML(eXtensible Markup Language)**
    - W3C에서 개발
    - 다른 특수한 목적을 갖는 마크업 언어를 만드는데 사용하도록 권장하는 다목적 마크업 언어
  - **YAML(Yet Another Markup Language)** [¶](https://www.ibm.com/kr-ko/topics/yaml)
    - 사람이 쉽게 이해할 수 있고 기계가 해석할 수 있는 방식으로 구조화된 데이터를 표현하는 표준화 형식을 제공

### Authentication
- 인증과 관련된 내용들
- **API Keyes**

## 문제점
>사전학습된 모델은 특정 데이터나 작업에 대해 학습된 상태
>>다른 작업에 맞추려면 **Fine-Tuning** 필요
>>> 비용 문제 : 미세 조정이나 추가 학습을 위한 클라우드 서비스나 고성능 장비에 비용이 요구됨

## 극복 방법
- 무료 또는 저비용 클라우드 서비스: **Google Colab [¶](https://colab.research.google.com/?hl=ko), AWS [¶](https://aws.amazon.com/ko/)**
- 사전학습된 모델: **Hugging Face-Transformers [¶](https://huggingface.co/docs/transformers/index), PyTorch Hub [¶](https://pytorch.org/hub/)** 
  - 일부 파라미터만 조정하여 사용 가능
- 경량화 모델: **DistilBERT [¶](https://huggingface.co/docs/transformers/model_doc/distilbert), TinyBERT**

[¶ Top](#openproject--api)