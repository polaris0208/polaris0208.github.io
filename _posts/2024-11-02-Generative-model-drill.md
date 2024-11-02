---
layout: post
title: Generatvie Model
subtitle: TIL Day 41
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, DL]
author: polaris0208
---

# Generatvie Model
> 텍스트 생성, 이미지 생성, 음악 작곡 등 다양한 창의적 작업 수행<br>
>[¶ 생성형 모델 개념](#생성형-모델-개념)<br>
>[¶ 생성형 모델 활용](#생성형-모델-활용)

# 생성형 모델 개념
> 주어진 입력에 따라 새로운 콘텐츠를 생성하는 인공지능

## 종류

### 이미지
- **DALL-E** [¶](https://openai.com/index/dall-e-3/)
- **Midjourney** [¶](https://www.midjourney.com/home)
- **Stable Diffusion web UI** [¶](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
### 동영상
- **Gen-3** [¶](https://runwayml.com/research/introducing-gen-3-alpha)
- **PixVerse** [¶](https://runwayml.com/research/introducing-gen-3-alpha)
- **Sora** [¶](https://openai.com/index/sora/)
### 텍스트
- **GPT** [¶](https://openai.com/index/gpt-4/)
- **Claude** [¶](https://claude.ai/login?returnTo=%2F%3F)
- **Gemnini** [¶](https://gemini.google.com/?hl=ko)
### 음악
- **Suno** [¶](https://suno.com)
- **Udio** [¶](https://www.udio.com)


## 생성형 모델 제작의 어려움

### 대규모 데이터와 컴퓨팅 자원 필요
- 일반적으로 딥러닝 기법을 활용, 수십억 대의 파라미터를 가진 모델을 학습

#### 데이터 수집의 어려움: 수백만 개의 고품질 데이터 필요
- 편향된 데이터, 윤리적으로 문제가 되는 데이터가 포함되면 결과에도 영향
- 텍스트: 방대한 양의 텍스트
- 이미지: 수많은 이미지와 그에 대한 설명

#### 컴퓨팅 자원의 한계
- **GPU, TPU**와 같은 고성능 하드웨어에서 오랜시간 학습
  - **GPU(Graphic Processing Unit)** [¶](https://www.ibm.com/kr-ko/topics/gpu) : 컴퓨터 **그래픽 및 이미지 처리** 속도를 높이도록 설계된 전자 회로
  - **TPU(Tensor Processing Unit)** [¶](https://cloud.google.com/tpu?hl=ko) : 대규모 AI 모델의 학습과 추론에 최적화된 커스텀 설계된 **AI 가속기**
- 클라우드 서비스를 이용할 경우 상당한 비용 발생
  - **Cloud Service** [¶](https://cloud.google.com/learn/what-is-cloud-computing?hl=ko) : **컴퓨팅 리소스(스토리지 및 인프라)를 인터넷을 통해** 사용할 수 있는 주문형 서비스

#### 모델 구조의 복잡성
- 모델 아케텍처 설계: **Attention** 매커니즘. **transformer** 구조 등 다양한 매커니즘을 올바르게 사용하기 어려움
- 특정 도메인을 위한 도메인 knowledge 코드
  - **Domain Knowledge [¶](https://ko.wikipedia.org/wiki/도메인_지식#지식_포착)**대상 시스템을 운영하는 환경에 관한 지식
- 하이퍼 파라미터 튜닝: 많은 실험과 경험이 필요

#### 훈련과정의 불안정성
- 모델 붕괴 **model collapse**: 무작위하게 출력이 고정되거나 의미없는 결과를 생성
- 균형 잡힌 학습: 다양한 출력을 생성하도록 섬세한 조절이 필요

## 파인 튜닝 Fine-tuning
- 사전 학습된 모델을 특정 작업에 맞게 추가로 학습시키는 과정

### 필요성
- 생성형 AI의 복잡성과 훈련의 어려움 때문에 **Fine-tuning** 이 매우 중요

### 필수성
- 특정 작업에 최적화하기 위해서는 추가적인 학습 필요
- **도메인 특화**: 일반적인 텍스트 데이터로 학습된 모델을 특정 분야의 데이터로 파인 튜닝하여 용어와 패턴에 맞게 모델을 조정
- **작업 맞춤**: 특정 작업(에 모델을 맞추기 위해 파인 튜닝

## 생성형 AI 제작 
- 사전학습된 모델 활용
- 클라우드 서비스 활용
- 작은 프로젝트부터 시작하기
  - 작은 데이터셋과 간다한 모델로 실험을 시작
  - 점진적으로 복잡한 모델로 확장

## 생성형 AI의 원리

### 랜덤성
- 출력 데이터를 생성할 때, 일정한 확률에 따라 다양한 선택지를 고려
- 확률 분포: 학습 데이터를 통해얻은 확률 분포를 기반으로 새로운 데이터 생성

### 조건성
- 조건 입력: 입력된 조건에 따라 결과를 다르게 생성
- 조건성의 중요성: 사용자가 원하는 특정 스타일, 주제, 분위기에 맞춰 출력 데이터 생성

### 텍스트 기반 생성형 모델의 원리
- 입력 토큰화
- 확률 예측: 다음에 올 단어의 확률을 예측
- 랜덤 선택: 예측된 확률 뷴포에서 랜덤하게 선택, 랜덤성 조절 가능
- 반복 생성: 문장이 완성될 때까지 반복

### 이미지 기반 생성형 모델의 원리
- 텍스트 인코딩: 입력된 텍스트 조건을 벡토로 인코딩하여 모델에 입력
- 이미지 생성: 입력으로 주어진 조건에 맞게 이미지의 주요 특징 생성
- 세부 사항 추가: 랜덤성을 적용하여 세부적인 이미지 요소 생성, 합성하여 최종 이미지 생성

### 오디오 기반 생성형 모델의 원리
- 텍스트 또는 멜로디 인코딩
- 오디오 생성: 입력을 바탕으로 오디오 신호 생성
- 랜덤성 적용: 랜덤성을 통해 음성의 미세한 변화 추가, 다양한 오디오 생성

### 랜덤성과 조건성의 상호작용
- 랜덤성: 결과의 세부적인 변화 생성
- 조건성: 출력의 전반적인 틀과 스타일 결정
- 상호작용 결과: 창의적이고 예측 불가능한 결과 생성

# 생성형 모델 활용
- 고성능 모델의 경우 **API** 발급 필요

## GPT-2

```py
# 텍스트 생성
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

generated_text = generator('Once upond a time', max_length=50, num_return_sequences=1)

print(generated_text[0]['generated_text'])
```

>once upond a time, he will not want to wait any longer to take over the world, and the two may become a great couple. However, once he moves on to some other path, he'll inevitably die. He'll also have to

## GPT-4o 
-**API** 발급 필요 [¶](https://www.magicaiprompts.com/docs/gpt-chatbot/openai-api-usage-guide/)

[¶ Top](#generatvie-model)