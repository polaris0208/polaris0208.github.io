---
layout: post
title: AI 활용 개념과 Hugging Face 
subtitle: TIL Day 38
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---
# AI 활용 개념과 Hugging Face
> [¶ AI 활용 개념](#ai-활용-개념)<br>
> [¶ Hugging Face](#hugging-face)

# AI 활용 개념

## 연구 vs 활용
### 연구: AI의 성능 향상이 목적
- 모델 개선 또는 개발
- 수학적으로 구조적으로 복잡한 내용

### 활용 : 이미 만들어진 모델 & 서비스 이용
- 모델을 개발할 필요없이 바로 이용
- 다양한 모델을 결합하여 사용 가능
- 의미있게 활용하기 위한 이해가 필수적

## API 개념
>**Application Programming Interface** 

- 프로그램 끼리 통신하는 방식
- AI 서비스가 제공하는 프로그램과 자신의 프로그램을 연결하는 개념
- **ChatGPT, ElevenLabs** 등

## 사전학습 모델
> **Pre Trained Model**

- 많은 학습 데이터로 사전 학습된 모델
- 다양한 모델과 결합 가능
- 검증이 끝난 모델로 안정성이 높음
- 직접 개발 할 경우 많은 데이터가 필요: 모델이 무겁고 불안정

## AI 활용의 주의점
- 기존 모델을 활용할 경우 중요한 것은 **Fine Tuning**
- AI에 대한 이해가 부족할 경우 문제점 발생
  - 성능을 이끌어 내지 못하는 경우
  - 사용 중 발생하는 문제에 대처 불가능
  - 수많은 모델 중에서 적합한 모델 선별의 문제
  - 결과를 해석하지 못하는 문제

[¶ Top](#ai-활용-개념과-hugging-face)

# Hugging Face 
> AI 모델 제공 및 다양한 기능 제공 [¶](https://huggingface.co) <br>
> 커뮤니티가 크게 발달

## 개요
- 최신 모델. 검증된 모델 제공하는 허브역할
- 개발, 학습 과정 생략 가능
- 오픈 소스 커뮤니티에서 수많은 개발자가 함께 개발한 모델 사용 가능
- 다양한 개발자와 의견 교환 가능

### 장점
- 쉬운 접근성 : 쉽고 직관적, 이해하기 쉬운 예시
- 광범위한 모델선택 : 다양한 언어
- 강력한 커뮤니티 : 적극적인 소통 가능 

### 단점
- 리소스 요구량
- 복잡한 초기 설정
- 특화된 모델 : **NLP**

[¶ Top](#ai-활용-개념과-hugging-face)
