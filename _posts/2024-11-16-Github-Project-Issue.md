---
layout: post
title: GitHub Projects and Issues
subtitle: TIL Day 55
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

> **Issue**와 **Project**를 활용해 작업을 체계적으로 관리, 보드 형태로 시각화하여 진행도를 쉽게 파악 가능

## Project
- 작업의 진행도를 한눈에 파악
- 각각의 이슈들을 작업, **task**로 나타내 관리
- 주로 보드형태로 사용
- 다양한 테이블 형태도 적용 가능

### 생성 방법
1. **GitHub Home -> Projects -> New Project** 선택
2. **Board** 형태로 선택, 생성
3. 생성된 보드에 이슈를 할당하고 진행도 관리

## Issue
- 프로젝트의 기획, 작업, 개선사항, 버그 수정 등을 기록하고 관리하는 기능

### 생성 및 할당 방법
1. **Repository -> Projects -> Add project** 선택
2. **Issue -> New issue**로 이슈 등록

### 작성

#### 템플릿
- 제목 : 이슈를 구분할 수 있는 제목 작성
- 내용 : 마크다운 형식으로 작성
- 템플릿 설정
  - **Settinhs - General - Features - Set up templates**
  - 이슈 종류에 맞춰 사전에 템플릿 틀 제작 및 등록 가능

#### 태그
- 담장자 : 담당자를 지정
- 레이블 : 이슈의 종류, `bug`, `docs`, `enhancement` 등
- 프로젝트 : 이슈가 할당 될 프로젝트, 할당될 프로젝트의 단계 지정
- 마일드스톤 : 관련된 마감 기한이나 참고사항 지정
- 개발 : 해당 이슈에 대해서 `baranch` 를 생성 및 `pull request` 등의 개발 과정 관리

#### 상태 변경 및 관리
- 이슈 상태를 No Status에서 In Progress, Done으로 변경 가능
- 드래그 앤 드랍으로 간단한 상태 변경 가능

## 마무리
- **GitHub Issues**와 **Projects**를 통해 현재 작업 중인 내용 관리 시작
- 앞으로도 계속 사용하며 기능을 익히고자 함
- ![LLM-RAG Chatbot 제작 진행 과정](https://drive.google.com/thumbnail?id=1mVUOUC2Y9zzAOfWabB2rV4yBqcZG3x1i)