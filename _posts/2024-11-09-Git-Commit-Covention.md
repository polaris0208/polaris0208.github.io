---
layout: post
title: Git Commit Convention
subtitle: TIL Day 48
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

## Git Commit
- `commit` : 변경사항을 저장
- **Commit Message** : 변경사항을 구분하기 위한 메시지

### 필요성
- 과거의 수정사항을 이해하기 쉽게 만듬
- 좋은 커밋 메세지는 협업과 유지 보수에 필수적

### 기본 기능
- `-m`: 메세지를 설정
- `-a -m`, `-all -m`: 모든 파일을 자동으로 커밋
- `--amend -m`: 마지막 커밋을 새 메시지로 수정

### Editor
- `git config --global core.editor` : 현재 에디터 확인
- `git config --global core.editor "vim"` : **zshrc** 에디터 **vim** 으로 설정
- `git commit` : 에디터 열기
  - **vim** 작성법으로 **commit** 작성 가능

### Command Line
- `git commit -m "Subject" -m "Description"`
- `-m "Subject"` : 제목 입력
- `-m "Description"` : 부연 설명 입력

## Git Commit Convention
- 정해진 규칙이나 관습을 적용
- 누구든지 빠르게 변경사항을 이해할 수 있게 작성

### 구조
- `Type : Subject` : 수정 종류 : 제목
  - `feat` : 새로운 기능 추가
  - `fix` : 버그 수정
  - `hotfix` : 치명적인 버그 수정
  - `docs` : 문서 추가, 수정
  - `style` : 코드 formatting, 세미콜론(;) 누락, 코드 변경이 없는 경우
  - `refactor` : 코드 구조 변경
  - `test` : 테스트 코드
  - `chore` : 자잘한 수정, 패키지 매니저 등
- `Subject`
  - 50자 미만, 마침표를 붙이지 않음
  - 명령조로 작성
  - 제목과 본문은 한 줄 띄움
  - 제목의 첫 글자는 대문자로 작성
  - 제목이나 본문에 이슈 번호를 작성
- `Body` : 본문
  - 한 줄에 72자를 넘기지 않음
  - **How**보다 **What, Why**
  - 커밋 작성의 이유를 설명
- `Footer` : 부연 설명
  - 선택사항
  - `Closes`, `Fixes`, `Resolves`, `Ref`, `Related to`, `Issue tracker ID`

## 실습
- `git add .`
- `git commit`
- `i` - 내용 작성 - `esc` - `:` - `wq` - `enter` 

```
docs : Add TIL_24_1109

2024년 11월 9일 TIL Markdown 문서 작성
Git Commit Convention에 관해 학습한 내용을 기록
```
