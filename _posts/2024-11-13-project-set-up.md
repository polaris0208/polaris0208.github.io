---
layout: post
title: 프로젝트 개발환경 준비
subtitle: TIL Day 52
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

> 프로젝트 진행을 위한 개발환경 설정
> 지금까지 학습한 각각의 도구들을 활용하여 진행

# Index
>[¶ 가상환경 설정](#가상환경-설정)<br>
>[¶ .gitignore 작성](#gitignore-작성)<br>
>[¶ Repository 생성 및 연결](#repository-생성-및-연결)<br>
>[¶ branch 설정](#branch-설정)<br>
>[¶ python 개발환경 설정](#python-개발환경-설정)<br>
>[¶ 환경변수 설정](#환경변수-설정)

# 가상환경 설정

## 가상환경 생성
- 독립적인 **Python** 환경을 만들어 버전, 패키지, 모듈, 라이브러리 간의 충돌을 방지
- `python -m venv 가상환경이름`
  - `python`으로 실행되지 않으면 `python3` 로 시도
  - `-m venv` : **virtual environment** 가상환경 생성 
  - `가상환경이름` : 프로젝트명이나 관련있는 이름으로 설정

## 가상환경 활성화 
- `source 가상환경이름/bin/activate` : 활성화
- `deactivate` : 종료
- **VSCode** : 해당 가상환경 폴더에서 실행하면 자동으로 활성화

[¶ Top](#index)
><br>

# .gitignore 작성

## 작성 내용
- 가상환경으로 작업 중인 폴더를 버전관리 할 경우 제외할 파일의 목록
  - **.env** 등 민감 정보를 담고 있는 파일
  - 가상환경 관련 파일
  - 각종 로그 파일
  - 테스트 파일

## 작성
- **gitignore.io** [¶](https://github.com/toptal/gitignore.io)
  - 제외할 파일유형을 체크하여 자동 생성
- **github** 제공 [¶](https://github.com/github/gitignore)
  - 필요한 유형 다운로드 후 편집해서 사용
- 문법에 따라 작성 [¶](https://github.com/polaris0208/TIL/blob/main/Tools/TIL_24_1106_dotenv_gitignore.md#문법)


## 적용 및 확인
- `git init` : 프로젝트 폴더 버전관리 시작
- 파일 작성 : `Readme.md`
- `git add .` : 전체 파일 등록
- `git status` : 다른 파일은 제외되고 `.gitignore`, `readme.md` 만 **staging area**에 등록된 것 확인
- `git commit -m 'git initialize` : **commit** 실행
- `git log` : **commit** 결과 확인

```
feat : Initial commit
.gitignore 작성
브랜치 이름 변경 및 기본 설정

Ref: 브랜치 변경 명령어

git branch -m main
git config --global init.defaultBranch main
```

[¶ Top](#index)
><br>

# Repository 생성 및 연결

- **github** 접속하여 **Repository** 생성 후 **URL** 복사
- `git remote add origin https://github.com/username/repository.git`
  - 사용자의 **Repository** 를 `origin`이란 명칭으로 연결

# branch 설정
- `git branch` : 브랜치 확인
- `git checkout 브랜치명` : 해당 브랜치로 이동

## 기본 branch 설정
- `git branch -M main` : 기본값은 `master`, 최근에는 지양, `main`으로 변경
- `git config --global init.defaultBranch main` : 전체 기본값을 `main`으로 변경
- `git push -u origin main` : `main` 브랜치로 `origin`에 `push`
  - `origin` - 연결된 레포지토리 한 개일 경우 생략 가능

## develop branch 생성 및 제거
- `main` 과 분리하여 개발을 진행할 `develop` 브랜치 생성
  - `git branch develop` : 생성
  - `git checkout -b develop` : 생성 후 해당 브랜치로 이동
  - `git push -u origin develop` : 원격 저장소에 등록
- `main` 에 `merge` 후 제거 또는 오류로 인한 제거
  - `git branch -d develop` : 제거
  - `git branch -D develop` : 강제 제거
  - `git push origin --delete develop` : 원격 저장소에서 제거

## branch명 예시
- `develop` : 개발
- `feature` : 기능 개발
- `hotfix` : 긴급 버그 수정
- `release` : 다음 배포 개발

## git-flow 기본 사용법

```
main ·----------------->
     |             |
     ·-------·-----·
  develop    |
     ·--------
  feature 
```

- `merge` : `develop` 의 내용을 `main` 에 병합

```bash
git checkout main
git merge develop
git push origin main 
```

- `rebase` : `main` 의 내용을 `develop`에 덮어쓰기

```bash
git checkout develop
git rebase main
```

[¶ Top](#index)
><br>

# python 개발환경 설정
- 프로젝트에 맞는 패키지, 모듈, 라이브러리 설치
- `pip(package installer for Python)` 설치
  - `python -m pip install --upgrade pip`
  - `python`으로 실행되지 않으면 `python3` 로 시도
- **Jupyter Notebook** 사용 설정
  - 생성한 가상환경의 **kernel** 등록
  - `pip install ipykernel`
  - `python -m ipykernel install --user --name 가상환경이름 --display-name 사용할 이름`

# 환경변수 설정
- `.env` : 프로젝트에 필요한 **API Key**등의 환경변수를 등록
  - `.gitignore`에 반드시 포함하여 버전관리 대상에서 제거
  - 다른 장비에서도 사용 가능
  - `.env` 파일이 노출될 경우 위험
- **shell** 문서에 저장하여 호출
  - **Mac : zshrc**
  - 노출 가능성이 적음
  - 장비마다 등록 필요

  [¶ Top](#index)
><br>