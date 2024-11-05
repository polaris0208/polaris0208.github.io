---
layout: post
title: HTML & CSS 기초
subtitle: TIL Day 20
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Web]
author: polaris0208
---

# html & CSS 기초
>html 은 뼈대
CSS는 꾸미기 기능
java script는 동작

## vscode 사용 tip

```
html 작성에서 ! 하면 자동완성
command / 주석처리
추가기능 open in browser 설치
```

## html 기본구조[¶](https://github.com/polaris0208/TIL/blob/main/Web/TIL_24_1012_html%26CSS_basic.md#기본-요소-tags)
- **Github Pages** : **Jekyll** 사용
  - **post** 내부에 **html** 코드가 포함되면 **markdown**을 **html**로 변환하는 과정에 오류 발생
  - 별도의 링크로 연결하여 내용 대체

### 기본 요소 tags
* hr, div, span 의 용도구분에 주의

* 선택자의 종류는 매우 다양
* 포함관계가 있는 경우가 있음
  * tags의 하위 관계(포함 관계) - 부모-자식 등
  * 부모-자식 선택자는 css를 공유(전체는 아님)
* 선택자 간의 우선 순위가 달라질 수 있음
  * 선택자 개념: https://coding23213.tistory.com/15

```py
* 전체 선택자 *
* 하위 선택자 - 요소 내부의 모든 요소 / 현재 요소의 선택자(공백)하위 요소의 선택자
* 자식 선택자 - 요소 내부의 요소(하위의 하위 요소는 해당하지 않음) 현재 > 자식
* 인접 선택자 - 현재 요소 뒤에 나오는 요소 / 현재 + 인접
* 형제 선택자 - 같은 계층에 있는 요소 / 현재 ~ 형제
* 그룹 선택자 - 여러 요소들을 묶어서 / 요소1, 요소2, 
```

### flex
* html은 박스형태 
* block 1줄 전체를 차지 - 위에서 아래로
* inline 글자같은 것들, 자신의 크기 만큼만 차지 - 가로로 배치

```
display : flex 작성 기준이 위-아래 에서 좌-우로 변경
justify-content: center 중앙 정렬(좌우폭)
align-items: center 중앙 정렬 (상하폭)
개발자 도구- 하단 - 스타일; 다른 설정 확인 가능
```