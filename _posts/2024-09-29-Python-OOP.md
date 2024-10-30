---
layout: post
title: Python OOP 기본 개념
subtitle: TIL Day 7
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

## Python OOP 개념

1. OOP - Object Oriented Programming / 객체지향프로그래밍
* 코드 구성하는 방법의 규칙, 실행 방법

2. class - 데이터의 구조 정의 - 설계도면과 같은 것
* instance - class를 적용해 만들어진 실제 사례
* init - 생성자, 초기화 initialize method, 초기값을 설정, self를 사용
_init은 항상 첫번째에 사용_
* self - instance: 객체 자신을 의미
* str - 객체의 전달 내용을 문자열로 전달

```py
class Puppy:

  def __init__(self, name, breed):
    self.name = name,
    self.age = 0.1,
    self.breed = breed

  def __str__(self):
    return f"{self.breed} puppy named {self.name}"

ruffus = Puppy(
  name= "ruffus", 
  breed = "beagle")

print(ruffus)
```
