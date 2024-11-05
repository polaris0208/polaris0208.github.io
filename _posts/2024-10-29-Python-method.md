---
layout: post
title: Python 메서드 정리
subtitle: TIL Day 37
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python method

## lambda 
- 익명함수
- 식을 간단하게 표현 가능
- 병렬로 사용할 때 장점

```py
def solution(num1, num2):
    num1 + num2
    return num1 + num2
#
solution=lambda *x:sum(x)
```
```py
def solution(num1, num2):
    answer = num1 - num2
    return answer
#
solution = lambda num1, num2 : num1 - num2
```

## 특수 메서드
1. 객체 생성 및 소멸
- __init__(self, ...): 객체 초기화
- __new__(cls, ...): 객체 생성
- __del__(self): 객체 소멸
2. 문자열 표현
- __str__(self): print()용 문자열 반환
- __repr__(self): 개발자용 문자열 반환
3. 연산자 오버로딩
- __add__(self, other): + 연산자
- __sub__(self, other): - 연산자
- __mul__(self, other): * 연산자
- __truediv__(self, other): / 연산자
- __eq__(self, other): == 연산자
- __lt__(self, other): < 연산자
- __floordiv__(self, other): // 연산자
4. 컨테이너 관련 메서드
- __len__(self): len() 호출 시 길이 반환
- __getitem__(self, key): 인덱싱 지원
- __setitem__(self, key, value): 값 설정
- __delitem__(self, key): 값 삭제
- __iter__(self): 반복 지원
5. 컨텍스트 관리자
- __enter__(self): with 문 시작 시 호출
- __exit__(self, exc_type, exc_val, exc_tb): with 문 종료 시 호출
6. 기타
- __call__(self, ...): 객체를 함수처럼 호출
- __contains__(self, item): in 연산자

## math 모듈
1. 수학 상수
- math.pi: 원주율 π
- math.e: 자연상수 e
2. 삼각 함수
- math.sin(x): x의 사인
- math.cos(x): x의 코사인
- math.tan(x): x의 탄젠트
3. 역삼각 함수
- math.asin(x): 사인 값 x의 아크사인
- math.acos(x): 코사인 값 x의 아크코사인
- math.atan(x): 탄젠트 값 x의 아크탄젠트
4. 로그 함수
- math.log(x[, base]): x의 자연 로그 또는 주어진 밑(base)의 로그
- math.log10(x): x의 밑 10 로그
- math.log2(x): x의 밑 2 로그
5. 제곱근 및 제곱
- math.sqrt(x): x의 제곱근
- math.pow(x, y): x의 y 제곱
6. 기타 유용한 함수
- math.factorial(n): n! (팩토리얼)
- math.gcd(a, b): a와 b의 최대 공약수
- math.ceil(x): x보다 크거나 같은 최소 정수
- math.floor(x): x보다 작거나 같은 최대 정수
- math.radians(x): x도를 라디안으로 변환
- math.degrees(x): x라디안을 도로 변환
7. 난수 생성
- math.isclose(a, b): a와 b가 근사하게 같은지를 확인

## Statistics
1. 평균 관련
- mean(data): 산술 평균
- median(data): 중앙값
- median_low(data): 중앙값 하한
- median_high(data): 중앙값 상한
2. 최빈값
- mode(data): 최빈값
- multimode(data): 모든 최빈값
3. 분산 및 표준편차
- variance(data): 분산
- s- tdev(data): 표준편차
4. 기타
- range(data): 범위
- quantiles(data, n=4): 분위수

## join 메서드
1. 기본 설명
`str.join(iterable)`
- str: 문자열
- iterable: 문자열 요소들을 포함하는 iterable(리스트, 튜플)
2. 특징
- 구분자: str, 각 요소 사이에 이 문자열이 삽입
- 입력 조건: 모든 요소는 문자열, 그렇지 않으면 TypeError가 발생

## map & zip
1. map
`map(function, iterable)`
- 주어진 함수를 iterable의 각 요소에 적용, 새로운 iterable을 생성
- map 객체 반환 (결과를 리스트로 변환 가능)

2. zip
`zip(iterable1, iterable2, ...)`
- 여러 iterable의 요소를 묶어 튜플
- 각 iterable의 같은 인덱스에 있는 요소들을 묶어 새로운 iterable을 생성
- zip 객체 반환 (결과를 리스트로 변환할 수 있음)
