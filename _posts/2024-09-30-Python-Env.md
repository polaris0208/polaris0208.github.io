---
layout: post
title: Python 개발환경
subtitle: TIL Day 8
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# Python

## collection 데이터

### list 

```
# [] 사용, 다양한 형태의 데이터를 요소로 포함 가능
```

* [0] #첫 번째 데이터에 접근
* len(리스트) # 리스트 길이 확인
* .remove # 제거, .append # 추가
* 첫 번째 요소 변경 # 리스트[0] = "변경"
* .sort() # 정렬

### tuple 

```
#() 사용
```

* 리스트와 유사함, 조회에 사용하는 대부분의 기능 그대로 사용가능
* 불변 자료이기 때문에 데이터를 변경할 수 없음

### dictionary 

```
# {} 
```
* key : value 조합으로 사용
* del dict(key) # 삭제
* dict.keys() # key 확인
* dict.values() # value 확인

### set # {} 사용, key가 없음
* 중복 데이터 자동으로 없애줌
* .add, .remove

## 제어문과 반복문

### 조건문 

```
# 주어진 조건에 따라 프로그램이 결정
```

* if 조건:
(들여쓰기)실행할 코드
* else:
(들여쓰기)실행할 코드(모든 조건이 거짓일 때)
* elif 다른 조건:
(들여쓰기)실행할 코드(다른 조건) # 조건 검사는 순차적으로 이루어지기 때문에 조건 설정에 유의
* .upper(), .lower() # 대문자 변환, 소문자 변환

### 반복문
* for문 # 데이터들을 순차적으로 순회하면서 코드 실행
* while: # 참인 경우만 반복 / 초기선언조건문 
_거짓이 되는 부분을 설정 해주지 않으면 무한히 반복_

```python
age =18
while age < 20:
	print("미성년자 입니다.")
    age += 1    
```
   
* countinue 아래의 내용을 출력 하지 않고 다시 반복
* break 즉시 반복 정지
* range(시작, 종료, 단계): 실행할 코드 # 시작 기본값 = 0, 종료는 미포함, 단계 기본값 = 0

```python
for i, j enumerate(list_sample): 
	print(i,j) 
    # 위치, 값
```

## 함수

### 내장함수(built-in)
* print() # 내부에 함수가 있는 경우에 연산이 먼저 진행된 후에 출력
* input() # 입력값은 문자열로 받기 때문에 가공 필요
* int(), float(), str() # int = 정수형, florat = 실수형
* sorted() # 기본 오름차순
* abs() # 절대값
* round(수, 반올림 할 자릿수) # 1 = 소수점 한자리로 반올림 ex) 3.14 -> 3.1

### 함수 만들기
* 복습 : https://velog.io/@sh6771/TIL-Day-5-1
* retun 코드를 통해 얻은 값= 반환값
* *args 가변인자 # 튜플 형식으로 데이터 전달
* **kwargs 키워드 인자 # 딕셔너리 형식으로 데이터 전달

```python
def create_profile(name, age, *args, **kwargs):
	profile = { 
    	"name" : name,
        "age" : age,
        "intersets" : args,
        "etc" : kwargs
        }
    return profile
    
profile = create_profile("Alice", 15, "여행", city="원더랜드")
print(profile)
```

* *args, **kwargs  순서로 사용

### 모듈
* 함수, 클래스, 변수를 하나로 미리 묶어놓은 것
* 모듈 불러오기 : import
* from 모듈 import 함수 # 전체는 *

_이름 충돌 주의
모듈 탐색 경로 설정 주의_

### 패키지 
* 모듈을 묶어놓은 단위

```py
_init_.py #패키지를 초기화하는 파일(필수)
  module1.py
  module2.py
```

* 터미널을 이용한 설치

```bash
pip # 패키지 설치 및 관리
pip install 패키지
pip install 패키지==버전번호
pip install --upgrade 패키지 
pip uninstall 패키지
pip list
pip cache purge
```

### 가상환경
* 패키지 설치 등으로 인한 충돌방지
* 따로 가상환경을 만들어 패키지를 설치하고 테스트

```bash
python -m venv 가상환경이름
# 가상환경 생성
# venv = virtural environment
source 가상환경이름/bin/activate
# 가상환경실행
# deactivate : 비활성화 명령어
```

### 터미널에서 Python 구동
* 명령어 python 입력

### 터미널 명령어 참고
https://ko.appflix.cc/useful-collection-of-mac-terminal-commands-to-know/#mv-파일-폴더-이동-파일-이름-변경

## interpretor
* compile 방식과 다르게 코드를 한줄씩 해석하여 전달
* 즉각적인 실행 가능, 수정이 쉬움
-> 속도가 느림
-> error가 있어도 발생지점 전까지는 실행됨 

### 고수준 언어, 추상화
* 인간이 이해하기 쉬운 방식

### IDE == 통합개발 환경
* VSCode : 마이크로소프트의 무료 코드 편집기, 확장성, 다양한 언어 지원, 통합 터미널
https://code.visualstudio.com

* Pycharm : JetBrain의 Python 전용 IDE 커뮤니티 에디션 무료 사용, 강력한 디버깅 도구, 자동화된 코드 분석, 통합된 테스트 도구
https://www.jetbrains.com/pycharm/

* Jupyter Notebook :데이터 과학, 머신러닝, 
-인터렉티브 환경(코드, 설명, 시각화), 셀 기반 실행(셀단위 코드실행, 결과 확인), Markdown(코드와 설명 작성) 지원
https://www.anaconda.com/download

_Jupyther Notebook 코드와 결과를 직관적으로 보여주기 때문에 학습에 적합함_
_결과 확인 - shift+enter
변수의 이름만 입력하면 출력_

### 이름짓기
* 변수(variable) 이름에 함수이름은 피한다
* 의미를 명확히

### 메모리
* 변수는 메모리 내부에 저장됨
* 전역변수 : 프로그램 전체에서 접근 가능
* 지역변수 : 특정 영역에서만 접근 가능

## 연산자
* % : 나머지 , // : 몫
* 논리를 구성할 때 자주 활용

```
a = 10
a % 3 # a를 3으로 나눈 후 나머지
# 1
```

* == : 같음 != : 다름

### 복합대입연산자
* = 할당 ex) a += b = a= a+b # +, -, /, *, ** 동일

```
a %=b = a를 b로 나누고 나머지를 a에 할당
a //= b = a를 b로 나누고 몫을 a에 할당
```

### 비트연산자 : 이진수로 변경 후 연산

```
a = 5  # 이진수로 101
b = 3  # 이진수로 011

print(a & b)  # 1 (이진수 001) # and
print(a | b)  # 7 (이진수 111) # or 
print(a ^ b)  # 6 (이진수 110) # xor
# xor - 베타적 논리합: 둘 중 한 개만 참인 경우 판단
print(~a)     # -6 (이진수 보수) # not 
```

* 보수 : 보충하는 수 ex) 10진수 4의 10의 보수는 6
* 이진수의 보수 1의 보수, 2의 보수
* 1의 보수 숫자의 합을 2의 제곱수 -1로, 각 자리 -1 = 비트 반전
* 2의 보수 숫자의 합을 2의 제곱수로 1의 보수 + 1
* 양수 이진수에는 0, 음수 이진수에는 1이 데이터 왼쪽 끝에 적용

```
a=5 # 이진수 0101
~a # not 연산 
# 0101 반전 1010 으로 데이터에 저장
# 결과를 출력할 때는 역산으로 출력
# 2의 보수 역산 : -1 후에 비트 반전
# 음수 010 에서 -1 = 음수 001 
# 비트 반전 음수 110 = -6

# 공식은 -(x+1)
```

### 데이터 타입
* 문자형 ",' 을 문자열 안에 사용할 때
* "I'm" 과 같이 감싸는 것을 구분한다