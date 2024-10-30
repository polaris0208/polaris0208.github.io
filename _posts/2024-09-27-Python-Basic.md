---
layout: post
title: Python 기초
subtitle: TIL Day 5
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---
# Python 기초

## 기본
_작성은 top to bottom_

### variable - 데이터 명명, 명명된 이름으로 데이터에 접근

* camel case - myAge = javascript에서 주로 사용
* snake case _ my_age = python 
* 숫자만 사용하는 것은 지양, 문자로 시작하되 숫자를 섞는 것은 가능

### 문자
* "" 사용

### True, False
* T, F 는 항상 대문자

### print()
* print("Hello my name is", my_name)

### function
* def 을 통해 정의
* (): 사용

```py
def say_hello():
   print("hello how r u?")

say_hello()
```

### 공백의 사용 - python의 특징
* 두 칸 들여쓰기 - 코드 포함
* tap 1번 클릭 or space 2번 클릭

### function 커스터마이즈

```py
def say_hello(user_name):
  print("hello", user_name, "how r u?")

say_hello("name")
```

* user_name = parameter(매개변수), name = argument(전달인자)
* parameter 와 argument는 여러개 사용 가능하며, 이 때는 개수와 순서에 맞춰 작성

```py
def say_hello(user_name, user_age):
    print("hello", user_name)
    print("you r", user_age, "years old")

say_hello("name", 28)
```

* 연산에 parameter 활용 
* 거듭제곱 연산 = ** , power / 2제곱 = square_

```py
def tax_calculateor(salary):
    print(salary*0.35)

tax_calculateor(100)
```
* paramete에 기본값 설정

```py
def say_hello(user_name="everyone"):
  print("hello", user_name)

say_hello()
```

### return
* 함수 밖으로 값을 보냄

```py
def tax_calc(money):
  return money*0.35

def pay_tax(tax):
  print("thank you for paying", tax)
  
pay_tax(tax_calc(100000000))
```

* f-string

```py
print(f"hello I'm {my_name}")
```
* return 활용

```py
def make_juice(fruit):
  return f"{fruit}+juice"

def add_ice(juice):
  return f"{juice}+ice"

def add_sugar(iced_juice):
  return f"{iced_juice}+sugar"

juice = make_juice("apple")
cold_juice = add_ice(juice)
perfect_juice = add_sugar(cold_juice)

print(perfect_juice)

# apple+juice+ice+sugar
```

_return은 함수의 끝(이후 값은 전달 x)_

### 조건문

```py
password_correct = False

if password_correct:
  print("Here is your money")
else : 
  print("Worng password")
```
* if, else, elif
* if 조건 : 결과
* else: 대안
* elif 다른 조건: 대안
* 상단의 조건이 충족되면 아래의 조건이 충족되어도 출력되지 않음

_같다 ==, 다르다 !=, =은 값을 나타낼 때, ==은 값을 비교할 때_

#### input
* 입력값을 return값으로 사용

#### type
* variabl의 타입을 설명

#### int
* 문자형으로 표현된 숫자를 정수형 숫자로 변환

```py
age = int(input("how old are you?"))

if age < 18:
  print("You can't drink.")

elif age >= 18 and age <= 35:
  print("You drink bear!")

else: 
  print("go ahead")
```

### python standard library 
* built_in_functions 기본 포함된 함수들 ex) print, int....
* 나머지 함수들은 필요에 맞게 적용해서 사용

```py
from random import randint, uniform
```
* random 모듈에서 randint, uniform 함수 불러오기

```py
user_choice = int(input("Choose number."))
pc_choice = randint(1, 50)

if user_choice == pc_choice:
  print("You won!")

elif user_choice > pc_choice:
  print("Lower!", pc_choice)

elif user_choice < pc_choice:
  print("Higher!", pc_choice)
```
* randint 적용하여 무작위 정수값 출력