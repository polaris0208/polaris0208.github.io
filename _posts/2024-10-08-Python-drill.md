---
layout: post
title: Python 연습
subtitle: TIL Day 16
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---
# Python 실습
## 숫자 맞추기 게임 코드 작성
### 조건

```py
1부터 10까지의 숫자를 선택
플레이어의 선택과 컴퓨터의 선택이 같으면 정답
오답일 경우 정답에 대한 정보 제공
범위를 벗어날 경우 경고
게임 종료 시 안내
```

### 계획

```py
값은 1 부터. 10, 정수형 -> int(), input()
무작위로 숫자 생성 -> 모듈 활용
입력값 비교 -> 연산자 활용
조건 설정 -> if, elif, else
반복 설정 -> while 또는 for loop
```

## 작성
### random 모듈 활용
* 정수형만 취급
* randint(1,10)
* randrange(1,11,1)

```py
pip install random
import random
from random import randint
```

### 입력값 설정
* randint()로 컴퓨터의 값 생성
* input()으로 플레이어 값 생성
* int()로 플레이어 값을 정수형으로 변환하여 컴퓨터 값과 비교

```py
  player_choice = int(input('1과 10 사이의 숫자를 입력하세요.'))
  computer_choice = randint(1,10)
```

### 값 비교

```py
player_choice < 1 or player_choice >10:
player_choice == computer_choice
player_choice > computer_choice
player_choice < computer_choice
```

### 조건 형성
* 조건이 여러개
* if-elif  사용하여 정보 출력
* 범위 조건을 제일 먼저 작성
* 다른 조건이 위에 위치 할 경우 범위를 벗어나도 작동되는 경우 발생

```py
 if player_choice < 1 or player_choice >10:
   print('범위를 벗어 났습니다.')
elif player_choice == computer_choice:
  print('정답!')
elif player_choice > computer_choice:
  print('다운!')
elif player_choice < computer_choice:
  print('업!')
```

### 반복 설정
* 정답을 맞출 때까지 반복하도록 설정
* while 반복문 사용

```py
playing_game = True
computer_choice = randint(1,10)
while playing_game: # 숫자 비교 반복문
  player_choice = int(input('1과 10 사이의 숫자를 입력하세요.'))
  if player_choice < 1 or player_choice >10:
    print('범위를 벗어 났습니다.')
  elif player_choice == computer_choice:
    print('정답!')
  elif player_choice > computer_choice:
    print('다운!')
  elif player_choice < computer_choice:
    print('업!')
```

### 추가조건
> 재시작 또는 종료 기능

* 정답일 경우 재시작 또는 종료 선택

```py
elif player_choice == computer_choice:
     print('정답!')
     answer = input('다시하려면 r, 종료하려면 q를 눌러주세요.')
```

* 재시작 - 기존 숫자 맞추기 반복문 밖에 while 반복문 추가
* while 반복문 시작을 위한 switch 코드 제작

```py
while switch_on == True:
  game_on = True
  print('게임을 시작합니다.')
  computer_choice = randint(1,10)
  print('컴퓨터가 숫자를 선택중입니다.')
  while game_on: 
  				...
```

```py
switch = input('turn on/off')
while not switch == 'on' and not switch == 'off':
  try: 
    switch = input('turn on/off')
    if switch == 'on' or switch == 'off': switch_on = True
  except: continue
if switch == 'on':
  print('booting...')
  switch_on = True
elif switch == 'off':
  print('hold')
  switch_on = False
```

* 종료 - while 반복문 정지

```py
# 숫자 맞추기 while 반복문 종료 후
# 게임 반복문 종료 값을 반환하여 종료
    elif player_choice == computer_choice:
     print('정답!')
     answer = input('다시하려면 r, 종료하려면 q를 눌러주세요.')
     if answer == 'r': 
      print('다시 시작합니다.')
      continue
     elif answer == 'q':
      switch_kill = 'k'
      break
  if switch_kill == 'k':
    print('종료합니다.')
    break
```

## Person Class 생성
### 조건

```
이름, 나이, 성별로 class 구성
성별은 male 또는 female
성별 유효성 검사
instance 생성
출력 조건 = (key : value)형태, 행 분리
```

### 계획

```
instance의 성별을 조건문을 통해 판별
input()을 통해 정정값 받기
정정값이 조건에 부합할 때까지 반복
출력에 줄바꿈 사용
```

### 기본구조 작성

```py
class Person:
  def __init__(self, name, age, gender):
    self.name = name
    self.age = age
    self.gender = gender
    
faker= Person(name = "이상혁", age=20, gender = 'god')
```

### 성별 유효성 검사
* male도 female도 아닌 경우 판별

```py
not self.gender == 'male' and not self.gender == 'female'
```

### 정정값 받기
* 받은 정정값도 조건에 해당하는지 판별

```py
self.gender = input('male or female')
if self.gender == 'male' or self.gender == 'femlae'
```

### 예외 처리
* try-except 사용
* 단순 while문으로 구성할 경우 조건 설정이 복잡

```py
while not self.gender == 'male' and not self.gender == 'female': 
# instance의 성별이 male 또는 female가 아닐 경우(예외) 작동
    try: # 정정값을 받고 male 또는 female 값을 입력하면 종료
      self.gender = input('male or female')
      if self.gender == 'male' or self.gender == 'femlae': break 
    except: continue # 정정값이 조건에 맞지 않으면 반복
```

### 출력
* f-string 으로 데이터를 받아 입력
* \n 으로 줄바꿈 실행하여 행 분리

```
  def display(self):
    print(f'이름 : {self.name}, 성별 : {self.gender}\n나이 : {self.age}')
```

### 정리

```py
class Person:
  def __init__(self, name, age, gender):
    self.name = name
    self.age = age
    self.gender = gender
    while not self.gender == 'male' and not self.gender == 'female': 
      try: 
        self.gender = input('male or female')
        if self.gender == 'male' or self.gender == 'femlae': break
      except: continue
        
  def display(self):
    print(f'이름 : {self.name}, 성별 : {self.gender}\n나이 : {self.age}')
    
faker= Person(name = "이상혁", age=28, gender = 'god')
faker.display()
# 성별을 god으로 설정하여 메시지 출력
male or female male 
# input 'male'
이름 : 이상혁, 성별 : male
나이 : 20
```