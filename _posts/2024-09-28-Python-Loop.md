---
layout: post
title: Python Loop
subtitle: TIL Day 6
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---
## Python 기초

1. 주석 달기

* '#' 뒤에 작성한 내용은 인식되지 않음, 메모로 활용
* """내용""" 

### while
* 조건이 False가 되기 전까지 반복
```py
from random import randint

playing = True

while playing:
  user_choice = int(input("Choose number."))
  pc_choice = randint(1, 50)
  
  if user_choice == pc_choice:
    print("You won!")
  
  elif user_choice > pc_choice:
    print("Lower!")
  
  elif user_choice < pc_choice:
    print("Higher!")
```

### method
* method - 데이터에 결합해서 사용하는 함수 - .를 붙여 사용
* list, [] 사용, 복수형 사용

```py
days_of_week = ["Mon", "Tue", "Wed", "Thur", "Fri"]

print(days_of_week.count("Wed"))

# 1

days_of_week.reverse()

print(days_of_week)

#['Fri', 'Thur', 'Wed', 'Tue', 'Mon']

```
* 위치를 0 부터 인식하기 때문에 4번째 값을 호출하기 위해서는 3을 입력
```
print(days_of_week[3])

# Tue
```
#### tuples () 사용
* list와 다르게 불변성을 가짐

#### dictionary, {} 사용
* key - value 의 조합으로 이루어짐
* 내부에 리스트를 넣어 사용 가능
* 수정 가능

```py
player = {
  'name' : 'user',
  'age' : 20,
  'alive' : True,
  'fav_food' : ["pizza", "hamburger"]
}

player['fav_food'].append("noodle")
print(player.get('fav_food'))
print(player['fav_food'])

# ['pizza', 'hamburger', 'noodle']
['pizza', 'hamburger', 'noodle']
```

### for loop 순차 실행

```py
from requests import get

websites = [
  "google.com",
  "airbnb.com",
  "x.com",
  "facebook.com",
  "tiktok.com"
]

results ={}

for website in websites:
  if not website.startswith("https://"):
    website = f"https://{website}"
  response = get(website)
  if response.status_code >= 200 and response.status_code < 300:
    results[website] = "OK"
  elif response.status_code >= 300 and response.status_code < 400:
    results[website] = "REDIRECT"
  elif response.status_code >= 400 and response.status_code < 500:
    results[website] = "ERROR"
  else:
    results[website] = "FAILED"

print(results)

# {'https://google.com': 'OK', 'https://airbnb.com': 'OK', 'https://x.com': 'OK', 'https://facebook.com': 'OK', 'https://tiktok.com': 'OK'}

```