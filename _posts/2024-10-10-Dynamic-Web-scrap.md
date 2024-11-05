---
layout: post
title: 동적 Web 스크랩
subtitle: TIL Day 18
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Web]
author: polaris0208
---

# Dynamic Scrap

> Python으로 브라우저를 조작하여 Scrap
Playwright 모듈 활용
수집한 데이터 csv 파일로 저장

## Playwright 설치

```bash
pip install playwright
# 여기까지만 하면 동작하지 않음
# 한번 더 설치
python playwright install
# 해당 명령어로 설치가 되지 않을 경우
# python -m playwright install
```

## 필요한 module 설치

```py
from playwright.sync_api import sync_playwright
# .sync_api는 jyupiter에서 동작하려면 별도 설정 필요(어려움)
from bs4 import BeautifulSoup
import time
# 동작 지연용
import csv
# csv 파일편집용
```

## Browser 조작
### 새로운 페이지 생성

```py
p = sync_playwright().start()
    browser = p.chromium.launch(headless=False)
    # 브라우저는 크롬/ headless=False는 동작과정을 화면으로 표시
    page = browser.new_page()
```

### 페이지 이동

```py
page.goto(url)
# 입력한 url로 이동
time.sleep(1)
# 컴퓨터의 조작이 페이지 응답 속도보다 빠를 수 있기 떄문에 시간 지연

for i in range(5):
  page.keyboard.down("End")
  # 키보드의 end키 입력
  # 사용자의 키보드에 없더라도 사이트가 허용하는 키 모두 사용 가능
  time.sleep(1)
  # End키를 사용하는 이유
  # 스크롤을 내릴 때마다 추가 데이터를 호출하는 사이트 존재
  # End키를 반복 시켜 스크롤을 끝까지 내려줌
```

### 페이지 종료
* html 데이터를 추출한 뒤 페이지 종료
* 추출한 html 데이터를 활용 가능하도록 parsing

```py
 content = page.content()
 p.stop()

 soup = BeautifulSoup(content, "html.parser")
```

### 데이터 scrap
* 일반 scrap 과 동일한 방식을 취함
* .find()로 element를 추적하여 데이터 추출

```py
job_db = []
jobs = html_content.find_all("div", class_="JobCard_container__REty8")

for job in jobs:
      url = f'https://wanted.co.kr{job.find("a")["href"]}'
      title = job.find("strong", class_="JobCard_title__HBpZf").text
      company_name = job.find("span", class_="JobCard_companyContent___EEde").text
      reward = job.find("span", class_="JobCard_reward__cNlG5").text

      job_data = {"title" : title,
      			"company_name" : company_name,
      			"reward" : reward,
                "link" : url}
      job_db.append(job_data)
```

### csv 파일로 저장
* csv 파일을 생성
* .writer()로 작성 설정
* .writrow() : 한 줄씩 작성
* with문 : 자동으로 파일을 닫아줌
(코드가 간결해지고 열린상태로 방치되는 것 방지)

```py
with open(file=f'{keyword}_jobs_db.csv', mode="w", encoding="utf-8") as file:
# with ~ as 형식으로 사용
writer = csv.writer(file)
writer.writerow(["Title", "Company", "Reward", "Link"])
for job in job_db:
  writer.writerow(job.values())
  # .values() 딕션너리에서 벨류값만 추출
file.close() 
# with문으로 닫아주지만 습관적으로 적어주는 것이 좋다
```
