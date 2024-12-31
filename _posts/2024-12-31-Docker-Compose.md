---
layout: post
title: DRF Docker Container 생성
subtitle: TIL Day 99
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

## .dockerignore 작성
- 컨테이너 생성에 불필요한 파일 제외

```py
# .git 디렉터리: 버전 관리 파일은 불필요
.git/

# Python 바이트 코드 파일: 이미 빌드된 파일이므로 제외
__pycache__/
*.pyc

# 테스트 파일 및 폴더
tests/

# 환경 설정 파일(비밀 키나 중요한 데이터 포함될 수 있음)
.env

# 임시 파일이나 개발용 파일
*.log
*.bak
*.swp
*.idea/
*.vscode/
```

## `settings.py`
- 데이터베이스 설정
- `POSTGRES_HOST`
  - `docker-compose` 사용할 경우 : `db` 서비스 이름

```py
import os
from dotenv import load_dotenv

load_dotenv()
DRF_SECRET_KEY = os.getenv("DRF_SECRET_KEY")
POSTGRES_NAME = os.getenv("POSTGRES_NAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")

DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': POSTGRES_NAME,
            'USER': POSTGRES_USER,
            'PASSWORD': POSTGRES_PASSWORD,
            'HOST': POSTGRES_HOST,
            'PORT':'5432'
        }
    }
```

## 로컬 DB 데이터 생성

### Sample 데이터 생성

```json
[
  {"title": "Post 1", "content": "This is the content for Post 1"},
  {"title": "Post 2", "content": "This is the content for Post 2"},
  {"title": "Post 3", "content": "This is the content for Post 3"},
  {"title": "Post 4", "content": "This is the content for Post 4"},
  {"title": "Post 5", "content": "This is the content for Post 5"},
  {"title": "Post 6", "content": "This is the content for Post 6"},
  {"title": "Post 7", "content": "This is the content for Post 7"},
  {"title": "Post 8", "content": "This is the content for Post 8"},
  {"title": "Post 9", "content": "This is the content for Post 9"},
  {"title": "Post 10", "content": "This is the content for Post 10"}
]
```

### SQL 데이터화
- **Json** 데이터 호출
- **SQL** 쿼리 작성
- 테이블 이름 : `sample`
- `SERIAL` : **PostgreSQL**
- `AUTO_INCREMENT` : **MySQL**


```py
import json
import os

with open('DB_test/test_set.json', 'r') as json_file:
    data = json.load(json_file)

sql_query = """
CREATE TABLE IF NOT EXISTS sample (
    id SERIAL PRIMARY KEY,
    title VARCHAR(50),
    content VARCHAR(3000)
);

INSERT INTO sample (title, content) VALUES
"""

values = []
for entry in data:
    # 작은 따옴표 제거
    title = entry['title'].replace("'", "''")  
    content = entry['content'].replace("'", "''")
    values.append(f"('{title}', '{content}')")

# SQL INSERT
sql_query += ",\n".join(values) + ";"

output_dir = 'DB_test/init'
os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성
with open(os.path.join(output_dir, 'sample_dataset.sql'), 'w') as file:
    file.write(sql_query)
```

## 경로 생성
- `DB_test/init`
  - `init` : **PostgreSQL** 초기화 시 추가할 데이터 경로
- `DB_test/pgadmin`
  - **pgadmin4** 경로

## Dockerfile
- `PostgreSQL` : 컨테이너에서 `PostgreSQL`에 접근할 수 있도록 설치
- `dockerize` : 컨테이너 환경에서 대기 또는 다양한 환경 설정을 처리하는 데 사용
- `wget` : **HTTP, HTTPS, FTP** 에서 파일 다운로드
  - `-y` 모두 확인
- `tar` : 압축 해제
  - `-x` : 압축 해제
  - `-v` : 과정 표시
  - `-z` : `.tar.gz` 형식
  - `-f` :  이름 지정
- `mv` : 이동

```yml
# Python 이미지를 사용하여 기본 이미지 설정
FROM python:3.10-slim

# PostgreSQL 설치
RUN apt-get update && \
    apt-get install -y postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# dockerize 설치 / 의존성 관리 도구
RUN apt-get update && \
    # 패키지 목록을 업데이트
    apt-get install -y wget && \
    wget https://github.com/jwilder/dockerize/releases/download/v0.6.1/dockerize-linux-amd64-v0.6.1.tar.gz && \
    tar -xvzf dockerize-linux-amd64-v0.6.1.tar.gz && \
    mv dockerize /usr/local/bin/

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 8000을 노출
EXPOSE 8000
```

## docker-compose.yml
- `dockerize -wait tcp://db:5432 -timeout 30s &&` : `db` 포트가 열릴 때까지 대기, 30초 제한
- `exec python manage.py runserver 0.0.0.0:8000` : 1번 프로세스로 실행
- `restart: always` : 종료되면 재시작
- `--noinput || true &&` : 입력이 없으면 넘어가기
- `./DB_test/init:/docker-entrypoint-initdb.d` : `sql`형식의 파일을 찾아 반영

```yml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: test_container
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      DJANGO_SUPERUSER_USERNAME: admin
      DJANGO_SUPERUSER_EMAIL: admin@example.com
      DJANGO_SUPERUSER_PASSWORD: password
    depends_on:
      - db
    command: >
      sh -c "
      dockerize -wait tcp://db:5432 -timeout 30s &&
      python manage.py makemigrations &&
      python manage.py migrate &&
      python manage.py createsuperuser --noinput || true &&
      exec python manage.py runserver 0.0.0.0:8000
      "
  db:
    image: postgres:15
    container_name: postgres_database
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: postgres
    ports:
      - "5432:5432"
    volumes:
      - ./DB_test/data:/var/lib/postgresql/data # 볼륨 저장
      - ./DB_test/init:/docker-entrypoint-initdb.d # 컨테이너 생성 시 초기화 sql 파일

  pgadmin:
    image: dpage/pgadmin4
    restart: always
    container_name: pgadmin
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: pgadmin4@pgadmin.org
      PGADMIN_DEFAULT_PASSWORD: password
    # 볼륨 설정
    volumes:
      - ./DB_test/pgadmin/:/var/lib/pgadmin

# volumes: {}
```

## 실행 및 확인

```bash
# 프로젝트 코드 클론
git clone -b <branchname> <remote-repo-url>

# 컨테이너 생성
docker-compose up --build

# db 확인
docker exec -it postgres_database psql -U user -d postgres
\dt
SELECT * FROM accounts_user;
```

## pgadmin 접속
- 접속 : `http://localhost:5050`
- 새로운 서버 생성
- `Name` : 서버명(지정)
- `Host name` : **Postgres** 컨테이너 이름