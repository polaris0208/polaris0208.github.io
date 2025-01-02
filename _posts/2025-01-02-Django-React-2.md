---
layout: post
title: Django - React 연결 2
subtitle: TIL Day 101
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools, Web]
author: polaris0208
---

# Django - React 연결 적용

## 구조

```
project/
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── package-lock.json
│   ├── src/
├── backend/
│   ├── Dockerfile
│   ├── manage.py
│   └── ...
├── db/
├── docker-compose.yml
```

## 실횅

```bash
# 컨테이너 실행
docker-compose up --build

# 컨테이너 종료
ctrl + c

# 컨테이너 삭제
docker-compose down
```

## .env
- `DJANGO_SETTINGS_MODULE` : `settings.py`의 위치를 알려주는 변수
- `REACT_APP_API_URL` : **React**가 연결할 포트

```bash
DJANGO_SETTINGS_MODULE="coding_helper.settings"
REACT_APP_API_URL="http://localhost:8000"
```

## backend Dockerfile
- `WORKDIR /app/backend` : 명시적으로 구분

```yml
FROM python:3.10-slim

# PostgreSQL 설치
RUN apt-get update && \
    apt-get install -y postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# dockerize 설치 / 의존성 관리 도구
RUN apt-get update && \
    # 패키지 목록을 업데이트
    apt-get install -y wget && \
    # wget 설치 / HTTP, HTTPS, FTP 에서 파일 다운로드 / -y 모두 확인
    wget https://github.com/jwilder/dockerize/releases/download/v0.6.1/dockerize-linux-amd64-v0.6.1.tar.gz && \
    tar -xvzf dockerize-linux-amd64-v0.6.1.tar.gz && \
    # tar : 압축 해제 / -x 압축 해제 /-v는 과정 표시 / -z는 .tar.gz 형식, -f 이름 지정
    mv dockerize /usr/local/bin/
    # mv 이동

# 작업 디렉토리 설정
WORKDIR /app/backend

# 종속성 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 8000을 노출
EXPOSE 8000
```

## frontend Dockerfile

```yml
# frontend/Dockerfile
FROM node:20-slim

WORKDIR /app/frontend

COPY package.json yarn.lock ./
RUN yarn install

RUN yarn add react-icons

COPY . .

EXPOSE 3000

CMD ["yarn", "start"]
```

## docker-compoose.yml
- `context: ./frontend` : `Dockerfile`의 경로 지정

```yml
services:
  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    env_file:
      - .env
    environment:
      REACT_APP_API_URL: http://localhost:8000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: drf
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
    container_name: postgres
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: postgres
    ports:
      - "5432:5432"
    volumes:
      - ./db/data:/var/lib/postgresql/data # 볼륨 저장
      - ./db/init:/docker-entrypoint-initdb.d # 컨테이너 생성 시 초기화 sql 파일

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
      - ./db/pgadmin/:/var/lib/pgadmin

# volumes: {}
```