---
layout: post
title: Python File
subtitle: TIL Day 96
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---

# File

## 파일 열기 
- **파일이름**: 실제로 접근하려는 파일의 경로와 이름
- **모드(Mode)**: 파일의 용도
  - `"r"`
    - 기본값
    - 읽기 전용 모드
    - 파일이 존재하지 않으면 에러 발생
  - `"w"`
    - 쓰기 모드
    - 이전 파일 존재 : 초기화
    - 이전 파일이 없는 경우 : 새로 생성
  - `"a"`
    - 추가(Append) 모드
    - 새로운 내용 추가
    - 이전 파일이 없는 경우 : 새로 생성
  - `"x"`
    - 쓰기 전용으로 새 파일 생성
    - 파일이 이미 존재하면 에러 발생
  - `"r+"`
    - 읽기/쓰기 겸용 모드
  - `"b"`
    - 바이너리 모드. `"rb"`, `"wb"` 등의 형태로 다른 모드와 조합
    - 텍스트가 아닌 이미지, 음악 파일 등 이진 데이터 처리 시 사용
  - `"t"`
    - 기본값
    - 텍스트 모드
    - `"r"`, `"w"` 등이 기본적으로 `"t"` 모드
- `file.close()` : 파일 사용 후 종료

```python
file = open("example.txt", "r")
```

## 파일 읽기
- `read()`: 파일 전체 내용
- `readline()`: 파일에서 한 줄씩
- `readlines()`: 파일 전체 내용을 한 줄씩 나누어 리스트 반환

### 예시

```python
file = open("example.txt", "r", encoding="utf-8")  # UTF-8 인코딩 지정 가능
content = file.read()  # 파일 전체 읽기
line = file.readline()  # 첫 번째 줄 읽기
print(line)
print(content)
lines = file.readlines()  # 모든 줄을 읽어 리스트로 반환
for l in lines:
    print(l.strip())  # strip()으로 양 끝 공백 제거
file.close()
```

## 파일 쓰기

### 쓰기 모드

```python
file = open("output.txt", "w", encoding="utf-8")
file.write("첫 번째 줄\n")
file.write("두 번째 줄\n")
file.close()
```

### 추가 모드

```python
file = open("output.txt", "a", encoding="utf-8")
file.write("새로운 줄을 추가합니다.\n")
file.close()
```

## with문
- `with`: 자동으로 파일 종료

```python
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)
```

## 바이너리 파일

- 이미지, 오디오, 영상, **PDF** 등 텍스트로 이루어지지 않은 파일
- `"rb"` : 원본 이미지
- `"wb"` : 복사본을 작성

```python
# 이미지 파일 복사 예제
with open("image.jpg", "rb") as src:
    data = src.read()

with open("copy.jpg", "wb") as dst:
    dst.write(data)
```

## 파일 포인터 위치 제어
- `seek()` : 파일 내 특정 위치로 포인터를 이동
- `tell()` : 현재 위치

```python
with open("example.txt", "r", encoding="utf-8") as f:
    print(f.read(5))   # 처음 5글자 읽기
    print(f.tell())    # 현재 파일 포인터 위치 확인
    f.seek(0)          # 파일 포인터를 다시 파일 시작 위치로 이동
    print(f.read(5))   # 다시 처음 5글자 읽기
```

## 예외 처리
- `try-except` 구문 사용
- 파일이 존재하지 않거나 경로가 잘못된 경우
- 권한 부족으로 파일을 열 수 없는 경우
- 디스크 문제가 발생한 경우

이런 상황에 대비해 할 수 있습니다.

```python
try:
    with open("nonexistent.txt", "r", encoding="utf-8") as f:
        data = f.read()
        print(data)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except PermissionError:
    print("파일에 접근할 권한이 없습니다.")
```

## 실용 예제

### 텍스트 파일에서 특정 단어 개수 세기

```python
word_to_count = "Python"
count = 0

with open("example.txt", "r", encoding="utf-8") as f:
    for line in f:
        # 한 줄에서 해당 단어가 몇 번 나오는지 세기
        count += line.count(word_to_count)

print(f"'{word_to_count}' 단어는 {count}번 등장합니다.")
```

### 로그 기록하기

```python
import datetime

def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open("app.log", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)

write_log("프로그램 시작")
write_log("사용자가 로그인했습니다.")
```
