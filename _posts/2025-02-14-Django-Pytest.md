---
layout: post
title: Django Pytest
subtitle: TIL Day 144
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

# Django Test Code

## 코드 개요
- 사용자 회원가입, 로그인, JWT 토큰 발급 및 검증, 보호된 엔드포인트 접근 등을 테스트

---

## 주요 클래스 및 메서드

### 1. `BaseTest` 클래스
- **기능**: 공통적으로 사용되는 초기 설정을 정의
- **구성 요소**:
  - `setUp()` 메서드:
    - 테스트에 필요한 URL Reverse 설정 (`signup_url`, `login_url`, `protected_url`)
    - 기본 사용자 역할(Role) 생성
    - 테스트용 사용자 계정 생성 (`testuser`)

```python
class BaseTest(APITestCase):
    def setUp(self):
        self.signup_url = reverse('signup')
        self.login_url = reverse('login')
        self.protected_url = reverse('protected')

        self.user_role, _ = Role.objects.get_or_create(name="USER")

        self.user = User.objects.create_user(
            username="testuser",
            password="testpassword",
            nickname="Test Nickname"
        )
        self.user.roles.add(self.user_role)
```

---

### 2. `AuthenticationTests` 클래스
- **기능**: 회원가입 및 로그인 관련 테스트 수행
- **테스트 메서드**:
  1. **회원가입 성공 테스트 (`test_signup_success`)**
     - 정상적인 회원가입 요청이 성공하는지 확인
     - 응답 상태 코드: `201 Created`
     - 응답 데이터: 사용자 정보(`username`, `nickname`, `roles`) 확인

```python
def test_signup_success(self):
    payload = {
        "username": "newuser",
        "password": "newpassword",
        "nickname": "New Nickname"
    }
    response = self.client.post(self.signup_url, data=payload)
    self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    self.assertEqual(response.data["username"], payload["username"])
    self.assertEqual(response.data["nickname"], payload["nickname"])
    self.assertEqual(response.data["roles"], [{"role": "USER"}])
```

  2. **회원가입 실패 테스트 (`test_signup_failure`)**
     - 필수 필드가 누락된 경우 회원가입 실패 확인
     - 응답 상태 코드: `400 Bad Request`
     - 응답 데이터: 누락된 필드 정보(`password`, `nickname`) 포함 여부 확인

```python
def test_signup_failure(self):
    payload = {"username": "newuser"}
    response = self.client.post(self.signup_url, data=payload)
    self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    self.assertIn("password", response.data)
    self.assertIn("nickname", response.data)
```

  3. **로그인 성공 테스트 (`test_login_success`)**
     - 유효한 사용자 정보로 로그인 요청 성공 확인
     - 응답 상태 코드: `200 OK`
     - 응답 데이터: JWT 토큰 포함 여부 확인

```python
def test_login_success(self):
    payload = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = self.client.post(self.login_url, data=payload)
    self.assertEqual(response.status_code, status.HTTP_200_OK)
    self.assertIn("token", response.data)
```

  4. **로그인 실패 테스트 (`test_login_failure`)**
     - 잘못된 사용자 정보로 로그인 요청 실패 확인
     - 응답 상태 코드: `400 Bad Request`

```python
def test_login_failure(self):
    payload = {
        "username": "wronguser",
        "password": "wrongpassword"
    }
    response = self.client.post(self.login_url, data=payload)
    self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
```

---

### 3. `JWTAuthenticationTests` 클래스
- **기능**: JWT 토큰 발급, 검증 및 보호된 엔드포인트 접근 테스트 수행
- **테스트 메서드**:
  1. **액세스 토큰 및 리프레시 토큰 추출 테스트 (`test_access_token_and_refresh_token_extraction`)**
     - 로그인 후 반환된 JWT 토큰에서 액세스 토큰과 리프레시 토큰 추출 성공 여부 확인

```python
def test_access_token_and_refresh_token_extraction(self):
    login_response = self.client.post(self.login_url, data={
        "username": "testuser",
        "password": "testpassword"
    })
    self.assertEqual(login_response.status_code, status.HTTP_200_OK)

    full_token = login_response.data.get("token")
    self.assertIsNotNone(full_token)

    refresh_token = RefreshToken(full_token)
    self.access_token = str(refresh_token.access_token)
    self.refresh_token = str(refresh_token)

    self.assertTrue(self.access_token)
    self.assertTrue(self.refresh_token)
```

  2. **액세스 토큰 검증 성공 테스트 (`test_access_token_validation_success`)**
     - 유효한 액세스 토큰으로 보호된 엔드포인트 접근 성공 여부 확인
     - 응답 상태 코드: `200 OK`

```python
def test_access_token_validation_success(self):
    self.test_access_token_and_refresh_token_extraction()
    self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.access_token}")
    response = self.client.get(self.protected_url)
    self.assertEqual(response.status_code, status.HTTP_200_OK)
```

  3. **리프레시 토큰으로 액세스 토큰 재발급 성공 테스트 (`test_refresh_token_to_access_token_success`)**
     - 유효한 리프레시 토큰으로 새로운 액세스 토큰 발급 성공 여부 확인
     - 응답 상태 코드: `200 OK`

```python
def test_refresh_token_to_access_token_success(self):
    self.test_access_token_and_refresh_token_extraction()
    response = self.client.post("/api/token/refresh/", data={"refresh": self.refresh_token})

    self.assertEqual(response.status_code, status.HTTP_200_OK)
    self.assertIn("access", response.data)
```

---

## 사용 기술 및 라이브러리
- Django REST Framework (DRF)
- Simple JWT (`rest_framework_simplejwt`)
- Django Test Framework (`APITestCase`)

---

## 주요 URL
- `/signup/`: 회원가입 API 엔드포인트
- `/login/`: 로그인 API 엔드포인트
- `/protected/`: 인증이 필요한 보호된 API 엔드포인트
- `/api/token/refresh/`: 리프레시 토큰으로 새 액세스 토큰 발급 API

---

## 주요 HTTP 상태 코드
- `201 Created`: 회원가입 성공 시 반환
- `200 OK`: 로그인 성공, 보호된 엔드포인트 접근 성공, 새 액세스 토큰 발급 시 반환
- `400 Bad Request`: 잘못된 요청(회원가입/로그인 실패 등) 시 반환
- `401 Unauthorized`: 인증 실패(잘못된/만료된 JWT 등) 시 반환

---