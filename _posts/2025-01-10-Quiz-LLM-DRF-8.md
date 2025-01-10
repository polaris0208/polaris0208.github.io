---
layout: post
title: Quiz_LLM JWT 인증
subtitle: TIL Day 109
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---

> **JWT** 인증을 통해 로그인된 사용자만 **API**에 접근이 가능하도록 설정

## JWT 설정
- `settings.py`
- 설치 : `pip install djangorestframework-simplejwt==5.3.1`
- **Token**
  - **Access** : **API** 접근용
  - **Refresh** : 토큰 재발급용
- `DEFAULT_AUTHENTICATION_CLASSES`
  - 인증 방식 설정 : **JWT**
- `DEFAULT_PERMISSION_CLASSES`
  - 기본 권한 설정 : 인증된(로그인된) 사용자

```py
...
INSTALLED_APPS = [
    ...
    "rest_framework_simplejwt",
    ...
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=1),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=1),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": False,
    "UPDATE_LAST_LOGIN": False,
}
...
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',  
    ),
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
}
```

## 토큰 발급

### 최초발급
- 사용자가 회원가입 후 이메일 인증을 완료하면 발급
- `res.set_cookie("access", access_token, httponly=False)`
  - **API** 사용을 위해 `access` 토큰은 쿠키에 저장
- `user.refresh_token = refresh_token`
  - `refresh` 토큰은 안전하게 **DB**에 저장

```py
class EmailVerificationView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, uidb64, token):
        try:
            # 유저 ID 복호화
            uid = urlsafe_base64_decode(uidb64).decode()
            user = get_object_or_404(User, pk=uid)

            # 토큰 검증
            if default_token_generator.check_token(user, token):
                user.is_active = True  # 계정 활성화
                user.save()

                # JWT 토큰 발급
                from rest_framework_simplejwt.tokens import RefreshToken

                refresh = RefreshToken.for_user(user)
                access_token = str(refresh.access_token)
                refresh_token = str(refresh)

                # 응답에 토큰 포함
                res = Response(
                    {
                        "message": "Email verified successfully.",
                        "token": {
                            "access": access_token,
                            "refresh": refresh_token,
                        },
                    },
                    status=status.HTTP_200_OK,
                )

                # JWT 토큰을 쿠키에 저장
                res.set_cookie("username", user.username, httponly=False)
                res.set_cookie("access", access_token, httponly=False)
                # res.set_cookie("refresh", refresh_token, httponly=True)
                user.refresh_token = refresh_token
                user.save()
                return res
            else:
                return Response(
                    {"message": "Invalid or expired token."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except Exception as e:
            return Response(
                {"message": "Invalid request."}, status=status.HTTP_400_BAD_REQUEST
            )
```

### 로그인/로그아웃
- `def get_permissions(self):`
  - 로그인과 회원가입은 모든 사용자가 접근 가능하도록 설정
- `token = TokenObtainPairSerializer.get_token(user)`
  - 로그인 후 토큰 발급
- 로그아웃
  - 쿠키의 `access` 토큰 삭제
  - **DB**의 `refresh` 토큰 삭제

```py
class AuthAPIView(APIView):

    def get_permissions(self):
        """POST 요청은 인증 없이 허용"""
        if self.request.method == "POST":
            return [AllowAny()]
        return [IsAuthenticated()]

    # 로그인
    def post(self, request):
        # 유저 인증
        user = authenticate(
            username=request.data.get("username"), password=request.data.get("password")
        )
        # 이미 회원가입 된 유저일 때
        if user is not None:
            serializer = UserSerializer(user)
            # jwt 토큰 접근
            token = TokenObtainPairSerializer.get_token(user)
            refresh_token = str(token)
            access_token = str(token.access_token)
            res = Response(
                {
                    "user": serializer.data,
                    "message": "login success",
                    "token": {
                        "access": access_token,
                        "refresh": refresh_token,
                    },
                },
                status=status.HTTP_200_OK,
            )
            # jwt 토큰 => 쿠키에 저장
            res.set_cookie("username", user.username, httponly=False)
            res.set_cookie("access", access_token, httponly=False)
            user.refresh_token = refresh_token
            user.save()
            return res
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    # 로그아웃
    def delete(self, request):
        username = request.data.get("username")
        user = get_object_or_404(User, username=username)

        # 쿠키에 저장된 토큰 삭제 => 로그아웃 처리
        response = Response(
            {"message": f"{username} Logout success"}, status=status.HTTP_202_ACCEPTED
        )
        response.delete_cookie("access")
        # response.delete_cookie("refresh")
        user.refresh_token = ""
        user.save()
        return response
```

## 토큰 재발급
- 쿠키의 `access` 토큰을 디코딩하여 사용자를 인식
- 인식된 사용자의 **DB**에서 `refresh` 토큰을 꺼내어 재발급 요청
- 발급된 토큰을 다시 쿠키와 **DB**에 저장
- 예외 발생
  - `access` 토큰이 잘못되엇을 경우
  - `access` 토큰이 만료되었을 경우
  - `refresh` 토큰이 만료되었을 경우
- 오류 코드가 발생되면 프론트엔드에서 캐치하여 로그인 요청(리다이렉션)

```py
class TokenRefresh(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            # access token을 decode 해서 유저 id 추출 => 유저 식별
            access = request.COOKIES["access"]
            payload = jwt.decode(access, SECRET_KEY, algorithms=["HS256"])
            pk = payload.get("user_id")
            user = get_object_or_404(User, pk=pk)
            refresh_token = user.refresh_token
            data = {"refresh": refresh_token}
            serializer = TokenRefreshSerializer(data=data)

            # 유효성 검사 및 응답 직렬화
            if serializer.is_valid(raise_exception=True):
                access = serializer.data.get("access", None)
                serializer = UserSerializer(instance=user)

                # 새로운 access와 refresh 토큰으로 응답 생성
                res = Response(
                    {"access": access}, status=status.HTTP_200_OK
                )
                res.set_cookie("username", user.username, httponly=False)
                res.set_cookie("access", access, httponly=False)
                # res.set_cookie("refresh", refresh)
                return res

        except jwt.exceptions.InvalidTokenError as e:
            return Response({"detail": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
        except KeyError:
            return Response({"detail": "토큰을 확인할 수 없습니다."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"detail": "오류가 발생하였습니다. : " + str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

## `refresh` 토큰 보관 방법 트러블 슈팅
- 보안 강화를 위해 **방안 2** 선택
- `access` 토큰 탈취 방지를 위하여 만료 시간 축소 : 분단위 
  - 만료가 되기 전에 주기저으로 프론트엔드에서 재발급 요청
  - 타이머 등

### 공통사항

- 로그인
  - `access`, `refresh` 토큰 발급
  - `refresh` 토큰을 `DB`에 저장

- 로그아웃
  - 쿠키에서 `access` 토큰 삭제
  - `DB`에서 `refresh` 토큰 삭제

### 방안 1

- 쿠키
  - `access` 토큰
  - `refresh` 토큰

- `DB`
  - `refresh` 토큰

- 재발급
  - 쿠키와 `DB`의 `refresh` 토큰을 대조한 뒤에 `access` 토큰과 `refresh` 토큰을 재발급
  - 다른 기기에서 로그인하여 `DB`의 `refresh` 토큰이 갱신되었거나 
  - 로그아웃하여 `DB`의 `refresh` 토큰이 삭제된 경우 재로그인 요구

- 특징
  - `access` 토큰이 만료된 이후에 접근하여도 재발급 가능
  - `refresh` 토큰이 다른 곳에서 사용될 경우 확인 가능
  - `refresh` 토큰이 쿠키에 저장되어 탈취 가능성이 있음

### 방안 2

- 쿠키
- `access` 토큰만

- `DB`
- `refresh` 토큰만

- 재발급
- `access` 토큰을 디코딩하여 사용자를 인식한 후 DB에서 refresh 토큰을 가져옴
- 가져온 `refresh` 토큰으로 `access` 토큰 재발급

- 특징
- `access` 토큰만 쿠키에 저장하여 `refresh` 토큰을 안전하게 유지
- `access` 토큰이 만료될 경우 재로그인이 필요
	- 별도의 로직을 추가하여 해결 가능
		- 타이머를 사용하여 `access` 토큰 만료 전에 재발급
		- `API` 요청마다 `access` 토큰 재발급
		사용자가 별도의 동작을 하지 않으면 30분 후에 로그아웃
		동작을 하면 할 때마다 갱신