---
layout: post
title: Test Framework
subtitle: TIL Day 76
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

# unittest와 pytest
> **Python**에서 테스트를 작성하고 실행하는 두 가지 주요 테스트 프레임워크

## 기본 기능

### unittest
- 내장 라이브러리로
- `TestCase` 클래스에서 상속을 받아 테스트를 정의
  - 객체 지향적인 접근 방식
- 다양한 `assertion` 메서드 제공
  - `assertEqual, assertTrue, assertFalse` 등
- `setUp`과 `tearDown` 메서드를 사용
  - 테스트 전후에 필요한 설정/정리

### pytest
- 외부 라이브러리로 설치가 필요
  - `pip install pytest`
- 함수 기반의 간단하고 직관적인 테스트 작성 방식
- **Python**의 기본 `assert` 문법을 그대로 사용
- `unittest`보다 더 많은 플러그인과 기능을 제공
  -  마크, 파라미터화 테스트 등
- `pytest discover`테스트를 자동으로 찾아서 실행할 수 있는 기능

## 작성 방식

### unittest
- 테스트 케이스는 `unittest.TestCase`를 상속받은 클래스 내에 작성
- 각 테스트 메서드는 `test_`로 시작

```py
import unittest

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(2 + 3, 5)

    def test_subtract(self):
        self.assertEqual(5 - 3, 2)

if __name__ == "__main__":
    unittest.main()
```

### pytest
- 함수 기반으로 작성되며, 함수명은 `test_`로 시작
- `assert` 문을 직접 사용하여 테스트를 작성

```py
def test_add():
    assert 2 + 3 == 5

def test_subtract():
    assert 5 - 3 == 2
```

## 설정 및 정리 SetUp/TearDown

### unittest
- 테스트 전후에 `setUp()`과 `tearDown()` 메서드를 정의하여 필요한 설정과 정리

```py
class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.x = 2
        self.y = 3

    def test_add(self):
        self.assertEqual(self.x + self.y, 5)

    def tearDown(self):
        self.x = None
        self.y = None
```

### pytest
- `fixture`를 사용하여 설정 및 정리
  - 함수, 클래스, 모듈 범위 등을 지원
  - `yield`를 사용하여 정리 코드를 작성

```py
import pytest

@pytest.fixture
def setup_data():
    x = 2
    y = 3
    yield x, y
    # 테어다운 코드 (필요시)

def test_add(setup_data):
    x, y = setup_data
    assert x + y == 5
```

## 동작 방식

### unittest
- `unittest.main()`을 호출하여 테스트를 실행
- 테스트가 실패하면 전체 테스트가 멈추지 않고 계속 실행

### pytest
- `pytest` 명령어를 사용하여 실행
- 실패한 테스트만 보고할 수 있으며, 전체 테스트가 멈추지 않음
- `assert` 문을 사용하여 실패한 경우, 오류 메시지를 자동으로 제공

## 유용한 기능

### unittest
- 다양한 `assertion` 메서드 제공
  - `assertEqual, assertNotEqual, assertTrue` 등
- 다양한 기능 제공 
  - `setUp, tearDown, skipTest, expectedFailure` 등

### pytest
- 오류 메시지를 자동으로 생성
- 마크 기능을 통해 특정 테스트를 선택적으로 실행
  -  `@pytest.mark.parametrize`
- 파라미터화된 테스트 기능
- 다양한 플러그인과 확장 기능을 지원
  - `pytest-cov`로 커버리지 측정
- 테스트 디스커버리 기능
  - 파일 및 폴더를 자동으로 탐색하고 실행

## 테스트 실행 및 결과 확인

### unittest
- `python -m unittest` 명령어를 사용하여 실행
- 실패한 테스트는 `FAILED`로 표시
- `python -m unittest test_calculator.py`

### pytest
- `pytest` 명령어를 사용하여 실행
- 실패한 테스트에 대한 자세한 정보와 함께 자동으로 오류 메시지를 제공
- `pytest -v`
  - `-v` : 자세한 설명 옵션