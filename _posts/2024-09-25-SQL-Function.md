---
layout: post
title: SQL 기초 기능
subtitle: TIL Day 3
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, SQL]
author: polaris0208
---
# SQL 기능

## 복잡한 연산

### 복습
* 조건문
 replace - 문자 변경
 substring - 문자 추출
 concat -문자 합성
 if(조건, 충족, 충족x)
 case when then else end
 
### subquery
* 여러번의 연산을 수행(연산결과를 다른 연산에 활용 등)
* ()사용
* subquery를 이용하여 단계별 연산을 수행
* 가장 안쪽에 들어갈 조건 부터 작성
* 별명 설정을 해야 작성이 용이

### join
* 서로 다른 테이블에 있는 데이터 조회
* 공통 컬럼을 기준으로 묶기
* left join 하나의 테이블과 두 테이블이 공통되는 부분 조회
* inner join 공통되는 부분만 조회
_ex) 테이블1 left join 테이블2 on 테이블1.공통컬럼=테이블2.공통칼럼_
* 조회할 컬럼명에도 출처를 밝혀 적는다 ex)테이블1.컬럼
* 중복값이 많을 수 있기 때문에 distinct 활용

### 오류 대처
* 제외하고 싶은 구문에 null 활용-사용할 수 없는 데이터가 포함되면 결과값이 달라질 수 있기 때문에 중요
* is not null - 빈 데이터 제외
* 다른 값으로 대체 - 조건문 사용 또는 coalesce(컬럼, 대체값)
* 상식적이지 않는 값 제거를 위해 조건문으로 범위 지정

### pivot table
* case 또는 if 조건문을 활용
* 행축에 갈 컬럼을 가장 먼저 작성
* 열에 들어갈 컬럼은 max, sum 등의 집계함수를 통해 묶어준다 (확인필요)

### window function
* rank() over (partition by 구분 기준 order by 순위 기준)
* sum() over (partition by)

### 날짜 데이터
* 날짜 형식의 데이터는 시계 표시
* 날짜 형식의 데이터는 연산 가능
* date_format(컬럼, '%Y') "년" - 년월일 붙여서도 가능
* datediff(뒤, 앞)
* date(now()) 현재 날짜