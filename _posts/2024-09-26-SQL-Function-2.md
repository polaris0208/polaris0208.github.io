---
layout: post
title: SQL 기초 기능-2
subtitle: TIL Day 4
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, SQL]
author: polaris0208
---
# SQL 연습

## 기억할 만한 요소


```c
select user_name, cnt, rnk
from
(
select user_name, count(feedback_date) cnt, DENSE_RANK() over(order by count(feedback_date) desc) rnk
from lol_feedbacks
group by user_name
) a
where rnk<=3

```

### rank 함수 활용
* rank - 중복 순위 부여 후, 중복 개수를 고려하여 다음 순위 부여 ex) 1, 2, 2, 2, 5
* dense_ranke - - 중복 순위 부여 후, 다음 순위는 중복순위를 제외하고 부여 ex) 1, 2, 2, 3, 4
* row_number - 중복된 값에도 순위 부여
* 해당 문제에서는 dense_rank를 사용하였기 때문에 limt 3 보다 순위 값이 3 이하라는 조건 사용

```c
select feedback_date
from 
(select feedback_date, avg(satisfaction_score) avg
from lol_feedbacks
group by feedback_date) a
where avg = (
select max(avg)
from
(select feedback_date, avg(satisfaction_score) avg
from lol_feedbacks
group by feedback_date) a
)
```

### 연산된 값을 활용하여 연산할 경우 subquery를 활용

```c
select count(name)
from
(
select name, birth_date, datediff(NOW(), date(birth_date)) d
from patients
) a
where TRUNCATE(d / 365, 0) >= 40
```
### TRUNCATE(숫자, 자리수) 자리수 아래로 버림
* round( 숫자, 자리수) 반올림 0 = 소수점 첫째 자리, 1 = 둘째, -1 = 정수 첫째 자리

```c
select count(name)
from
(
select name, birth_date, date_format(birth_date, '%y') y
from patients
) a 
where substr(y, 1, 1) = '8'

```

### date_format(컬럼, 형식), substr(컬럼, 시작문자, 추출할 문자 수) 활용
