---
layout: post
title: Python OOP 기능
subtitle: TIL Day 9
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Python]
author: polaris0208
---
# Python

## Python 기호
* " * " - 모든 것
* 가변인자 - *args, **kwargs
* -> - 함수 정의
* ... - 생략부호, pass 처럼 사용 가능
* % - 문자열 포메팅 - %s 문자, %d 정수, %f 소수
```python
fruits = ["사과", "딸기", "배"]
cnt = 1
for fruit in fruits:
  print (" %s %d" %(fruit, cnt), "개")

# "%타입" %(타입인수) 구조
#  인수가 한개인 경우 ()생략
  
# 사과 1 개 딸기 1 개 배 1 개

```
* @ - decorator 기능

## Decorator
* 함수를 수정하지 않고 기능을 추가
* 함수를 감싸는 형태 -> wrapper

```python
def deco(origin):
  def wrapper(): # wrapper 함수로 원본 함수를 감싸 내용추가
    print("비오네☔️🌧️...")
    origin()
    print("징징징🎸🎸🎸~ ")
  return wrapper # wrapper 함수를 반환

@deco
def origin():
  print("IU - Bye Summer")

origin()

# 
비오네☔️🌧️...
IU - Bye Summer
징징징🎸🎸🎸~ 

```
## Iterator
* 반복 가능한 객체를 하나씩 꺼냄
```python
numbers = [1,2,3,4,5] #리스트
iterator = iter(numbers) 
next(iterator) #next() 메서드로 호출
# 1
next(iterator)
# 2

```

* class 적용 예
```python
class myiterator:
  def __init__(self, data):
    self.data = data
    self.index = 0 #.index = 원소의 위치

  def __iter__(self):
    return self
  
  def __next__(self):
    if self.index < len(self.data):
      result = self.data[self.index]
      self.index += 1
      return result
    else:
      raise StopIteration 
      #raise : 에외 설정(버그 대비) stopIteration - 정지
      
 my_iter = myiterator([1,2,3])
 for a in my_iter:
  print(a)
 # 1 2 3
```
## Generator
 * generator는 iterator를 생성
 * 단 모든 값을 한번에 생성하지 않고 필요할 때 생성
 * yield 키워드를 사용하여 값을 하나씩 변환
 
```python
[1,2,3,4,5]

 def generate_5(): 
  yield 1 
  yield 2
  yield 3
  yield 4
  yield 5
  
gen = generate_5()
next(gen)
# 1
```
* 피보나치 수열 만들기
```python
def fibonacci(n):
  a, b = 0, 1
  for _ in range(n): 
  #의미 없는 변수 _ 사용 
  https://dkswnkk.tistory.com/216
    yield a #generator
    a, b = b, a + b
    
for num in fibonacci(10):
  print(num)

# 0 1 1 2 3 5 8 13 21 34
```
## 파일 다루기
* f : file
* r : 읽기 / w : 쓰기 / a : 추가하기
```python
f = open("/Users/유저명/bye_summer.txt", "r")
# 우클릭 - alt - 경로 이름 복사
line = f.readline() # 한줄만
print(line)
f.close # 파일 닫기

# while True: 
  line = f.readline()
  if not line: break
  print(line)
f.close
# 비오네☔️🌧️... 
```
* 한 줄씩 전체
```python
while True: # 값이 True인 동안 반복
  line = f.readline() # 한 줄씩
  if not line: break
  print(line)
f.close

#비오네☔️🌧️... ~(생략) 서늘한 바람이.. 💨💨💨🌪️🌬️

```
* 파일 전체(줄)
```python
for line in f:
  print(line)
f.close
```
* 여러줄
```python
lines = f.readlines() #readline 아닌 lines
for line in lines: # 전체
  line = line.strip() #.strip 공백없이
  print(line)
f.close
```
* 파일 전체 읽기
```python
data = f.read() #f.read() 읽기
print(data)
f.close
```
* 내용 추가하기 (context manager)
```python
with open("/Users/유저명/bye_summer.txt", "a") as f:
# with 구문을 사요하면 f.close 생략 가능
  f.write("IU - Bye Summer") # f.write() 쓰기
  
# 다시 데이터를 읽으면 마지막 부분에 "IU - Bye Summer" 추가됨
```

## OOP

### class 는 object를 만들기 위한 template 즉 설계도

```python
class 클래스:
	def __init__(self, 속성A, 속성B):
    	self.속성A = 속성A,
        self.속성B = 속성B
        
인스터스 = 클래스("A", "B")

# 클래스라는 틀을 이용해 속성A, 
# 속성B에 대응하는 속성, A, B를 가지는 객체 생성
```
### magic method
* init 생성자, 초기화
* repr 공식적인 문자열 반환
```
# 그대로 사용했을 때 동일한 객체 생성하는 문자열을 반환해야한다.
# 주로 디버깅에 상용
```
* add 객체간의 덧셈(문자열도 가능)
* eq 두 객체가 같은지 비교
* str  비공식저인 문자열 반환

```
# 보여지기 위한 문자열
# 사용자가 보기 좋은 문자열
```
### class method
* 클래스 단위로 사용

```python
calss myclass:
	class_variable = 0
    
    @classmethod 
    def increment(cls): 
    cls.class_varible += 1
    
myclass.increment()
print(myclass.classvariable)

# 1 
# 한번 더 실행하면 2

a = myclass
a.class_variable

# 2
# 클래스 내부에서 값이 변경 되었기 때문에 생성된 객체인 a에서도 값을 공유
```

### static method
* 정적 메서드
```python
# 클래스나 객체의 상태와 상관없이 정의
clss utility:
	@staticmethod
    def add(x, y)
    	return x + y
result = utility.add(2, 3)
print(result)
# 5
```
### 상속(inheritance)
* 부모 클래스에게서 속성과 메서드를 물려받아 공유
```python
class animal: # 상위 클래스 생성
  def __init__ (self, name): # 속성부여
    self.name = name

  def speak(self): # 메서드 생성
    return "소리를 냅니다."
    
class dog(animal): # 하위 클래스 생성
  def speak(self): # 상속받은 메서드 사용
    return f"{self.name}가 멍멍 짖습니다."
  # 오버라이드 하여 새로운 기능 추가
  
my_dog = dog("Brandon")
print(my_dog.speak())

#Brandon가 멍멍 짖습니다.
```
* 부모 클래스 초기화
```python
super().__init__()

부모 클래스의 매서드를 자식 클래스에서 실행

class dog(animal):
  def __init__(self, name, age):
    super().__init__(name) #animal의 메서드
    self.age = age
  def speak(self): 
    return f"{self.name}가 멍멍 짖습니다."
  
my_dog = dog("Brandon", 0.1)
print(my_dog.speak()
```
