---
layout: post
title: Tokenizer
subtitle: TIL Day 31
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, NLP]
author: polaris0208
---

# Tokenizer
> NLTK, torchtext
>> tokenize: corpus를 token 단위로 나누는 작업
>>> corpus(복수형: corpora): 자연언어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합

## 데이터 전처리
### 데이터셋 불러오기
- `NLTK` - **Natural Language Tookit**

```py
import pandas as pd # 
import nltk
import re

df = pd.read_csv("/Users/사용자이름/netflix_reviews.csv") 
```

### 문장부호, 숫자 제거, 소문자 처리

>The only reason I didn't give it four stars is that In my opinion, there are too many foreign films. When I'm done working hard all day. I don't want to have to read subtitles. UPDATE: 10/19/24

- `cleaned = re.sub('[^a-zA-Z]', ' ', sample)` : 알파벳이 아닌 경우 공백 처리
- `cleaned.lower()` : 소문자화

>the only reason i didn t give it four stars is that in my opinion  there are too many foreign films  when i m done working hard all day  i don t want to have to read subtitles  update

### 불용어 처리
- 불용어: 자주 등장하지만 실제 의미 분석에는 기여가 없는 단어
- `from nltk.corpus import stopwords`: 불용어 목록 불러오기
- `nltk.download('stopwords')`: **NLTK** 는 패키지 설치시 모든 데이터를 다운로드하지 않음, 모듈마다 필요한 데이터를 따로 다운로드
- `eng_stopwords = stopwords.words('english')`: 영어 불용어 목록 불러오기
> ['i',
 'me',
 'my',
 ...
 "won't",
 'wouldn',
 "wouldn't"]
- `sample.split()`: 공백을 기준으로 단어를 끊어서 나열, 불용어 목록과 대조하기 위해 사용
> ['The',
 'only',
 'reason',
 ...
 'BEGINNING!!!!!!',
 "I'M",
 'DONE']

### 문장 전처리 함수화
- `isinstance` : 객체의 타입 확인
- `' '.join()` : 문자열 하나로 합치기

```py
def preprocessing(sentence):
  if isinstance(sentence, float): return '' 
  cleaned = re.sub('[^a-zA-Z]', ' ', sentence)
  cleaned = cleaned.lower()
  cleaned = [word for word in cleaned.split() if word not in eng_stopwords ] 
  # 불용어 목록에 없는 단어만 반환
  return ' '.join(cleaned)
  # .split() 처리한 것 다시 붙여주기
```

### 데이터셋 전체 적용
- `.apply()` : 함수 적용

```py
reviews = df['content']
cleaned_reviews = reviews.apply(preprocessing)
cleaned_reviews.head(10)
```

> 0                                                 open
  1                                             best app
  2    famous korean drama dubbed hindi sense paying ...
  3    superb please add comments section us like you...
  4    reason give four stars opinion many foreign fi...
  5                                              amazing
  6                                       pure greatness
  7                                                 good
  8                                   experiencing error
  9                   anti indian propoganda filler fool
  Name: content, dtype: object

### 표제어 추출
- **NLTK** 모듈 사용

```py
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
# 단어의 형태 통일 - 동사의 원형 등
```
- 명사 또는 동사의 원형 추출

```py
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('runs')
lemmatizer.lemmatize('runs', 'v')
#
'run'
```

- 함수 선언

```py
def process_lemma(sentence):
    return [lemmatizer.lemmatize(word, 'v') for word in sentence]
```

### 문장 전처리 모듈화

```py
def preprocessing(sentence):
  if isinstance(sentence, float): return ''
  cleaned = re.sub('[^a-zA-Z]', ' ', sentence)
  cleaned = cleaned.lower()
  cleaned = cleaned.strip() # 띄어쓰기 제외 공백 제거
  cleaned = [word for word in cleaned.split() if word not in eng_stopwords ]
  cleaned = process_lemma(cleaned)
  return ' '.join(cleaned) # 공백 제거 ' ' -> ''
```

> 0                                                 open
  1                                             best app
  2    famous korean drama dub hindi sense pay subscr...
  3    superb please add comment section us like youtube
  4    reason give four star opinion many foreign fil...
  5                                                amaze
  6                                       pure greatness
  7                                                 good
  8                                     experience error
  9                   anti indian propoganda filler fool
  Name: content, dtype: object

## Tokenize

### torchtext
### 모듈 `import`
- `import torchtext; torchtext.disable_torchtext_deprecation_warning()`: "torchtext의 마지막 버전 경고" 무시
- `from torchtext.data.utils import get_tokenizer`
- `from torchtext.vocab import build_vocab_from_iterator`
  - iterator를 이용하여 vocab(단어장) 생성

  
### tokenize

  ```py
  words = "Betty Botter bought bit of Bitter Butter"
  tokenizer = get_tokenizer('basic_english')
  tokenizer(words)
  # 
  ['betty', 'botter', 'bought', 'bit', 'of', 'bitter', 'butter']
  ```

### vocab
- torchtext의 vocab 클래스의 object: 단어 집합
- parametrt
  - 사용할 iterator
  - min_freq: 최소 빈도
  - specials: special token(별도 처리)의 list
- 기능
get_stoi(): str : index 반환
get_itos(): str 반환

```py
vocab = build_vocab_from_iterator(tokenizer(words), specials=['<unk>'])
vocab.get_stoi() 
#
{'o': 4,
 'h': 10,
 'i': 6,
 'f': 8,
 'b': 2,
 'u': 7,
 'e': 3,
 'y': 11,
 'g': 9,
 't': 1,
 'r': 5,
 '<unk>': 0}
 ```

### tokenize iterator
#### generator 사용
- 문장이 주어질 때마다 토큰 반환

```py
def yield_tokens(sentences):
    for text in sentences:
        yield tokenizer(text)
```

#### vocab 생성

```py
vocab = build_vocab_from_iterator(yield_tokens(df['reviews'].tolist()), 
                                  specials=['<UNK>'],   # 스페셜 토큰
                                  min_freq=2,           # 최소 빈도 토큰
                                  max_tokens=1000,      # 최대 토큰 개수
                                 )
# string -> index
stoi = vocab.get_stoi()
# index -> string
itos = vocab.get_itos()
```

### tensorflow

### tokenizer 세팅
- `from tensorflow.keras.preprocessing.text import Tokenizer`

```py
tokenizer = Tokenizer(oov_token='<OOV>') # 없는 데이터 치환
tokenizer.fit_on_texts(cleaned_reviews) # 단어집 생성
len(tokenizer.word_index) # 단어집 개수 33014
```

> netflix 2
  app 3
  watch 4
  show 5
  movies 6
  good 7
  like 8
  get 9
  use 10
  love 11
  work 12
  please 13
  time 14
  update 15
  great 16
  phone 17
  download 18
  even 19
  try 20
  go 21
