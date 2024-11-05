---
layout: post
title: 시퀀스 데이터 변환
subtitle: TIL Day 32
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, NLP]
author: polaris0208
---

## Text to Sequence
> 문자열 데이터를 모델 학습이 가능한 sequence 데이터로 변환
>> tokenize 전처리 필요

### tokenize 요약
- 분리된 문자열 단어들에 인덱스를 부여해서 정수형 데이터로 변환
`torchtext`

```py
from torchtext.data.utils import get_tokenizer 
from torchtext.vocab import build_vocab_from_iterator 

tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter): # iteratable한 문자열 데이터를 변환
  for text in data_iter:
    yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>']) # 없는 단어 치환
```

`tensorflow`

```py
from tensorflow.keras.preprocessing.text import Tokenizer
 
tokenizer = Tokenizer(oov_token='<OOV>') # 없는 데이터 치환
tokenizer.fit_on_texts(data) # 단어집 생성
len(tokenizer.word_index) # 단어집 개수
```

### Sequence data로 변환
`tensorflow`
- 문자열을 토큰화 한 뒤 sequence 데이터로 변환

```py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_sq = tokenizer.texts_to_sequences(data)
```

`torchtext`
- 토큰화와 데이터 변환을 따로 진행
- `torch.tensor(data)` : `torch.Tensor()`와는 다름
  - 입력된 데이터를 tensor 형태로 변환
  - 데이터가 변형될 가능성이 있음
- `torch.Tensor()` : 클래스 선언, 아무것도 입력하지 않아도 빈 tensor 데이터를 를 생성함

### Padding
- 데이터 길이를 맞춰 주는 작업
- 데이터 길이가 동일해야 학습에 사용 가능

`tensorflow`
- sequence 데이터를 보완
- MAX_LENGTH : 최대 문장의 길이
- TRUNC :넘칠 경우 자르기 / 앞부분 = pre, 뒷부분 = post
- PAD :모자랄 경우 채우기(0) / 위와 같음

```py
data = pad_sequences(data, maxlen= MAX_LENGTH, truncating = TRUNC, padding = PAD)
```

`torchtext`
- `batch_first=True`: 첫번째 배치의 크기를 따라감
- 최대크기를 지정
- 최대 크기와 비교
- 초과할 경우 최대크기 까지의 내용으로 선언
- 부족할 경우 길이의 차이만큼 0을 추가

```py
from torch.nn.utils.rnn import pad_sequence

max_length = 50
numericalized_data = []

for text in data:
    indices = [vocab[token] for token in tokenizer(text)]
    if len(indices) > max_length:
        # 길이가 초과할 경우 뒷부분 잘라내기
        indices = indices[:max_length]
    elif len(indices) < max_length:
        # 길이가 모자랄 경우 0으로 패딩하기
        indices += [0] * (max_length - len(indices))
    numericalized_data.append(torch.tensor(indices))

# 시퀀스 패딩 (이미 길이가 맞춰졌으므로 패딩 필요 없음)
padded_data = pad_sequence(numericalized_data, batch_first=True)

# NumPy 배열로 변환
data = padded_data.numpy()
```

### 활용
- `tensorflow` 사용 : 메서드로 쉽게 사용 가능
- 미리 제작한 전처리 함수와 결합
- **text_pipeline** : 자연어 학습에 필용한 전처리를 한번에 진행

```py
   def preprocessing(sentence):
        if isinstance(sentence, float): return ''
        cleaned = re.sub('[^a-zA-Z]', ' ', sentence) # 알파벳 외에 공백 처리
        cleaned = cleaned.lower() # 소문자 처리
        cleaned = cleaned.strip() # 띄어쓰기 외에 공백 제거
        cleaned = [word for word in cleaned.split() if word not in eng_stopwords ]
        # 영어 불용어 제거
        cleaned = process_lemma(cleaned) # 동사 원형 처리
        return ' '.join(cleaned) # 공백 제거 ' ' -> ''
        # .split() 처리되어 분리된 문자열 다시 합치기

    def text_pipeline(data):
        processed_data = preprocessing(data)
        sequence = tokenizer.texts_to_sequences([processed_review])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
        return padded_sequence[0] 
```
