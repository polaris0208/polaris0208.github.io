---
layout: post
title: Hugging Face Transformers
subtitle: TIL Day 40
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, Tools]
author: polaris0208
---

# Hugging Face Transformers
> **Transformers** **NLP** 모델 제공, 오픈소스 라이브러리<br>
> **`BERT`, `GPT-2`, `DistilBERT`** 등<br>
[¶ Transformers](#transformers)<br>
[¶ Pre-training & Fine-tuning](#pre-traingfine-tuning)

# Transformers

## GPT-2
- **OpenAI** 개발 언어 생성 모델
  - 문장을 생성하거나 이어지는 텍스트를 예측
  - **GPT-2**는 **Transformers** 라이브러리에서 바로 사용 가능

### 텍스트 생성
- 호환성을 고려해 낮은 버전 사용
- 경고문은 코드가 정상적으로 작동될 때만 무시 설정
- `num_return_sequences`: 생성 개수
- 주피터 노트북으로 진행 할 경우 버전문제로 작동하지 않는 경우가 존재
- 모델을 불러와서 사용할 때는 별도의 IDE로 작성을 권장

```py
!pip install transformers==4.37.0

import warnings
warnings.filterwarnings(action ='ignore')

from transformers import pipeline

# GPT-2 기반 텍스트 생성 파이프라인 로드
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
result = generator("The Invisible Dragon barked", max_length=50, num_return_sequences=1)
print(result)
```
<br>
>'generated_text': 'The Invisible Dragon barked at me like a monkey trying to hide something! He was an adult, so I did my best. I was so afraid to bite him. This had me wondering if he would let me try his new toy.

### 감성어 분석
- 텍스트를 전달받아 긍정/중립/부정의 감성을 분석

```py
import warnings
warnings.filterwarnings(action ='ignore') 
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis")
result = sentiment_analysis("hate rodrigo")
print(result)
#
'label': 'NEGATIVE', 'score': 0.9987123012542725
```

## RoBERTa (Robustly Optimized BERT Approach)
- `BERT` 모델을 최적화
- 더 많은 데이터와 더 긴 학습 시간을 통해 성능을 향상
- 텍스트 분류와 감정 분석에 뛰어난 성능

```py 
import warnings
warnings.filterwarnings(action ='ignore') 
from transformers import pipeline

# RoBERTa 기반 감정 분석 파이프라인 로드
classifier = pipeline("sentiment-analysis", model="roberta-base")

# 감정 분석 실행
result = classifier("This product is amazing!")
print(result) 
```

- 결과:`'label': 'LABEL_1', 'score': 0.5604719519615173`
  - 해석할 수 없는 레이블과 수치
  - `BERT`모델의 특징에 기인 : 바로 사용할 수 있는 가중치가 없기 떄문에 파인튜닝 필요
    - 사용자가 제공하는 데이터에 맞춰 추가 학습이 필요
- 바로 사용 가능한 모델 : 완전 학습 모델

## Embedding
- 고차원 데이터를 저차원 공간으로 변화하는 기법
- 주로 벡터 표현으로 구현 : 숫자의 나열로 의미를 유지하면서 작은 차원의 공간에서 데이터를 표현
- **자연어 처리, 이미지 처리** : 비정형 데이터를 모델이 처리할 수 있는 수치화 데이터로 변환
- 신경망으로 통해서 학습: 의미적으로 유사한 단어가 가까운 저차원 공간에 배치되도록 학습
  - 유사한 문맥에서 자주 등장하는 패턴을 학습

### Word Embedding
- **Word2Vec, Fast, Text**
- 단어간의 유사도를 벡터공간에서 측정

### Sentence Embedding
- 문장간의 유사도 분석, 의미적 관계 분석

### Image Embedding
- 픽셀 데이터를 저차원 배열로 변환

### Graph Embedding
- 최근 주목
- 그래프 구조를 벡터로 변환
- 노드간의 관계를 벡터공간에서 표현 : 네트워크 분석, 추천 시스템

## 유사도
- 두 데이터의 비슷한 정도
- 벡터간의 각도나 거리를 측정
  - 코사인 유사도, 유클리디안 거리 등

### Word2Vec
- `vector_size` 매핑시킬 차원
- `window` 관계를 분석할 너비
- `min_count` 최소 등장 회수
- `sg` 학습할 알고리즘 - **Skipgram, CBOW**

```py
import warnings
warnings.filterwarnings(action ='ignore') 

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine

sentenses = ["The lion is often called the king of the jungle.",
             "Dolphins are known for their intelligence and playful behavior.",
             "Elephants are the largest land animals on Earth.",
             "Birds can fly thanks to their lightweight bones and strong wings.",
             "Many species of turtles can live for over a hundred years."
            ]

processed = [simple_preprocess(sentence) for sentence in sentenses]
print(processed)

model = Word2Vec(sentences = processed, vector_size= 5, window = 5, min_count=1, sg = 0)

Dolphins = model.wv['dolphins']
Elephants = model.wv['elephants']

sim = 1 - cosine(Dolphins,Elephants)
print(sim)
#
0.2764505147704356
```

### BERT 모델
- `torch.no_grad` : **Autograd** 정지
  - **Autograd ; Automatic gradient calculating API**
  - 해당 변수가 계산되는 데에 사용했던 모든 변수들의 미분값을 구하면서 forward 또는 backward를 진행

```py
import warnings
warnings.filterwarnings(action ='ignore') 

from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sentences = [
  "The king ruled his kingdom with wisdom and courage, earning the respect of his subjects.",
  "The queen held a grand banquet to celebrate the peace treaty, inviting nobles from far and wide"
             ]

input1 = tokenizer(sentences[0], return_tensors='pt')
input2 = tokenizer(sentences[1], return_tensors='pt')

with torch.no_grad(): # 오토그레드 비활성화
  output1 = model(**input1)
  output2 = model(**input2)

embedding1 = output1.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
embedding2 = output2.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

similarity = 1 - cosine(embedding1,embedding2)

print(f"Cosine similarity between the two sentences: {similarity:.4f}")
# Cosine similarity between the two sentences: 0.7213
```

[¶ Top](#hugging-face-transformers)

# Pre-traing/Fine-tuning

## Pre-training
### 개념
- 대규모의 텍스트 데이터셋을 사용해 모델이 **일반적인** 언어 이해 능력을 학습

### 특징
- 대규모 데이터셋 : 수십억 개 문장
- 일반적인 언어 이해 : 문장 구조, 문맥 등 언어의 전반적인 특징 학습
- 작업 비특화 : 특정 작업에 맞춰지지 않음

### 목적
- 언어의 기본적인 규칙 학습한 이후에 특정 작업에 빠르게 적응할 수 있도록 준비

### BERT 예시
#### Masked Language Modeling (MLM)
- 일부 단어를 마스킹(masking)한 후, 이를 예측하도록 학습

#### Next Sentence Prediction (NSP)
- 두 문장이 주어졌을 때, 문장이 자연스럽게 이어지는지를 예측
- 문장 간의 관계를 이해하는 능력을 학습

## Fine-tuning
### 개념
- 사전 학습된 모델을 특정 작업에 맞게 추가로 학습시키는 과정

### 특징
- 작업 특화 : 특정 작업에 맞춰 모델을 최적화
- 사전 학습 가중치 활용 : 작업에 맞게 일부 가중치만 조정
- 적은 데이터로도 가능: 사전 학습되었기 때문에 비교적 적은 데이터로도 효과적인 학습 가능

### 목적
- 특정 작업에서 최상의 성능을 발휘하도록 모델을 조정
- 사전 학습을 통해 더 빠르고 적은 데이터로도 수행 가능

### IMDb 데이터셋 예시 - BERT 모델

#### Fine-tuning 적용하지 않은 예시
- 평가 데이터만 사용
- 필요 패키지 설치

```bash
pip install transformers datasets torch
pip install --upgrade transformers accelerate
```

- 패키지 `import`

```py
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score
```

- 데이터셋 설정

```py
dataset = load_dataset('imdb') 
# 영화리뷰 감성 분석용 데이터셋(훈련/학습 데이터로 분할되어 있음)
# 훈련용 데이터에는 레이블 포함

test_dataset = dataset['test'].shuffle(seed=42).select(range(500))
```

- 토큰화 및 포맷 설정
- `batched` : 베치 단위로 함수 적용

```py
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# 영어 특화, 대소문자 구분하지 않음, 문장 토큰 512 제한

def tokenize_function(examples):
  return tokenizer(examples['text'], padding='max_length', truncation=True)
# 패딩은 최대 길이, 최대길이를 초과하면 잘라냄

test_dataset = test_dataset.map(tokenize_function, batched = True) 

test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
```

- 모델 정의 - 평가

```py
# 모델 정의
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels =2)

# 예측 및 평가
model.eval()

all_preds = []
all_labels = []

for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
  with torch.no_grad():
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
  logits = outputs.logits
  preds = np.argmax(logits.numpy(), axis=1)
  all_preds.extend(preds)
  all_labels.extend(batch['label'].numpy())

# 정확도 계산
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy without fine-tuning: {accuracy:.4f}')
```

#### Fine-tuning 적용 예시
- 학습용 데이터셋 추가 : 평가용 보다 많은 비율 할당
- 이외에는 동일하게 적용

```py
from transformers import BertTokenizer, BertForSequenceClassification
# pip install transformers datasets torch
# pip install --upgrade transformers accelerate

from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

dataset = load_dataset('imdb') 
# 영화리뷰 감성 분석용 데이터셋(훈련/학습 데이터로 분할되어 있음)
# 훈련용 데이터에는 레이블 포함
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))
test_dataset = dataset['test'].shuffle(seed=42).select(range(500))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples): 
  return tokenizer(examples['text'], padding = 'max_length', truncation = True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels =2)

# BERT 모델 로드
```
<br>

- `Trainer`, `TrainingArguments` 적용
  - **Transformers** 라이브러리에서 제공하는 클래스 - 모델훈련시 필요한 설정들을 관리

```py
from transformers import Trainer, TrainingArguments
```

- `TrainingArguments` : 파라미터
  - `output_dir` : 모델 파일, 로그 저장할 위치 
  - `num_train_epochs` : 에포크 수
  - `per_device_train_batch_size` : 베치 사이즈
  - `per_device_eval_batch_size`
  - `evaluation_strategy` : 평가 전략
  - `save_steps` : 해당 스탭마다 모델 저장
  - `save_total_limit` : 저장할 최대 체크 포인트

```py
# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results', # 모델 파일, 로그 저장할 위치 
    num_train_epochs=3, # 에포크 수
    per_device_train_batch_size=8, # 베치 사이즈
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch", # 평가 전략 = 에포크 종료마다 평가 진행
    save_steps=10_000, # 만 스탭마다 모델 저장
    save_total_limit=2,  # 저장할 최대 체크 포인트
)

# 트레이너 설정 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 모델 훈련
trainer.train()
trainer.evaluate()
```

- 결과 확인

```py
import numpy as np
from sklearn.metrics import accuracy_score

# 평가 지표 함수 정의
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # 예측된 클래스
    labels = p.label_ids  # 실제 레이블
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {'accuracy': acc}

# 이미 훈련된 트레이너에 compute_metrics를 추가하여 평가
trainer.compute_metrics = compute_metrics

# 모델 평가 및 정확도 확인
eval_result = trainer.evaluate()
print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
```

[¶ Top](#hugging-face-transformers)