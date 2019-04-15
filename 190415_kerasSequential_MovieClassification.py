from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels)=imdb.load_data(num_words=10000)#10000개의 data를 load
#각각의 데이터와 부정/긍정을 의미하는 label

#형태를 변경, numpy로, 데아터 전처리
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
	results=np.zeros((len(sequences), dimension))#result를 담을 행렬을 만든다.
	for i, sequence in enumerate(sequences):
		results[i, sequence]=1.
	return results
x_train=vectorize_sequences(train_data)#벡터로 변경
x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#신경망 모델 만들기

#표현 공간의 차원(layer의 개수?)을 신경망이 내재된 표현을 학습할 때 가질 수 있는 자유도로 이해할 수 있다.
#은닉 유닛을 늘리면 신경망이 더욱 복잡한 표현을 학습할 수 있다.

#층을 쌓을 떄 1. 얼마나 많은 층을 사용할 것인가? 2. 각 층에 얼마나 많은 은닉 유닛을 둘 것인가?
#여기서는 16개의 은닉 유닛을 가진 2개의 은닉 layer
#예측을 출력하는 세번째 layer

#relu와 sigmoid: relu값은 음수 값을 0으로 만들고, sigmoid는 임의의 값을 0, 1사이로 압축한다. 
#예측을 위해서는 relu를 사용하되, 확률값을 낼 때는 sigmoid를 이용하는 것이 더 좋을 수 도 있다.
#따라서 은닉 층은 relu를 사용하고 마지막 층만 sigmoid를 이용한다.

from keras import models
from keras import layers
#Sequential model을 사용하였다.
model=models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

'''
#functional API 이용

from keras.layers import Input, Dense
from keras.models import model

input=Input()

# a layer instance is callable on a tensor, and returns a tensor
x= Dense(16, activation='relu')(input)
y=Dense(16, activation='relu')(x)
predictions=Dense(1, activation='sigmoid')(x)

'''

#확률을 출력하는 모델을 사용할 때는 크로스엔트로피가 최선의 선택이다.
#크로스엔트로피는 확률 분포간의 차이를 측정한다. 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

'''
#functional API

model=Model(inputs=input, outputs=predictions)
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['accuracy'])
'''

'''
#optimizer설정하기

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
				loss='binary_crossentropy',
				metrics=['accuracy'])
'''

#훈련 검증

x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]

#모델 훈련

history=model.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val, y_val))

# 훈련과 검증 손실 그리기

import matplotlib.pyplot as plt

history_dict=history.history
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs=range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 훈련의 정확도
results=model.evaluate(x_test, y_test)

# 휸련된 모델로 새로운 데이터에 대해 예측하기

model.predict(x_test)
#확률 예측 결과 보여준다. 각 배열의 값에 대해서

