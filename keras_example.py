#Keras Functional API
#https://keras.io/getting-started/functional-api-guide/

#keras도 결국에는 그냥 python library을 잊지말아라

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',#loss와 optimizer은 필수
              metrics=['accuracy'])
model.fit(data, labels)  # starts training #트레이닝 시작!

#훈련된 모델을 재사용하기 쉽다.
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)#20줄에 훈련시킨 model을 말함, y는 model의 predictions(12줄)

#이미지 input일 때 one line으로 해결된다면, 이미지의 연속인 비디오또한 one line으로 해결된다는 말이다.
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))
#행렬의 row, col이 뭐 가리키는 것인지 알 수 있어야 한다.

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 ectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
#21줄에서 훈련시킨 model, row하나가 이미지 하나로 생각, 20개의 이미지, 비디오라고 볼 수 있다.

###Multi-input and multi-output models###

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')
#input demension, 아무리 길어봐야 input length 100개 이다.
#main input으로 받을 것이냐

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#word embedding, 사전에 있는 단어가 몇 개이냐

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
# Embedding을 RNN(LSTM)에 넣을 것이다.
# 32개짜리 cell을 갖는다. 32개짜리 벡터 1개
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#logistic regression을 만들 때 보통 쓰는 것

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
#연결해줌
#5+32=37짜리 벡터

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)