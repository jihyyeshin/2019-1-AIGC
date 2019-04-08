'''
keras에는 세가지 모델이 있다.
1. the sequaltial model
2. the functional API
3. model subclassing#우리의 프로젝트에서는 거의 안쓴다.
'''

#getting started 

#sequential모델
from keras.models import Sequential
model=Sequential()

#layer을 생성
from keras.layers import Dense
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

#compile을 통해 학습을 시작
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#compile할 때 optimizer을 쓸 수도 있다.

#학습된 모델에 집어넣음
model.fit(x_train, y_train, epochs=5, batch_size=32)

#정확도 평가
loss_and_metrics=model.evaluate(x_test, y_text, batch_size=128)

#예측
classes=model.predict(x_text, batch_size=128)


#getting started 2 - specifying the input shape

from keras.models import Sequential
from keras.layers import Dense, Activation

model=Sequential([
	Dense(32, input_shape=(784,)),
	Activation('relu'),
	Dense(10),
	Activation('softmax'),
])

#모델을 추가하기 위해서는 add메소드를 사옹하면 된다
model=Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

'''
모델은 이것이 예측해야 하는 input shape에 대해 알고 있어야 한다.
따라서 sequential model의 첫번째 모델(첫번쨰 모델만, 왜냐하면 두번째 레이어부터는 
자동적인 추론)은 input shape에 대한 정보를 받아야 한다.
1. 첫번쨰 layer을 위한 input_shape 매개변수를 가져가야 한다. 
2. 2D layer의 일부(Dense)에서는 그들만의 input shape를 가지고 있다. 
이 때 input_dim이라는 매개변수를 사용하고, 3D는 input_dim, input_length를 이용한다.
3. input을 위한 batch size가 고정될 필요가 있다면, (stateful recurrent network에서 사용된다.)
batch_size매개변수를 보낸다. batch_size=32, input_shape(6, 8)을 layer에 둘 다 보내면, 
batch shape가 (32, 6, 8)임
'''

model=Sequential()
model.add(Dense(32, input_shape=(784, )))
#이 두개의 코드는 동일함.
model=Sequential()
model.add(Dense(32, input_dim=784))
#sequential모델을 만든 후에, 2D layer인 2D에서 input_shape를 784로 하나, input_dim=784로 하나 동일함.
#2D모델이기 때문에?! 둘 다 동일한 역할!?

#getting started - Compilation

'''
모델을 트레이닝하기 전에, "learning" 과정을 알아야 한다.
compile이라는 메소드를 통해 이 작업이 이루어지고,

1. optimizer : 기존 optimizer(ex. rmsprop, adagrad)의 문자열 식별자 또는 Optimizer클래스의 인스턴스
2. loss function : loss를 최소화하는 것이 목표이다. 기존 loss function(categorical_crossentropy or mse)의 문자열 식별자이거나 
목표 함수일 수 있다
3. a list of metrics : classification 문제에서, metrics=['accuracy']로 표현된다.
metrics는 기존 metrics의 문자열 식별자이거나 사용자 지정 metrics function.

이렇게 3가지의 매개변수를 가진다.
'''

#for a multi-class classification problem
model.compile(optimizer='rmsprop', 
			loss='categorical_crossentropy', 
			metrics=['accuracy'])
#for a binary classification problem
model.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['accuracy'])
#for a mean squared error regression problem
#classification problem이 아니기 때문에 metrics가 없다.
model.compile(optimizer='rmsprop',
			loss='mse')

#For custom metrics
#metric를 사용자 지정을 통해 지정해준다.
import keras.backend as K

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

model.compile(optimizer='rmsprop'
			loss='binary_crossentropy',
			metrics=['accuracy', mean_pred])

#getting started - Training

'''
keras 모델은 Numpy 배열에서 학습된다(input data, label).
모델을 학습시키기 위해서 fit function을 쓴다.
'''

# for a single-input model with 2 classes (binary classification):

model = Sequential()#sequential model 생성
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))#모델을 추가.
model.compile(optimizer='rmsprop',#학습 시작
				loss='binary_crossentropy',
				metrics=['accuracy'])#classification이다.

# Generate dummy data
import numpy as np
data=np.random.random((1000, 100))
label=np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
#data, label은 train을 위한 data

# for a single-input model with 10 classes(categorical classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

# Generate dummy data
import numpy as np
data=np.random.random((1000, 100))
labels=np.random.randint(10, size=(1000, 1))#10개

# Convert labels to categorical one-hot encoding
one_hot_labels=keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on th data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)


# Multilayer Perceptro (MLP) for multi-class softmax classification:
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizer import SGD

#Generate dummy data
import numpy as np
x_train=np.random.random((1000, 20))
y_train=keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test=np.random.random((100, 20))
y_test=keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model=Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))#첫번쨰 layer에서 input_dim을 지정해주었다.
model.add(Dropout(0.5))#dropout : overfitting을 방지하기 위해 지정된 비율만큼 임의의 입력 뉴런을 제외시킨다.
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd=SGC(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
			optimizer=sgd,
			metrics=['accuracy'])

model.fit(x_train, y_train,
		epochs=20,
		batch_size=128)

score=model.evaluate(x_text, y_text, batch_size=128)

'''
Keras functional API 사용
# keras functional API는 복잡한 모델을 정의하기 위하여 사용된다. 
ex. multi-output model, directed acyclic graphs, or models with shared layers
'''

'''
First example - densely-connected network

이 예시는 sequential 모델로 하는 것이 더 좋은 선택일 수도 있지만,
functional api로도 표현할 수 있다.
- layer 객체는 텐서에서 불리울 수 있고, tensor return을 갖는다.
- input, output tensor은 Mode을 정의하기 위해 쓰인다.
- sequential model이랑 model train은 비슷하다.
'''

from keras.layers import Input, Dense
from keras.models import model

# This returns a tensor
input=Input(shape=(784, ))

# a layer instance is callable on a tensor, and returns a tensor
x= Dense(64, activation='relu')(inputs)
y=Dense(64, activation='relu')(x)
predictions=Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model=Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
model.fit(data, labels) # start training

# all models are callable, just like layers

'''
functional api를 이용하면, 훈련된 모델을 재사용하는 것이 쉽다.
model들을 layer처럼 취급할 수 있다. tensor에서 부르면.
구조만 재사용하는 것이 아닌, weights도 재사용하게 된다.
'''

x=Input(shape=(784, ))
# This works, and return the 10-way softmax we defined above
y=model(x)

'''
이것은 빠르게 model을 만들고 input의 sequence를 프로세스할 수 있다.
예를 들어 image classification을 video classification으로 한 줄만으로 변경할 수 있다!
'''

from keras.layers import TimeDistributed
# input tensor for sequence of 20 timesteps,
# each containing a 784-dimensional vetcor
input_sequences= Input(shape=(20, 784))
# 784 이미지가 20장. 결국은 동영상을 의미하는 것으로 봐도 된다. 이미지의 sequence가 결국엔 동영상 이므로
# This applies out previous model to everu timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences=TimeDistributed(model)(input_sequences)
# 이미지를 classification하는 방법과 비디오를 classification하는 방법이 동일하다고도 볼 수 있다.

'''
Multi-input and multi-output models

functional API를 자주 사용하는 부분.
functional API는 큰 넘버의 intertwined datastream을 다루기 쉽게 해준다.

트위터에서 얼마나 많은 좋아요와 리트윗이 생길지 예측하는 시스템이라면,
뉴스 자체가 모델 input이 될 것이다. word sequence로.
여기에 보조적인 input이 들어갈 것이다. 뉴스가 쓰여진 날 등.
모델은 또한 두 개의 loss function을 통하여 학습(supervised)될 것이다. 
메인 loss function을 빠르게 쓰는 것은 deep model을 위해 좋다.
'''
# main input은 뉴스를 받을 것이다.정수의 sequence. 
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
# headline input: meant to receive sequences of 100 integers
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
# 모든 시퀀스의 정보를 가지고 있는 single vector로 transform 해주는 애가 LSTM
lstm_out = LSTM(32)(x)

# 여기에 우리는 부가적인 loss를 집어넣는다. 모델에서 loss를 줄인다.
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
# LSTM output과 연결!

auxiliary_input = Input(shape=(5, ), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
# layer을 추가한다. 많이

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# 이것은 두 개의 input, output 모델을 만든 것이다.
# functional API를 이용하여 multi- 모델을 만들었음.
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# 부가 loss function에 0.2의 가중치를 두고 학습시킨다(compile)
# 다른 loss 가중치나 loss를 각각의 다른 output에 주기 위해 우리는 딕셔너리나 list를 쓴다.
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
# 여기서는 하나의 loss를 쓴다. 같은 loss가 쓰임

model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32) # training