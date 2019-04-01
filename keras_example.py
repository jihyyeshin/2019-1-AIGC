#keras_example.py
#Keras Functional API
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
y = model(x)

