from keras.applications.vgg19 import VGG19
from keras.layers import Lambda, Input
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import time

model = VGG19(include_top=False, input_shape=(64, 64, 3), weights='imagenet')

# Freeze the layers
for layer in model.layers[:]:
  layer.trainable = False

output = model.output
output = Flatten()(output)
output = Dense(units=200, activation='softmax')(output)
model = Model(model.input, output)

model.summary()
time1 = time.time()
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(train_data, train_labels, epochs=20, batch_size=256, validation_data=(val_data, val_labels))
time2 = time.time()
print(history.history.keys())
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("Training Time", time2-time1)

score = model.evaluate(test_data, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
