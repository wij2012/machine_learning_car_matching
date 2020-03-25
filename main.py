import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

data, info = tfds.load("cars196", with_info=True)
train_data, test_data = data['train'], data['test']


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_data, info.features['label'], epochs=10)
#
# test_loss, test_acc = model.evaluate(test_data, info.features['label'], verbose=2)
#
# print('\nTest accuracy:', test_acc)