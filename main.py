# IMPORTING LIBRARIES and Defining our directories where files are stored
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers

train_dir = '/archive/train/'
test_dir = '/archive/test/'
classes = 7


# COUNT EXPRESSION FUNCTION TO STORE NO. OF IMAGES IN A EXPRESSION FOLDER
def count_expression(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))

    df = pd.DataFrame(dict_, index=[set_])
    return df


train_count = count_expression(train_dir, 'Quantity in train set')
test_count = count_expression(test_dir, 'Quantity in test set')
train_count.transpose().plot(kind="bar")
plt.title('Number of images in train dataset')
test_count.transpose().plot(kind="bar")
plt.title('Number of images in test dataset')
plt.figure(figsize=(14, 22))
i = 1
for expression in os.listdir(train_dir):
    img = keras.preprocessing.image.load_img((train_dir + expression + '/' + os.listdir(train_dir + expression)[3]))
    plt.subplot(1, classes, i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()


# Preprocessing
batch_size = 64
size = 48


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)


train_set = train_datagen.flow_from_directory(train_dir,
                                              batch_size=batch_size,
                                              target_size=(size, size),
                                              shuffle=True,
                                              color_mode="grayscale", class_mode='categorical')


test_datagen = ImageDataGenerator(rescale=1. / 255)


test_set = test_datagen.flow_from_directory(test_dir,
                                            batch_size=batch_size,
                                            target_size=(size, size),
                                            shuffle=True,
                                            color_mode="grayscale", class_mode='categorical')
train_set.class_indices
# MODEL-BUILDING
# initialising the CNN model
def get_model(input_size):
    # BUILDING the CNN
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    # COMPILING the CNN
    model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


our_model = get_model((size, size, 1))
our_model.summary()

steps_per_epoch = train_set.n // train_set.batch_size
validation_steps = test_set.n // test_set.batch_size

mod = our_model.fit(x=train_set,
                    validation_data=test_set,
                    epochs=3,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)

model_json = our_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
our_model.save_weights('model_best_weights.h5')
# PERFORMANCE PLOT
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 2)
plt.plot(mod.history['accuracy'])
plt.plot(mod.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 1)
plt.plot(mod.history['loss'])
plt.plot(mod.history['val_loss'])
plt.title('model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
train_loss, train_accu = our_model.evaluate(train_set)
test_loss, test_accu = our_model.evaluate(test_set)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_accu * 100, test_accu * 100))
# ON TRAINING SET
y_pred = our_model.predict(train_set)
y_pred = np.argmax(y_pred, axis=1)
class_labels = test_set.class_indices
class_labels = {v: k for k, v in class_labels.items()}

