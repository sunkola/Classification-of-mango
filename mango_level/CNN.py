import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt


generator = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2,
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )

testgenerator = ImageDataGenerator(rescale = 1./255,
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )

target_size = [180, 180]
batch_size = 32

"""### mango 分A B C 等級
載入mango數據集 從 csv檔
"""

train_dir = './mango/Train/'
test_dir = '/mango/Dev/'
dftrain = pd.read_csv(r'./mango/train.csv')
dftest = pd.read_csv(r'./mango/dev.csv')
print(dftrain.shape, dftest.shape)
print(dftrain.head())

train_gen = generator.flow_from_dataframe(dataframe=dftrain, directory=train_dir, x_col="image_id", y_col="label",
                                          class_mode="categorical",
                                          target_size=target_size,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          seed=101,
                                          subset='training')

val_gen = generator.flow_from_dataframe(dataframe=dftrain, directory=train_dir, x_col="image_id", y_col="label",
                                         class_mode="categorical",
                                         target_size=target_size,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         seed=101,
                                         subset='validation')

test_gen = testgenerator.flow_from_dataframe(dataframe=dftest, directory=test_dir, x_col="image_id", y_col="label",
                                         class_mode="categorical",
                                         target_size=target_size,
                                         batch_size=1,
                                         shuffle=False,
                                         )

print(train_gen.samples, val_gen.samples, test_gen.samples)

print(train_gen.class_indices)

# initializing label list and feeding in classes/indices
labels = [None]*len(train_gen.class_indices)

for item, indice in train_gen.class_indices.items():
    labels[indice] = item

labels

"""### 訓練Model"""

# 創建模型(不包含全連接層和預訓練權重)，最後一層卷積加上GlobalAveragePooling
base_model = tf.keras.applications.ResNet101V2(include_top=False,
                                               weights='imagenet',
                                               pooling='avg',
                                               input_shape=target_size+[3]) # [128, 128, 3]
# 將剛創建的InceptionV3模型接上兩層全連接層，並且最後一層使用Softmax輸出
model_1 = tf.keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_1.summary()

# 儲存最好的網路模型權重
model_mckp = keras.callbacks.ModelCheckpoint('./bestmangooo.h5',
                                             monitor='val_categorical_accuracy',
                                             save_best_only=True,
                                             mode='max')
# 設定停止訓練的條件(當Accuracy超過5迭代沒有上升的話訓練會終止)
model_esp = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=5)

adam = optimizers.Adam(learning_rate=1e-3)
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_delta=0.0001)

model_1.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

history = model_1.fit_generator(generator=train_gen,
                        steps_per_epoch = train_gen.samples//batch_size,
                        validation_data=val_gen,
                        validation_steps= val_gen.samples//batch_size,
                        epochs=30,
                        callbacks=[rlr, model_esp, model_mckp])
