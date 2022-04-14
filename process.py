from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

import matplotlib.pyplot as plt
from glob import glob
import pickle
train_path ="fruits-360_dataset/fruits-360/Training"
test_path="fruits-360_dataset/fruits-360/Test"

img =load_img(train_path + "/Apple Golden 1/0_100.jpg")
x=img_to_array(img)
plt.imshow(img)
plt.axis("off")
plt.show()

className=glob(train_path+'/*')
numberofclass=len(className)


#%%Model build
def model_create():
    model=Sequential([
        Conv2D(32, (3,3),activation=('relu'),input_shape=x.shape),
        MaxPool2D((2,2)),
        Conv2D(64,(3,3),activation=('relu')),
        MaxPool2D((2,2)),
        Conv2D(64,(3,3),activation=('relu')),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(60,activation='relu'),
        Dense(60,activation='relu'),
        Dense(numberofclass,activation='softmax')
        ])
    history=model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
    return model
batch_size=32
    
#%% Data generator
train_datagen=ImageDataGenerator(rescale=1/255,
                   shear_range=0.3,
                   horizontal_flip=True,
                   vertical_flip=True,
                   zoom_range=0.3)
test_datagen=ImageDataGenerator(rescale=1/255)

train_generator=train_datagen.flow_from_directory(train_path,
                                                  target_size=x.shape[:2],
                                                  batch_size=batch_size,
                                                  color_mode='rgb',
                                                  class_mode='categorical')
test_generator=test_datagen.flow_from_directory(test_path,
                                                  target_size=x.shape[:2],
                                                  batch_size=batch_size,
                                                  color_mode='rgb',
                                                  class_mode='categorical')
model=model_create()
model.fit_generator(generator=train_generator,
                    steps_per_epoch=1600//batch_size,
                    epochs=100,
                    validation_data =test_generator,
                    validation_steps=800//batch_size)
#%%
model.save_weights("result.h5")
#%%
print(model.history.history.keys())

import json
with open("cnnfruithistory.json","w") as f:
    json.dump(model.history.history, f)

import codecs
with codecs.open("cnnfruithistory.json","r",encoding="utf-8")as f:
    h=json.loads(f.read())
#%%
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"],label="Vallidation Loss")
plt.legend()
plt.show()

plt.plot(h["accuracy"], label="Train Loss")
plt.plot(h["val_accuracy"],label="Vallidation Loss")
plt.legend()
plt.show()
