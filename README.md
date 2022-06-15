# Curso de Modelos Predictivos con Python

## Optimizadores

* Optimizador Momentum
* Algoritmo de gradiente adaptativo
* Adadelta

### Adadelta
Elimina la nececidad de seleccionar una tasa manual


## Arquitecturas CNN

Combinan capas 

Input > Convolution > Pooling > Convolution > Pooling > Fully connected

## Tema 1: Clasificador de imagenes rurales vs ciudad

### Entrenamiento


Para este curso estaremos utilizando las siguientes librerías:
```Python
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.applications import resnet50

import cv2
import zipfile

import os
import numpy as np

import matplotlib.pyplot as plt
```

En el zip vienen las imagenes
```Python
zip_ref = zipfile.ZipFile('/content/drive/MyDrive/Data_Rural_vs_Urbano.zip', 'r')
zip_ref.extractall('/content/tmp')
zip_ref.close()
```

```Python
train_dir = '/content/tmp/Data_Rural_vs_Urbano/Train'
validation_dir = '/content/tmp/Data_Rural_vs_Urbano/Test'
```

Definimos los parámetros
```Python
width_shape = 224
height_shape = 224
num_classes = 2
epoch = 40
batch_size = 16
```

### Generar imagenes artificiales
```Python
train_datagen = ImageDataGenerator(
    rotation_range = 20,
    zoom_range= 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    # Rotar la imagen en eje vertical y hotizontal
    horizontal_flip = True,
    vertical_flip = True,

    preprocessing_function = preprocess_input)

valid_datagen = ImageDataGenerator(
    rotation_range = 20,
    zoom_range= 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    # Rotar la imagen en eje vertical y hotizontal
    horizontal_flip = True,
    vertical_flip = True,

    preprocessing_function = preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (width_shape, height_shape),
    batch_size = batch_size,
    class_mode = 'categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size = (width_shape, height_shape),
    batch_size = batch_size,
    class_mode = 'categorical'
)
```