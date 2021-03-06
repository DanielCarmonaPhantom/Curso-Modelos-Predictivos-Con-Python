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

Nota: En el ejemplo se utilizan 4 epoch. Si dejas los 40 puede que tarde 4 horas. 

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
```Bash
Found 878 images belonging to 2 classes.
Found 110 images belonging to 2 classes.
```

Se guardan los datos:
```Python
nb_train_samples = 878
nb_valid_samples = 110
```

```Python
image_input = Input(shape=(width_shape, height_shape, 3)) # Se coloca 3 para los 3 colores

m_Resnet50 = resnet50.ResNet50(input_tensor = image_input, include_top = 'False', weights='imagenet')
m_Resnet50.summary()
```

```Python
last_layer = m_Resnet50.layers[-1].output

x = Flatten(name = 'flatter')(last_layer)
x = Dense(128, activation = 'relu', name = 'fc1')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation = 'relu', name = 'fc2')(x)
x = Dropout(0.3)(x)

out = Dense(num_classes, activation='softmax', name= 'output')(x)
custom_model = Model(image_input, out)
custom_model.summary()
```
