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
```
zip_ref = zipfile.ZipFile('/content/drive/MyDrive/Data_Rural_vs_Urbano.zip', 'r')
zip_ref.extractall('/content/tmp')
zip_ref.close()
```

```
train_dir = '/content/tmp/Data_Rural_vs_Urbano/Train'
validation_dir = '/content/tmp/Data_Rural_vs_Urbano/Test'
```