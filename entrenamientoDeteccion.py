#librerias para manejo de archivos y directorios
import sys
import os
#libreria para procesar imagenes
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#libreria para armar la red neuronal usando la API de keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k

#mata cualquier otra sesion de keras que este activa para que solo nuestra red sea entrenada y use los recursos
k.clear_session()
####################################################PRIMERA PARTE QUE CONSISTE EN VALIDAR SI ES O NO UNA BAYA DE CAFE################################

#lectura de nuestras carpetas de imagenes
#cadenas con la direccion de nuestras imagenes de entrenamiento y validacion
data_entrenamiento='./data/EntrenamientoBaya/entrenamiento'
data_validacion='./data/EntrenamientoBaya/validacion'

#parametros para nuestra red

epocas = 25 #numero de veces que vamos a iterar sobre todo nuestro set de datos
longitud, altura = 150, 150 #altura y longitud a la que vamos a convertir nuestra imagen
batch_size = 10 #numero de imagenes que vamos a enviar a procesamiento en cada paso
pasos = 4000 #numero de veces que se va a procesar la informacion por epoca
pasos_validacion = 1000 #numero de veces que va a probar las imagenes de validacion para poder comprobar el entrenamiento

#tamano de nuestros filtros de convolucion
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)

tamano_pool = (2, 2) #tamano de la matriz de Pooling
clases = 2 #cantidad de clases o fases en las que vamos a clasificar(carpetas)
lr = 0.0005 #constante de aprendizaje

#tratamiento (pre-procesamiento de imagenes) Reescalado

entrenamiento_datagen = ImageDataGenerator(

    rescale=1./255, #vuelve cada valor de pixel posible con valores entre 1 o 0
    shear_range=0.3, #aleatoriamente inclina algunas de las imagenes para que la red aprenda a leer imagenes que no estan completamente verticales u horizontales
    zoom_range=0.3, #aleatoriamente le hace zoom a algunas imagenes para que aprenda a reconocer segmentos o fotos con zoom
    horizontal_flip = True #aleatoriamente invierte las imagenes para alterar los datos
)
validacion_datagen = ImageDataGenerator(
    #en la validacion nos interesa que las imagenes esten como esten, solo queremos volver los valores de pixeles entre 1 o 0
    rescale=1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
#ubicacion de la carpeta de entrenamiento
data_entrenamiento,
target_size= (altura, longitud), #altura y longitud a la que vamos a reescalar
batch_size = batch_size, #tamano del lote de imagenes que vamos a probar
class_mode = 'binary' #porque vamos a hacer una division binaria de las imagenes (Si es una baya de cafe, o no lo es)
)
imagen_validacion = validacion_datagen.flow_from_directory(
data_validacion,
target_size= (altura, longitud),
batch_size = batch_size,
class_mode = 'binary'
)

#crear la red neuronal

cnn= Sequential() #iniciamos una red de capas secuenciales

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu')) #convolucionamos
cnn.add(MaxPooling2D(pool_size=tamano_pool)) #pooleamos

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation = 'relu'))#convolucionamos
cnn.add(MaxPooling2D(pool_size=tamano_pool))#pooleamos

cnn.add(Flatten()) #capa plana para reducir la informacion de la imagen a una dimension
cnn.add(Dense(256,activation = 'relu')) #capa normal, con 256 neuronas
#cnn.add(Dropout(0,5)) #apaga el 50% de cada neura en las capas para que no aprenda un camino especifico si no que entrene bien
cnn.add(Dense(1, activation = 'sigmoid')) #ultima capa densa, solo se tiene una neurona porque esta nos dara la probabilidad de que haya una baya de cafe en la foto o no

cnn.compile(loss='binary_crossentropy', optimizer= optimizers.Adam(lr=lr), metrics=['accuracy']) #compilamos
cnn.fit_generator(imagen_entrenamiento, steps_per_epoch = pasos, epochs = epocas, validation_data=imagen_validacion, validation_steps = pasos_validacion ) #sirve para generar un modelo y pesos para usar sin necesidad de entrenamiento en otra red

#definimos donde queremos guardar nuestro modelo
target_dir = './modelo/deteccion/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
#guardamos modelo y pesos
cnn.save('./modelo/deteccion/modelo.h5')
cnn.save_weights('./modelo/deteccion/pesos.h5')
