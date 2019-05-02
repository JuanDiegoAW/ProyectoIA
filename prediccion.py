import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

longitud, altura = 150, 150
resultados = np.empty([2])

ruta_imagen = "./oficial2.jpeg"

#direccion del modelo y pesos de la deteccion de baya
modelo_det = './modelo/deteccion/modelo.h5'
pesos_modelo_det = './modelo/deteccion/pesos.h5'
#la cargamos a cnn que sera nuestra red
cnn = load_model(modelo_det)
cnn.load_weights(pesos_modelo_det)

#direccion del modelo y pesos de la clasificacion de baya
modelo_clas = './modelo/clasificacion/modelo.h5'
pesos_modelo_clas = './modelo/clasificacion/pesos.h5'
#la cargamos a cnn que sera nuestra red
cnn1 = load_model(modelo_clas)
cnn1.load_weights(pesos_modelo_clas)

#funcion que recibe direccion de imagen e identifica si es baya de cafe o no
def predict(file):
  #cargamos imagen y reescalamos a la longitud y altura previamente definidos
  x = load_img(file, target_size=(longitud, altura))
  #convertimos en un array nuestra imagen
  x = img_to_array(x)
  #en nuestra primera dimension anadimos una dimension extra
  x = np.expand_dims(x, axis=0)
  #arreglo en el que recibimos las predicciones por cada clase ejemplo [1, 0, 0, 0, 0] eso quiere decir que esta en la clase 1
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)#nos devuelve la posicion de la prediccion afirmativa en nuestro arreglo
  #el arreglo nos regresa dos dimensiones en la dimension 0 estan las predicciones
  if array[0][0] <= 0.2:
      print("")
      print("Si es una baya de cafe.")
      resultados[0] = 1

  else:
      continuar = 0
      print("")
      print("Esto no es una baya de cafe. Imagen: " + file)
      print("No se seguirá con la clasificación de la imagen.")
      resultados[0] = 0
  return answer

#funcion que recibe direccion de imagen y hace la clasificacion
def predict_class(file):
  #cargamos imagen y reescalamos a la longitud y altura previamente definidos
  x = load_img(file, target_size=(longitud, altura))
  #convertimos en un array nuestra imagen
  x = img_to_array(x)
  #en nuestra primera dimension anadimos una dimension extra
  x = np.expand_dims(x, axis=0)
  #arreglo en el que recibimos las predicciones por cada clase ejemplo [1, 0, 0, 0, 0] eso quiere decir que esta en la clase 1
  array = cnn1.predict(x)
  #el arreglo nos regresa dos dimensiones en la dimension 0 estan las predicciones
  result = array[0]
  answer = np.argmax(result)#nos devuelve la posicion de la prediccion afirmativa en nuestro arreglo
  if answer == 0:
      print("Fase 3: Amarillo Rojo.")
      resultados[1] = 3
  elif answer == 1:
      print("Fase 5: Podrido.")
      resultados[1] = 5
  elif answer == 2:
      print("Fase 4: Rojo Oscuro.")
      resultados[1] = 4
  elif answer == 3:
      print("Fase 2: Verde Claro.")
      resultados[1] = 2
  elif answer == 4:
      print("Fase 1: Verde.")
      resultados[1] = 1

  return answer

predict(ruta_imagen)
if resultados[0] == 1:
    print("Prosiguiendo con la clasificacion de la baya:")
    print("")
    print("################ Resultados de la red ################")
    print("")
    print("La baya de cafe se encuentra en la siguiente fase:")
    predict_class(ruta_imagen)

    def midpoint(ptA, ptB):
	       return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # load the image, convert it to grayscale, and blur it slightly
    imagen_original = cv2.imread(ruta_imagen)
    dim = (600, 600)
    image = cv2.resize(imagen_original, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(ruta_imagen, image.copy())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)

    # loop over the contours individually
    for c in cnts:
    	# if the contour is not sufficiently large, ignore it
    	if cv2.contourArea(c) < 95:
    		continue

    	# compute the rotated bounding box of the contour
    	orig = image.copy()
    	box = cv2.minAreaRect(c)
    	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    	box = np.array(box, dtype="int")

    	# order the points in the contour such that they appear
    	# in top-left, top-right, bottom-right, and bottom-left
    	# order, then draw the outline of the rotated bounding
    	# box
    	box = perspective.order_points(box)
    	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    	# loop over the original points and draw them
    	for (x, y) in box:
    		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
    	# between the top-left and top-right coordinates, followed by
    	# the midpoint between bottom-left and bottom-right coordinates
    	(tl, tr, br, bl) = box
    	(tltrX, tltrY) = midpoint(tl, tr)
    	(blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
    	# followed by the midpoint between the top-righ and bottom-right
    	(tlblX, tlblY) = midpoint(tl, bl)
    	(trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
    	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
    	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    		(255, 0, 255), 2)
    	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    		(255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
    	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # compute the size of the object
    	dimA = dA / 3.66666667
    	dimB = dB / 3.66666667

        # draw the object sizes on the image
    	cv2.putText(orig, "{:.1f}mm".format(dimB),
    		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    		0.65, (255, 255, 255), 2)
    	cv2.putText(orig, "{:.1f}mm".format(dimA),
    		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    		0.65, (255, 255, 255), 2)

    	# show the output image
    	cv2.imshow(ruta_imagen, orig)
    	cv2.waitKey(0)

    print("La altura del cafe es:", dimA, "mm")
    print("El longitud del cafe es:", dimB, "mm")
    diametro = (dimA + dimB) / 2
    print("diametro:", diametro)
    print("################ Datos finales ################")
    if resultados[1] == 1:
        if diametro <= 2:
            print("La baya tiene aproximadamente 5 semanas de edad. ")
            print("Su clasificacion de color es verde.")
            print("Se hubieran necesitado 27 semanas mas para que hubiera madurado.")
        elif diametro > 2 and diametro <= 9:
            print("La baya tiene aproximadamente 16 semanas de edad. ")
            print("Su clasificacion de color es verde.")
            print("Se hubieran necesitado 16 semanas mas para que hubiera madurado.")
        elif diametro > 9 and diametro < 11.54:
            print("La baya tiene aproximadamente 22 semanas de edad. ")
            print("Su clasificacion de color es verde.")
            print("Se hubieran necesitado 10 semanas mas para que hubiera madurado.")
    elif resultados[1] == 2:
        if diametro > 11.54:
            print("La baya tiene aproximadamente 22 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 10 semanas mas para que hubiera madurado.")
        elif diametro >= 11.54 and diametro < 13.21:
            print("La baya tiene aproximadamente 26 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 6 semanas mas para que hubiera madurado.")
        elif diametro >= 13.21 and diametro < 13.92:
            print("La baya tiene aproximadamente 27 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 5 semanas mas para que hubiera madurado.")
        elif diametro >= 13.92 and diametro < 14.22:
            print("La baya tiene aproximadamente 30 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 2 semanas mas para que hubiera madurado.")
    elif resultados[1] == 3:
        if diametro > 13.38:
            print("La baya tiene aproximadamente 27 semanas de edad. ")
            print("Su clasificacion de color es amarillo.")
            print("Se hubieran necesitado 5 semanas mas para que hubiera madurado.")
        elif diametro >= 13.38 and diametro < 13.85:
            print("La baya tiene aproximadamente 30 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 2 semanas mas para que hubiera madurado.")
        elif diametro >= 13.85 and diametro < 14.22:
            print("La baya tiene aproximadamente 29 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("Se hubieran necesitado 3 semanas mas para que hubiera madurado.")
    elif resultados[1] == 4:
        if diametro > 13.38:
            print("La baya tiene aproximadamente 30 semanas de edad. ")
            print("Su clasificacion de color es rojo.")
            print("Se hubieran necesitado 1 - 2 semanas mas para que hubiera madurado.")
        elif diametro >= 13.38 and diametro < 14.22:
            print("La baya tiene aproximadamente 32 - 34 semanas de edad. ")
            print("Su clasificacion de color es verde claro.")
            print("La baya esta madura.")
    elif resultados[1] == 5:
        print("iolo")
        if diametro > 14.22:
            print("La baya de cafe se pasó de 14.22m de diametro. Esto significa que perdió sus nutrientes y, por ende, está podrida.")
        else:
            print("Debido al color de la baya de cafe, esta esta podrida.")
