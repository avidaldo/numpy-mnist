import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"

type ImagesAndLabels = tuple[np.ndarray, np.ndarray]


def load_mnist(cache_dir: Path = Path("data")) -> tuple[ImagesAndLabels, ImagesAndLabels]:
    """Download MNIST into `cache_dir` once, check it, and return its (train, test) splits."""
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / "mnist.npz"
    if not path.exists():
        urllib.request.urlretrieve(MNIST_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MNIST_SHA256:
        path.unlink()
        raise ValueError("The download is corrupted and has been deleted: run the cell again")
    with np.load(path) as mnist:
        return (mnist["x_train"], mnist["y_train"]), (mnist["x_test"], mnist["y_test"])


(X_train, y_train), (X_test, y_test) = load_mnist() 


type(X_train), X_train.dtype, X_train.shape
y_train

def print_pixel_values(image: np.ndarray) -> None:
    for row in image:
        print("".join(f"{pixel:4}" for pixel in row))


print_pixel_values(X_train[0])

##
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


plot_image(X_train[0])

##

y_train[0]


def plot_first_100(images: np.ndarray) -> None:
    for index, image in enumerate(images[:100]):
        plt.subplot(10, 10, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


plot_first_100(X_train)

"""
Ejercicio1 1. Create matrix_first_100_labels, a 10×10 array with the first 100 labels, 
and display it. It should match the grid of images above.
"""
print("Ejercicio 1\n")
# Selecciono los primeros 100 elementos del array de imágenes
matrix_first_100_labels=y_train[:100].reshape(10,10)

# Los muestro                     
print (matrix_first_100_labels.reshape(10,10))

"""
2. Many models expect each example as one flat row of numbers:
 each 28×28 image becomes a vector of 784 pixels. 
 Create X_train_flat and X_test_flat, with shapes 60,000×784 and 10,000×784.
"""
print("Ejercicio 2\n")
# Creo los arrays a partir de los existentes y los ajusto con reshape(filas,columnas)

X_train_flat = X_train.reshape(60000,784)
X_test_flat  = X_test.reshape(10000,784)


"""
3. Change plot_image so that it accepts both a 28×28 image
 and a flat row of 784 pixels. Check it with X_train_flat[0].
"""
print("Ejercicio 3\n")
import matplotlib.pyplot as plt

def plot_image(image: np.ndarray) -> None:
    if image.ndim==1:                  
        image=image.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

plot_image(X_train_flat[0])


"""
4 Check, with one expression and no loop, 
that the bottom-right pixel is 0 in every training image
"""
print("Ejercicio 4\n")

# Con las funciones de indexación de NumPy
# seleccionamos todos los pixel de la esquina inferior
# derecha de las imagenes

es_cero=np.all(X_train[:,-1,-1] == 0)
if es_cero:
	print("Éxito => son cero")
else:
	print ("Fallo => No todos los pixeles son cero")
		

"""
5. Does any digit touch the edge of its frame? 
Store in n_touching_border the number of training images
with ink (a non-zero pixel) in their first or last 
row or column.

"""		
print("Ejercicio 5\n")

# Con un bucle recorremos las imágenes del array
# Comprobamos si son distintas de cero (1) y
# las guardamos.

n_touching_border=0

for imagen in X_train:
    if (np.any(imagen[ 0, :] !=0 ) or
    	np.any(imagen[-1, :] !=0 ) or
        np.any(imagen[ :, 0] !=0 ) or
   		np.any(imagen[ :,-1] !=0)):
	    n_touching_border+=1

print("Hay ",n_touching_border," imágenes que tocan el borde")

"""
En los bordes de las imágenes
El primer índice indica la fila
y el segundo la columna

Bordes:

[ 0, :] Borde Superior  
[-1, :] Borde Inferior
[ :, 0] Borde izquierdo 
[ :,-1] Borde derecho

El truco es recordar que con -1 indicamos la última fila 
o columna de la dimensión

Por ejemplo, esta matriz de coordenadas fila,columna:

 0,0  0,1  0,2
 1,0  1,1  1,2
 2,0  2,1  2,2

 Pasaría a:

  0,0  0,1  0,-1
  1,0  1,1  1, 2
 -1,0  2,1  2,-1

y los dos puntos => : representan todos los
elementos de esa dimensión.

 0, :  => Primera fila y tooodos los primeros elementos de las columnas
-1, :  => Ultima fila y todos los últimos elementos de las columnas
 :, 0  => Todos los elementos de la primera columna
 :,-1  => Todos los elementos de la última columna


Por eso seleccionamos
[ 0, :] Borde Superior  
[-1, :] Borde Inferior
[ :, 0] Borde izquierdo 
[ :,-1] Borde derecho   

"""

"""
6. Compute X_train[0] + X_train[0] and look at its maximum.
Then compute the same sum correctly, and store in n_overflowed_pixels the number
of pixels where the two results differ. Why do they differ?

"""
"""
print("Ejercicio 6\n")

Si sumamos directamente los valores de X_train
Como solo almacena enteros uint8 (máximo hasta 255)
habría o podría haber "desbordamiento", sumas que superasen
ese máximo.
Por eso hay que convertirlo a un tipo de dato que pueda
almacenar ese tamaño
"""
print("Ejercicio 6\n")
print("Suma directa\n")
suma=X_train[0]+X_train[0]
print(suma.max())
print("suma correcta, convertimos a int32 con astype()")
suma_correcta=X_train[0].astype(np.int32)+X_train[0].astype(np.int32)
print(suma_correcta.max())
# Hay otra forma pero esta me gusta más
# Como solo queremos saber cuanto desborda, siempre
# que sea mayor = que 128 desborda
n_overflowed_pixels = np.count_nonzero(X_train[0] >= 128)
# Son matrices no es lo mismo Multiplicar que Sumar!!!!


"""
7. One-hot encoding
Some models expect each label as a one-hot vector: ten numbers,
all 0 except a 1 at the position of the digit.
Label 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
Create y_train_onehot and y_test_onehot.

"""
print("Ejercicio 7\n")

# Cada dígito se representa como un vector one-hot
# de diez posiciones con un "1" donde corresponde
# 1 se representaria como una matriz:
#1=> [0,1,0,0,0,0,0,0,0,0]
#2=> [0,0,1,0,0,0,0,0,0,0]
#3=> [0,0,0,1,0,0,0,0,0,0]
#x=> [ceros excepto la posición x =>1]


print("Empleamos la función np.eye\\(\\) que crea una matriz identidad de 10x10\n")
y_train_onehot=np.eye(10,dtype=int)[y_train]
y_test_onehot=np.eye(10,dtype=int)[y_test]
print("y_train_onehot=> ",y_train_onehot,end=("\n"))
print("y_test_onehot=> ",y_test_onehot,end=("\n"))

"""
Ejercicio 8. Muestra el mínimo, máximo,
media y desviación estandar de los 
pixels de las imágenes de entrenamiento
"""
print("Ejercicio 8")

# np tiene funciónes para mostrar esos valores
print("Datos de los pixels de las imágenes de entrenamiento\n")
print("Máximo => ",np.max(X_train),
    " Mínino => ",np.min(X_train),
    " Media => ",np.mean(X_train),
    " Desviación estándar => ",np.std(X_train),sep=":::",end=(":::"))

"""
Ejercicio 9. Escala las imágenes de entrenamiento al rango
[0,1] como X_train_norm, tipo float32. Haz el plot de
la primera imagen
"""
print("Ejercicio 9")

# Convertímos los datos con astype y el tipo de dato
X_train_norm=(X_train/255.00).astype(np.float32)

"""
Reutilizo esta función para mostrar el plot

def plot_image(image: np.ndarray) -> None:
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()
"""

plot_image(X_train_norm[0])


"""
Ejercicio 10.
Store in pixel_norm_mean and pixel_norm_std the mean and the standard deviation 
of the pixels of X_train_norm. You will meet these two numbers again.

Almacen en => pixel_norm_mean y pixel_norm_std la media y la desviación estándar
de los pixels de X_train_norm. 
Débes hallar otra vez esos dos valores
"""
print("Ejercicio 10")

#Volvemos a calcular mean y std, ahora de X_train_norm (Con los valores normalizados)

pixel_norm_mean= np.mean(X_train_norm)
pixel_norm_std= np.std(X_train_norm)

print("pixel_norm_mean => ",pixel_norm_mean," pixel_norm_std =>",pixel_norm_std)



"""
Ejercicio 11.
Standardise the images: subtract the mean and divide by the standard deviation,
so that the pixels have mean 0 and standard deviation 1.
Create X_train_standardized and X_train_standardized.
Compute μ and σ on the training images only, 
and use those same two numbers for the test images.

Estandariza las imagenes: resta la media y divídela por la desviación estandar,
de forma que los pixels tengan media => 0 y desviación estándar => 1.

Crea X_train_standardized y X_test_standardized.
Calcula mu y sigma solo en las imágenes de entrenamiento y
emplea esos números para las imágenes de test.
"""

print("Ejercicio 11")

# Numpy nos permite vectorizar operaciones en sus elementos, no 
# necesitamos un bucle para estandarizar uno a uno los elementos

# Como con X_train_norm, hacemos X_test_norm
X_test_norm=(X_test/255.00).astype(np.float32)

X_train_standardized=(X_train_norm - pixel_norm_mean) / pixel_norm_std
X_test_standardized=(X_test_norm - pixel_norm_mean) / pixel_norm_std

print("Media estandarizada de X_train ",X_train_standardized.mean(),"\n")
print("Media estandarizada de X_test ",X_test_standardized.mean(),"\n")
print("Comparación\n")
print("Antes  X_train_norm[0,10,10] \n",X_train_norm[0,10,10],"\n")
print("Despues X_train_standardized[0,10,10]\n",X_train_standardized[0,10,10],"\n")

print ("Ejercicio 12")

"""
12. Standardise X_train_norm, the images scaled to [0, 1], the same way, 
as X_train_norm_standardized. Check with np.allclose that it equals X_train_standardized. 
Why must they be equal?
"""

X_train_norm_standardized=(X_train_norm - X_train_norm.mean())/X_train_norm.std()
if np.allclose(X_train_norm_standardized,X_train_standardized):
    print("\nX_train_norm_standardized y X_train_standardized son aproximadamente iguales")
print("\nLos valores son practicamente iguales debido a todos los valores se dividen entre\n"
    "255 por lo tanto ese factor se cancela al estandarizar")