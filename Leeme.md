# Las imágenes son arrays: MNIST con NumPy

Un primer proyecto con NumPy y con las herramientas habituales de cualquier proyecto de software: Git, un *fork*, un entorno virtual, pruebas y integración continua.

MNIST es un conjunto de datos clásico formado por 70.000 dígitos escritos a mano. Cada dígito se representa mediante una cuadrícula de 28 × 28 números. Cuando se cargan en Python, un conjunto de imágenes es simplemente un array de NumPy. El conjunto de entrenamiento tiene forma `60000 × 28 × 28`, por lo que todas las operaciones que se hacen sobre las imágenes —escalarlas, añadir ruido, desplazarlas o recortarlas— son manipulaciones de arrays.

Los [ejercicios](mnist_numpy.ipynb) abarcan desde las dimensiones (*shapes*) y los tipos de datos hasta la normalización sin fuga de información, la comparación entre bucles y vectorización, y el aumento de datos manual. Las pruebas comprueban tus respuestas, y una [solución](mnist_numpy_solution.ipynb) compara distintas formas de resolver cada ejercicio.

## Qué necesitas

- Python 3.13 o posterior, Git y una cuenta de GitHub.
- Conocimientos básicos de NumPy: crear arrays, indexar y seleccionar porciones (*slicing*), y cambiar la forma de los arrays (*reshaping*).
- La guía [NumPy: conceptos absolutamente básicos para principiantes](https://numpy.org/doc/stable/user/absolute_beginners.html) explica estos conceptos. También se trabajan en los cuadernos de [Fundamentos de datos con Python](https://github.com/avidaldo/ai/tree/main/pia/python-data-essentials).

## Pasos

### 1. Crea un fork y clona el repositorio

> **Consejo:** Si no conoces los *forks*, las ramas y las solicitudes de integración (*pull requests*), consulta la guía sobre [forks, ramas y pull requests](https://github.com/avidaldo/ai/blob/main/pia/setup/git_github.md).

Un **fork** es una copia propia de este repositorio en GitHub. Puedes enviar cambios a esa copia sin modificar el repositorio original.

Pulsa **Fork** en la parte superior de la página del repositorio. Después, clona tu fork en el ordenador:

```bash
git clone https://github.com/<tu-nombre-de-usuario>/numpy-mnist.git
cd numpy-mnist
```

> **Importante:** Para poder darte retroalimentación, el docente necesita identificar quién está haciendo los ejercicios. Por favor, publica la URL de tu perfil de GitHub en el [curso de Moodle](https://fpadistancia.edu.xunta.gal/mod/feedback/view.php?id=2318916&forceview=1).

### 2. Crea el entorno de trabajo

Instala los paquetes indicados en `requirements.txt` dentro de un entorno virtual propio.

Con **uv**:

```bash
uv venv
uv pip install -r requirements.txt
```

O con **conda**:

```bash
conda create --name numpy-mnist python=3.13
conda activate numpy-mnist
pip install -r requirements.txt
```

Abre `mnist_numpy.ipynb` en tu editor y selecciona este entorno como *kernel* del cuaderno.

### 3. Ejecuta las pruebas: fallarán al principio

```bash
uv run pytest
```

Con conda, o con el entorno virtual ya activado, puedes ejecutar:

```bash
pytest
```

Las pruebas ejecutan el cuaderno de principio a fin y comprueban todos los nombres de variables y elementos que pide cada ejercicio. Antes de resolverlos, todas las pruebas fallarán excepto una: la que comprueba que el cuaderno se puede ejecutar. Cada fallo indica qué ejercicio está pendiente.

La primera ejecución descarga MNIST (11 MB) en la carpeta `data/`. Git ignora esa carpeta, por lo que esos datos no se añaden al repositorio.

### 4. Activa la integración continua

El archivo `.github/workflows/tests.yml` indica a GitHub que ejecute las mismas pruebas en sus propios servidores cada vez que envíes cambios. Esto se llama **integración continua** (*Continuous Integration*, CI).

En los forks nuevos, GitHub mantiene los flujos de trabajo desactivados hasta que los autorices. Abre la pestaña **Actions** de tu fork y actívalos.

A partir de ese momento, cada vez que envíes un *commit*, aparecerá una marca verde si las pruebas se superan o una cruz roja si alguna falla. En este último caso, podrás consultar el registro para ver qué pruebas han fallado.

### 5. Trabaja en una rama, crea commits y envía los cambios

Mantén tu trabajo en una **rama** para que la rama `main` conserve el estado original del proyecto:

```bash
git switch -c exercises
```

Resuelve los ejercicios en `mnist_numpy.ipynb`, en la celda vacía situada debajo de cada enunciado, y ejecuta las pruebas mientras avanzas.

Cada vez que termines una parte, crea un *commit* con un mensaje que explique qué has hecho. Envía los cambios a GitHub cuando quieras guardar tu progreso:

```bash
git add mnist_numpy.ipynb
git commit -m "Exercise 7: one-hot labels"
git push -u origin exercises
```

El comando `git push -u origin exercises` se utiliza la primera vez. Después, normalmente bastará con ejecutar:

```bash
git push
```

> **Consejo:** Intenta resolver los ejercicios por tu cuenta primero. Puedes utilizar un asistente de IA para ayudarte, pero **asegúrate de comprender todas las soluciones**. El objetivo es aprender cómo funcionan los arrays y NumPy, cómo procesar imágenes con ellos y cómo trabajar en un proyecto con Git, GitHub, pruebas e integración continua.

### 6. Abre una pull request para recibir comentarios

Una **pull request** (solicitud de integración) propone incorporar los commits de una rama a otra. También muestra las diferencias (*diff*) entre ambas ramas, sobre las que otras personas pueden hacer comentarios.

Cuando hayas resuelto algunos ejercicios, abre una pull request desde la rama `exercises` de tu fork hacia la rama `main` del repositorio original. En la página de tu fork en GitHub, pulsa **Contribute** y después **Open pull request**.

Antes de enviarla, revisa los cambios como lo haría una persona revisora: comprueba las diferencias, el resultado de las pruebas y qué aspectos mejorarías.

Las pruebas se ejecutan también en la pull request, y los comentarios de retroalimentación se publican sobre las diferencias del código. Puedes seguir enviando cambios a la rama `exercises`: la pull request se actualizará automáticamente con cada nuevo commit.

Cuando termine la revisión, la pull request se cerrará **sin fusionarse**. Así, la rama `main` del repositorio original seguirá mostrando los ejercicios sin resolver para el resto del alumnado.

También son bienvenidas las pull requests que mejoren los propios ejercicios, por ejemplo, corrigiendo una errata, aclarando un enunciado o añadiendo una prueba que falte. Para esos cambios, crea una rama independiente; no los envíes desde `exercises`.

### 7. Compara tu trabajo con la solución

Cuando hayas intentado resolver cada ejercicio, abre [`mnist_numpy_solution.ipynb`](mnist_numpy_solution.ipynb). Este cuaderno compara varias formas de resolver la mayoría de los ejercicios y explica cuál es mejor y por qué.

Para ejecutar las pruebas con el cuaderno de soluciones, en Linux o macOS:

```bash
MNIST_NOTEBOOK=mnist_numpy_solution.ipynb uv run pytest
```

En Windows, desde PowerShell:

```powershell
$env:MNIST_NOTEBOOK="mnist_numpy_solution.ipynb"; uv run pytest
```

## Trabajar con un asistente de IA

Después de resolver los ejercicios, pide a un asistente de IA que te proponga otra forma de resolver dos de ellos. Mide el tiempo de ejecución de ambas versiones y anota cuál funciona mejor y por qué.

## Mantener tu fork actualizado

Si el repositorio original cambia después de crear tu fork, el botón **Sync fork**, situado en la página de tu fork en GitHub, permite incorporar esos cambios a tu rama `main`.

Después, si necesitas esos cambios en tu rama de trabajo, fusiona `main` con esa rama.

## Archivos del proyecto

| Archivo | Descripción |
| --- | --- |
| [`mnist_numpy.ipynb`](mnist_numpy.ipynb) | Cuaderno con los ejercicios. |
| [`mnist_numpy_solution.ipynb`](mnist_numpy_solution.ipynb) | Cuaderno con una solución y una comparación de alternativas. |
| [`tests/test_mnist_numpy.py`](tests/test_mnist_numpy.py) | Pruebas: una o varias por ejercicio. |
| [`.github/workflows/tests.yml`](.github/workflows/tests.yml) | Configuración del flujo de integración continua. |
| `requirements.txt` | Lista de paquetes necesarios. |
| `pytest.ini` | Configuración de las pruebas; los avisos (*warnings*) se tratan como errores. |
