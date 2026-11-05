#pytestSpot.py
"""
Para facilitar la lectura de la salida de pytest
uso:
$ python3 pytestSpot -h [--help] => muestra esta ayuda
$ python3 pytestSpot n           => n es el orden del ejercicio que se ejecuta

"""
import subprocess
import sys

# Muestra la ayuda
if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
    print("Uso:")
    print("$ pytestSpot -h,--help   => muestra esta ayuda")
    print("$ pytestSpot n           => n es el orden del ejercicio que se ejecuta")
    sys.exit(0)

# Recibir el número de prueba
numero = int(sys.argv[1])

# Obtener la lista de pruebas
resultado = subprocess.run(
    ["pytest", "--collect-only", "-q"],
    capture_output=True,
    text=True
)

pruebas = [
    linea.strip()
    for linea in resultado.stdout.splitlines()
    if "::" in linea
]

# Comprobar el número solicitado
if numero < 1 or numero > len(pruebas):
    print(f"Prueba no válida. Hay {len(pruebas)} pruebas.")
    sys.exit(1)

# Ejecutar la prueba seleccionada
subprocess.run([
    "pytest",
    pruebas[numero - 1],
    "--tb=short",
    "-q"
])