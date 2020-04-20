# Modelado inicial del problema y solucion por analisis estatico lineal.
from PySections import *
import matplotlib.pyplot as plt
estructura = Estructura()

h = 1.5
l = 0.05
hh = 1
concreto = Material('Concreto', 1, 0.2, 9.9 * 10 ** -6, 23.54)
seccion = Seccion('Elementos', TipoSeccion.GENERAL, [1, 1], concreto)
secinft = Seccion('Elementos', TipoSeccion.GENERAL, [9 * 10 ** 10, 9 * 10 ** -10], concreto)

estructura.agregarNodo(x=0, y=h / 2)
estructura.agregarNodo(x=l, y=0, fix=[False, False, False])
estructura.agregarNodo(x=l, y=h, fix=[False, False, False])
estructura.agregarNodo(x=hh, y=h / 2)
estructura.agregarNodo(x=l + hh, y=0, fix=[False, False, False])
estructura.agregarNodo(x=l + hh, y=h, fix=[False, False, False])

estructura.agregarElemento(nodoInicial=1, nodoFinal=0, seccion=seccion, defCortante=False, tipo=Tipo.CUATRO)
estructura.agregarElemento(nodoInicial=2, nodoFinal=0, seccion=seccion, defCortante=False, tipo=Tipo.CUATRO)
estructura.agregarElemento(nodoInicial=4, nodoFinal=3, seccion=seccion, defCortante=False, tipo=Tipo.CUATRO)
estructura.agregarElemento(nodoInicial=5, nodoFinal=3, seccion=seccion, defCortante=False, tipo=Tipo.CUATRO)
estructura.agregarElemento(nodoInicial=0, nodoFinal=3, seccion=secinft, defCortante=False, tipo=Tipo.UNO)

estructura.agregarCargaNodo(nodo=0, px=1)

param = [0, 0, 0, 0, 0,0,0, 0, 0, 0, 0, 0]
param[0] = 1
param[1] = 0.001
param[2] = 0.0001
param[3] = 0.1
param[4] = 70
param[5] = 20
param[6] = 1
param[7] = 0.05
param[8] = 0.3
param[9] = 'numiter'
param[10] = True
param[11] = 0

# e1 = param[0]
# e2 = param[1]
# e3 = param[2]
# dli = param[3]
# Nd = param[4]
# Nj = param[5]
# gamma = param[6]
# dlmax = param[7]
# dlmin = param[8]
# li = 0
# incremento = param[9]

print(estructura.solucionar(verbose=True, dibujar=True, guardar=False, carpeta='', analisis='CR', param=param))
plt.plot(estructura.RECORDU,estructura.RECORDF)
a = a