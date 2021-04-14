from PySections import *
import matplotlib.pyplot as plt
# Modelado con dos elementos

estructura = Estructura()

CONCRETO = Material('CONCRETO', 25000000, 1, 1, 0)
VIGA = Seccion('ELEMENTOS', TipoSeccion.GENERAL, [
               0.25, 0.005, 99999999999999], CONCRETO)

estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=6, y=0, fix=[True, True, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, False])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=1, nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
# estructura.agregarCargaElemento(elemento=0, wy=3.56)
# estructura.agregarCargaElemento(elemento=1, wy=3.56)
estructura.agregarCargaNodo(nodo=1, px=0, py=-80, m=0)
estructura.solucionar(True, False)
estructura.elementos[0].diagramas(100)
estructura.elementos[1].diagramas(100)

# Modelado con solo un elemento
estructura = Estructura()
estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=12, y=0, fix=[False, False, False])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
# estructura.agregarCargaElemento(elemento=0, wy=3.56)
estructura.agregarCargaPuntual(80, 6)
estructura.solucionar(True, False)
estructura.elementos[0].diagramas(100)
