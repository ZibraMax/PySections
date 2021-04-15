from PySections import *
estructura = Estructura()
CONCRETO = Material('CONCRETO', 25000000, 1, 1, 0)
VIGA = Seccion('ELEMENTOS', TipoSeccion.GENERAL, [
               0.25, 0.005, 99999999999999], CONCRETO)

h = 2.5  # Longitud de vano
n = 6  # Número de vanos
W = 5  # Carga
elementos = []
for i in range(n+1):
    estructura.agregarNodo(x=h*i, y=0, fix=[False, False, True])
for i in range(n):
    elementos += [i]
    estructura.agregarElemento(
        seccion=VIGA, nodoInicial=i, nodoFinal=i+1, tipo=Tipo.UNO, defCortante=False)
    estructura.agregarCargaDistribuida(WY=W)
estructura.agregarCargaNodo(nodo=0, m=W*h**2/24)
estructura.agregarCargaNodo(nodo=-1, m=-W*h**2/24)
estructura.solucionar(True, False)
estructura.diagramaConjunto(elementos)
