from PySections import *

# https://www.linsgroup.com/MECHANICAL_DESIGN/Beam/beam_formula.htm
estructura = Estructura()
CONCRETO = Material('CONCRETO', 25000000, 1, 1, 0)
VIGA = Seccion('ELEMENTOS', TipoSeccion.GENERAL, [
               0.25, 0.005, 99999999999999], CONCRETO)

estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=1, y=0, fix=[True, True, True])
estructura.agregarNodo(x=3, y=0, fix=[False, False, False])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=1, nodoFinal=2, tipo=Tipo.DOS, defCortante=False)
estructura.agregarCargaPuntual(80, 2*0.5, elemento=1)
estructura.agregarCargaTrapecio(10, 5, elemento=0)
# estructura.agregarCargaElemento(elemento=1, wy=3.56)
# estructura.agregarCargaNodo(nodo=1, px=0, py=-80, m=0)
estructura.solucionar(True, True)
estructura.diagramaConjunto([0, 1])
