from PySections import *

# https://www.linsgroup.com/MECHANICAL_DESIGN/Beam/beam_formula.htm
estructura = Estructura()
CONCRETO = Material('CONCRETO', 25000000, 1, 1, 0)
VIGA = Seccion('ELEMENTOS', TipoSeccion.GENERAL, [
               0.25, 0.005, 99999999999999], CONCRETO)

estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=2, y=0, fix=[False, False, False])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(8.5, 2*0.75)
estructura.agregarCargaTrapecio(5, 10)
# estructura.agregarCargaElemento(elemento=1, wy=3.56)
# estructura.agregarCargaNodo(nodo=1, px=0, py=-80, m=0)
estructura.solucionar(True, True)
estructura.diagramaConjunto([0])
