# %%
from PySections import *

# https://www.linsgroup.com/MECHANICAL_DESIGN/Beam/beam_formula.htm
CONCRETO = Material('CONCRETO', 25000000, 1, 1, 0)
VIGA = Seccion('ELEMENTOS', TipoSeccion.GENERAL, [
               0.25, 0.005, 99999999999999], CONCRETO)
# %%

estructura = Estructura()
W = 3.56
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, True])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaElemento(elemento=0, wy=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %%
estructura = Estructura()
W = 3.56
a = 3
b = 3
c = 3
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=a, y=0, fix=[True, True, True])
estructura.agregarNodo(x=a+b, y=0, fix=[True, True, True])
estructura.agregarNodo(x=a+b+c, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=2,
                           nodoFinal=3, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaElemento(elemento=1, wy=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1, 2])

# %%
estructura = Estructura()
W = 3.56
a = 3
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=a, y=0, fix=[True, True, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaElemento(elemento=0, wy=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()
W1 = 3.56
W2 = 2.5
a = 3
b = 3
c = 3
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=a, y=0, fix=[True, True, True])
estructura.agregarNodo(x=a+b, y=0, fix=[True, True, True])
estructura.agregarNodo(x=a+b+c, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=2,
                           nodoFinal=3, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaElemento(elemento=0, wy=W1)
estructura.agregarCargaElemento(elemento=2, wy=W2)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1, 2])

# %%
estructura = Estructura()
W0 = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaTrapecio(0, W0)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %%
estructura = Estructura()
W0 = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L/2, y=0, fix=[True, True, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaTrapecio(0, W0, elemento=0)
estructura.agregarCargaTrapecio(W0, 0, elemento=1)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %% Con dos elementos
estructura = Estructura()
P = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L/2, y=0, fix=[True, True, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaNodo(nodo=1, py=-P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(x=L/2, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
x = L/3
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(x=x, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
a = 1
b = 1
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(x=a, f=P, elemento=0)
estructura.agregarCargaPuntual(x=L-b, f=P, elemento=0)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
W = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[True, True, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, False])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaDistribuida(WY=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
estructura.agregarNodo(x=0, y=0, fix=[True, True, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, False])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaNodo(py=-P, nodo=0)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[True, True, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, False])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(elemento=0, x=a, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %% Con un elemento
estructura = Estructura()
P = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, False])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(elemento=0, x=a, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0])

# %%
estructura = Estructura()
W = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L+a, y=0, fix=[True, True, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaDistribuida(elemento=0, WY=W)
estructura.agregarCargaDistribuida(elemento=1, WY=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()
W = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L+a, y=0, fix=[True, True, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaDistribuida(elemento=1, WY=W)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()
P = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L+a, y=0, fix=[True, True, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaNodo(nodo=2, py=-P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()
P = 40
L = 12
a = 5
estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L, y=0, fix=[False, False, True])
estructura.agregarNodo(x=L+a, y=0, fix=[True, True, True])
estructura.agregarElemento(seccion=VIGA, nodoInicial=0,
                           nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(seccion=VIGA, nodoInicial=1,
                           nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(x=3, elemento=0, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()

estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=6, y=0, fix=[False, False, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, True])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=1, nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaElemento(elemento=0, wy=3.56)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])

# %%
estructura = Estructura()

estructura.agregarNodo(x=0, y=0, fix=[False, False, True])
estructura.agregarNodo(x=6, y=0, fix=[False, False, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, True])
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=0, nodoFinal=1, tipo=Tipo.UNO, defCortante=False)
estructura.agregarElemento(
    seccion=VIGA, nodoInicial=1, nodoFinal=2, tipo=Tipo.UNO, defCortante=False)
estructura.agregarCargaPuntual(x=3, elemento=0, f=P)
estructura.solucionar(True, False)
estructura.diagramaConjunto([0, 1])
