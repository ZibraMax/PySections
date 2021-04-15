from PySections import *  # Recomendado


E = 21000000  # Modulo de Young
v = 0.2  # Coeficiente de Poisson
alfa = 9.9 * 10 ** -6  # Coeficiente de expansión térmica
gamma = 23.54  # Peso unitario

concreto = Material('Concreto', E, v, alfa, gamma)
acero = Material('Acero', 200000000, 0.2, 9.9*10**-6, 78.6)

base = 0.3
altura = 0.6
radio = 0.45
area = 35.3*(0.0254)**2  # W12X120
inercia = 1070*(0.0254)**4  # W12X120
viga_concreto = Seccion('Viga 1', TipoSeccion.RECTANGULAR, [
                        base, altura], concreto)
columna_concreto = Seccion('C 1', TipoSeccion.CIRCULAR, [radio], concreto)
columna_acero = Seccion('C 2', TipoSeccion.GENERAL, [area, inercia], acero)

estructura = Estructura()
vanos = [4.5, 3.5/2, 3.5/2, 4.5, 0]
alturas = [4, 3, 3, 3, 0]
W = 40
offset = 2

Y = 0
for altura in alturas:
    X = 0
    for vano in vanos:
        estructura.agregarNodo(
            x=X, y=Y, fix=[1-(Y == 0), 1-(Y == 0), 1-(Y == 0)])
        X += vano
    Y += altura

for i in range(1, len(alturas)):
    for j in range(len(vanos)-1):
        nodo = i*len(vanos)+j
        estructura.agregarElemento(viga_concreto, nodo, nodo+1)
for i in range(len(vanos)):
    for j in range(len(alturas)-1):
        nodo = j*len(vanos)
        if i != offset:
            estructura.agregarElemento(
                columna_concreto, nodo+i, nodo+len(vanos)+i)
for j in range(len(alturas)-1):
    base = j*len(vanos)+offset
    top = (j+1)*len(vanos)+offset
    if j % 2 == 0:
        nodo1 = base
        nodo2 = top-1
        nodo3 = top+1
        estructura.agregarElemento(
            columna_acero, nodo1, nodo2, tipo=Tipo.CUATRO)
        estructura.agregarElemento(
            columna_acero, nodo1, nodo3, tipo=Tipo.CUATRO)
    else:
        nodo1 = top
        nodo2 = base-1
        nodo3 = base+1
        estructura.agregarElemento(
            columna_acero, nodo2, nodo1, tipo=Tipo.CUATRO)
        estructura.agregarElemento(
            columna_acero, nodo3, nodo1, tipo=Tipo.CUATRO)
for i in range((len(vanos)-1)*(len(alturas)-1)):
    estructura.agregarCargaElemento(elemento=i, wy=W)

for j in range(len(alturas)-1):
    nodo = j*len(vanos)
    fact = j/(len(alturas)-1)
    estructura.agregarCargaNodo(nodo=nodo, px=300*fact, py=0, m=0)

estructura.solucionar(True, True)


for j in range(len(alturas)-1):
    elementos = (np.array(list(range(len(vanos)-1))) +
                 (len(alturas)-1)*j).tolist()
    estructura.diagramaConjunto(elementos)
