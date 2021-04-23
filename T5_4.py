from PySections import *  # Recomendado
estructura = Estructura()

E = 20000000  # Modulo de Young
v = 0.2  # Coeficiente de Poisson
alfa = 9.9 * 10 ** -6  # Coeficiente de expansión térmica
gamma = 23.54  # Peso unitario
MULT_INF = 100000
concreto = Material('Concreto', E, v, alfa, gamma)

areaViga = 0.35*0.5
inerciaViga = 0.35*0.5**3/12

areaCol = 0.45**2
inerciaCol = 0.45**4/12

viga_concreto = Seccion('Viga 1', TipoSeccion.GENERAL, [
    areaViga*MULT_INF, inerciaViga], concreto)
columna_concreto = Seccion('C 1', TipoSeccion.GENERAL, [
                           areaCol, inerciaCol], concreto)

n = 5
m = 3

h = 3
hx = 5
for i in range(n):
    for j in range(m):
        estructura.agregarNodo(
            x=j*hx, y=i*h, fix=[not (i == 0), not (i == 0), not (i == 0)])
for i in range(0, n-1):
    for j in range(m-1):
        estructura.agregarElemento(
            viga_concreto, (i+1)*m+j, (i+1)*m+j+1, tipo=Tipo.UNO, defCortante=False)
    for j in range(m):
        estructura.agregarElemento(
            columna_concreto, i*m+j, (i+1)*m+j, tipo=Tipo.UNO, defCortante=False)
estructura.solucionar(verbose=False, dibujar=True)

visibles = [33, 24, 15, 6]
escondidos = []
for i in range(len(estructura.libres)):
    if not i in visibles:
        escondidos += [i]

SE = estructura.hacerSuperElemento(visibles, escondidos)
print(SE[1])
