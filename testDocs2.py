from PySections import *  # Recomendado
estructura = Estructura()
estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=6, y=0, fix=[False, False, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, False])
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

estructura.agregarElemento(viga_concreto, 0, 1, tipo=Tipo.UNO)
estructura.agregarElemento(viga_concreto, 1, 2, tipo=Tipo.UNO)

estructura.agregarCargaDistribuida(elemento=0, WX=0, WY=15)
estructura.agregarCargaNodo(nodo=1, px=0, py=0, m=63.5)

estructura.solucionar(verbose=False, dibujar=True)
