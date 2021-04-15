# PySections

[![Docs](https://github.com/ZibraMax/PySections/actions/workflows/docs.yml/badge.svg)](https://github.com/ZibraMax/PySections/actions/workflows/docs.yml)

# Analisis Estructural Avanzado

## Guia:

### Inicializar una estructura
```python
from PySections import * #Recomendado
estructura = Estructura() 
```
### Agregar nodos
```python
estructura.agregarNodo(x=0, y=0, fix=[False, False, False])
estructura.agregarNodo(x=6, y=0, fix=[False, False, True])
estructura.agregarNodo(x=12, y=0, fix=[False, False, False])
```
Al agregar nodos se debe especificar las coordenadas x,y del nodo y sus condiciones de apoyo. fix=[False, False, True] es un nodo con restriccion de movimiento pero sin restricción de rotación (pin)
, fix=[False, False, False] es un nodo empotradro y fix=[True, True, True] es un nodo libre. Si no se especifica el fix, se asume que el nodo es libre.

### Agregar elementos
#### Definir secciones y materiales.

Para poder agregar elementos se definen primero el material y seccion del cual esta compuesto el elemento, para ello:

```python
E = 21000000 #Modulo de Young
v = 0.2 #Coeficiente de Poisson
alfa = 9.9 * 10 ** -6 #Coeficiente de expansión térmica
gamma = 23.54 #Peso unitario

concreto = Material('Concreto', E, v, alfa, gamma)
acero = Material('Acero', 200000000, 0.2, 9.9*10**-6, 78.6)
```
Para cada material se define un nombre, su módulo de Young, su coeficiente de Poisson, coeficiente de expansión térmica y su peso unitario.

Luego de definir los materiales se pueden definir las secciones
```python
base = 0.3
altura = 0.6
radio = 0.45
area = 35.3*(0.0254)**2 #W12X120
inercia = 1070*(0.0254)**4 #W12X120
viga_concreto = Seccion('Viga 1', TipoSeccion.RECTANGULAR, [base,altura], concreto)
columna_concreto = Seccion('C 1', TipoSeccion.CIRCULAR, [radio], concreto)
columna_acero = Seccion('C 2', TipoSeccion.GENERAL, [area,inercia], acero)
```
#### Agregar elementos

Para agregar elementos se usa la siguiente sintaxis `estructura.agregarElemento(Seccion, nodo1, nodo2, tipo=Tipo[UNO,DOS,TRES,CUATRO])`

Para este caso:
```Python
estructura.agregarElemento(viga_concreto, 0,1, tipo=Tipo.UNO)
estructura.agregarElemento(viga_concreto, 1,2, tipo=Tipo.UNO)
```

Los elementos tipo 1 no tiene pines

Los elementos tipo 2 tienen un pin en el nodo inicial

Los elementos tipo 3 tienen un pin en el nodo final

Los elementos tipo 4 tienen pin en ambos nodos (tipo cercha)

### Agregar cargas

El programa cuenta con diferentes opciones para cargas, entre ellas:

- Cargas distribuidas en elementos `estructura.agregarCargaDistribuida`
- Cargas puntuales en elementos `estructura.agregarCargaPuntual`
- Cargas y momentos en nodos `estructura.agregarCargaNodo`
- Cargas por temperatura `estructura.agregarCargaPorTemperatura`
- Cargas por preesfuerzo axial `estructura.agregarCargaPreesfuerzoAxial`
- Cargas por preesfuerzo axial-flexión `estructura.agregarCargaPostensadoFlexionAxial`
- Cargas trapezoidales en elementos `estructura.agregarCargaTrapecio`

Las direcciones de las cargas en los elementos son positivas en la dirección de la gravedad y positivas hacia la derecha. Es decir, una carga uniforme sobre una viga de 4.45 kN/m en dirección de la gravedad se asigna como 4.45, no como -4.45.

Para este caso, se agregará una carga distribuida sobre una de las vigas y un momento puntual sobre el nodo interior

```Python
estructura.agregarCargaDistribuida(elemento=0,WX=0, WY=15)
estructura.agregarCargaNodo(nodo=1,px=0, py=0, m=63.5)
```

### Solución y post proceso

Para solucionar se usa el comando
```Python
estructura.solucionar(verbose=False, dibujar=True)
```

En este caso, se abrirán diferentes ventanas mostrando la estructura, reacciones en los apoyos y desplazamientos en nodos libres.

Al correr este comando las pantallas toman este aspecto:

**Estructura:**

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Estructura.jpeg'>


**Reacciones:**


<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Reacciones.jpeg'>

**Desplazamientos:**


<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Desplazamientos.jpeg'>

Para extraer diagramas de fuerzas internas se puede usar el comando:

```Python
elementos = [0,1]
estructura.diagramaConjunto(elementos)
```

En este caso, al método diagrama conjunto se le pasa por parámetro una lista de elementos sobre los que se quiere extraer el diagrama. Si se pasan varios elementos los diagramas se juntarán horizontalmente. Esto permite la visualización de varios diagramas al tiempo como el caso de vigas compuestas por varios elementos. Si se quiere el diagrama de un solo elemento se pasa por parámetro una lista con el elemento que se requiere.

El resultado anterior es el siguiente:

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Diagrama.png'>


## Ejemplo con pórtico
<details>
<summary>Código</summary>

```python
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
#Creación de nodos
Y = 0
for altura in alturas:
    X = 0
    for vano in vanos:
        estructura.agregarNodo(
            x=X, y=Y, fix=[1-(Y == 0), 1-(Y == 0), 1-(Y == 0)])
        X += vano
    Y += altura
#Creación de vigas
for i in range(1, len(alturas)):
    for j in range(len(vanos)-1):
        nodo = i*len(vanos)+j
        estructura.agregarElemento(viga_concreto, nodo, nodo+1)
#Creación de columnas
for i in range(len(vanos)):
    for j in range(len(alturas)-1):
        nodo = j*len(vanos)
        if i != offset:
            estructura.agregarElemento(
                columna_concreto, nodo+i, nodo+len(vanos)+i)
#Creación de riostras
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
#Creación de cargas
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
```
</details>

<details>
<summary>Resultado</summary>

**Estructura:**

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Estructura2.jpeg'>

**Reacciones:**

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Reacciones2.jpeg'>

**Desplazamientos:**

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Desplazamientos2.jpeg'>

**Diagramas:**
Viga del primer piso

<img src='https://raw.githubusercontent.com/ZibraMax/PySections/master/Test/Imagenes/Diagramas2.png'>
</details>
