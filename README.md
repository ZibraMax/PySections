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

Para el ejemplo se agregarán nodos para un pórtico completo:
<details>
<summary>Código</summary>
	```Python
	vanos = [4.5,3.5/2,3.5/2,4.5]
	alturas = [4,3,3,3]
	for altura in alturas:
		for vano in vanos:
			estructura.agregarNodo(x=vano, y=altura, fix=[False, False, False])
	```
</details>

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

