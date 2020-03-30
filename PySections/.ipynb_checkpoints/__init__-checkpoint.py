import numpy as np
from enum import Enum


class TipoSeccion(Enum):
    "Clase auxiliar para enumerar los tipos de secciones transversales (Rectangular, Circular o General)"
    RECTANGULAR = 1
    CIRCULAR = 2
    GENERAL = 3


class Tipo(Enum):
    "Clase auxiliar para enumerar el tipo de elemento a utilizar (Tipo1, Tipo2, Tipo3 o Tipo4)"
    UNO = 1
    DOS = 2
    TRES = 3
    CUATRO = 4


class Seccion:
    "Clase que representa la sección de los elementos a utilizar, asi como los materiales que la componen"
    def __init__(this, nombre, tipo, propiedades, material):
        """
        Método de inicialización de las secciones
        :param nombre: Nombre definido para la sección a crear
        :param tipo: Geometría de la sección transversal (Rectangular, Circular o General)
        :param propiedades: Dimensiones en metros relacionadas al tipo de geometría (Rectangular: base y altura, Circular: diámetro, General: área, inercia y área de cortante)
        :param material: Tipo de material a utilizar
        """
        this.propiedadesGeometricas = propiedades
        this.material = material
        this.nombre = nombre
        if tipo == TipoSeccion.RECTANGULAR:
            this.area = propiedades[0] * propiedades[1]
            this.inercia = (propiedades[0] * propiedades[1] ** 3) / 12
            this.As = 5 / 6 * this.area
        if tipo == TipoSeccion.CIRCULAR:
            this.area = propiedades[0] ** 2 * np.pi / 4
            this.inercia = np.pi / 4 * (propiedades[0] / 2) ** 4
            this.As = 9 / 10 * this.area
        if tipo == TipoSeccion.GENERAL:
            if len(propiedades) >= 3:
                this.area = propiedades[0]
                this.inercia = propiedades[1]
                this.As = propiedades[2]
            else:
                this.area = propiedades[0]
                this.inercia = propiedades[1]
                this.As = 10 * 10 ** 10


class Resorte:
    "Clase que simula la rigidez de un resorte en un nodo de la estructura"
    def __init__(this, nodo, rigidez, completo):
        """
    Método de i0nicialización de resortes
        :param nodo: Nodo sobre el cual se quiere asignar el resorte
        :param rigidez: Vector con las magnitudes de la rigideces asignadas al resorte en kiloPascales (Kx, Ky, Km)
        :param completo: TODO i dont know what is this Mama
        """
        this.nodo = nodo
        this.completo = completo
        this.rigidez = rigidez
        if completo:
            this.Ke = this.rigidez
        else:
            this.Ke = np.array([[this.rigidez[0], 0, 0], [0, this.rigidez[1], 0], [0, 0, this.rigidez[2]]])
        this.gdl = nodo.gdl

    def calcularKe(this, ngdl):
        """
TODO más tarde
        :param ngdl:
        """
        this.Kee = np.zeros([ngdl, ngdl])
        for i in range(0, 3):
            for j in range(0, 3):
                this.Kee[this.gdl[i], this.gdl[j]] = this.Ke[i, j]


class Nodo:
    "Clase que representa un nodo de la estructura y todos sus atributos"
    def __init__(this, pX, pY, pRestraints, pID):
        """
    Método de inicialización de nodos
        :param pX: Coordenada x del nodo a crear
        :param pY: Coordenada y del nodo a crear
        :param pRestraints: Restricciones del nodo (x,y,m)
        :param pID: Identificador del nodo a crear (Número de nodo actual)
        """
        this.x = pX
        this.y = pY
        this.restraints = pRestraints
        this.gdl = np.array([-1, -1, -1])#todo kejesto,
        this.cargas = np.array([[0], [0], [0]])
        this.ID = pID

    def definirCargas(this, pX, pY, pM, remplazar):
        """
    Función para asignar cargas puntuales en los nodos creados
        :param pX: Magnitud de carga puntual en x en kiloPascales
        :param pY: Magnitud de carga puntual en y en kiloPascales
        :param pM: Magnitud de momento puntual en kiloPascales-metro
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        if remplazar:
            this.cargas = np.array([[pX], [pY], [pM]])
        else:
            this.cargas = this.cargas + np.array([[pX], [pY], [pM]])

    def calcularFn(this, ngdl):
        """
    Función para definir el vector de fuerzas nodales
        :param ngdl: Número de grados de libertad de toda la estructura
        """
        this.Fn = np.zeros([ngdl, 1])
        for i in range(0, 3):
            this.Fn[this.gdl[i]] = this.cargas[i] #TODO Pinche nomenclatura


class Material:
    "Clase auxiliar para definir el material de los elementos"
    def __init__(this, nombre, E, v, alfa, gamma):
        """
    Método de inicialización de los materiales
        :param nombre: Nombre del material a crear
        :param E: Modulo de elasticidad del material a crear en kiloPascales
        :param v: Coeficiente de Poisson
        :param alfa: Coeficiente de expansión térmica en 1/°C
        :param gamma: Peso específico del material en kiloNewtons / metro cúbico
        """
        this.nombre = nombre
        this.E = E
        this.v = v
        this.alfa = alfa
        this.gamma = gamma


class Elemento:
    "Clase para definir los elementos de la estructura y sus atributos"
    def __init__(this, pSeccion, pNodoI, pNodoF, pTipo, pApoyoIzquierdo, pApoyoDerecho, pZa, pZb, pDefCortante):
        """
    Método de inicialización de los elementos
        :param pSeccion: Sección del elemento (creada anteriormente)
        :param pNodoI: Indicador del nodo inicial (creado anteriormente)
        :param pNodoF: Indicador del nodo final (creado anteriormente)
        :param pTipo: Tipo de elemento (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
        :param pApoyoIzquierdo:TODO
        :param pApoyoDerecho:TODO
        :param pZa:TODO
        :param pZb:TODO
        :param pDefCortante:TODO
        """
        this.seccion = pSeccion
        this.Area = this.seccion.area
        this.Inercia = this.seccion.inercia
        this.E = this.seccion.material.E
        this.Tipo = pTipo
        this.nodoI = pNodoI
        this.nodoF = pNodoF
        this.cargasDistribuidas = np.array([[0, 0], [0, 0]])
        this.cargasPuntuales = np.array([])
        this.defCortante = pDefCortante
        this.factorZonasRigidas(pApoyoIzquierdo, pApoyoDerecho, pZa, pZb)
        if not (this.nodoF.x - this.nodoI.x == 0):
            this.Angulo = np.arctan((this.nodoF.y - this.nodoI.y) / (this.nodoF.x - this.nodoI.x))
        else:
            if (this.nodoF.y > this.nodoI.y):
                this.Angulo = np.pi / 2
            else:
                this.Angulo = -np.pi / 2
        this.Longitud = ((this.nodoF.x - this.nodoI.x) ** 2 + (this.nodoF.y - this.nodoI.y) ** 2) ** (1 / 2)
        la = this.za * this.apoyoIzquierdo
        lb = this.zb * this.apoyoDerecho
        this.Longitud = this.Longitud - la - lb
        this.crearDiccionario()
        this.crearLambda()
        this.wx = 0
        this.wy = 0
        this.f = 0
        this.p0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
        this.P0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
        if this.Tipo == Tipo.UNO:
            this.alfa = 12
            this.beta = 4
            this.gamma = 6
            this.fi = 1
            this.fidiag = 1
        elif this.Tipo == Tipo.DOS:
            this.alfa = 3
            this.beta = 3
            this.gamma = 3
            this.fi = 0
            this.fidiag = 1
        elif this.Tipo == Tipo.TRES:
            this.alfa = 3
            this.beta = 3
            this.gamma = 3
            this.fi = 1
            this.fidiag = 0
        else:
            this.alfa = 0
            this.beta = 0
            this.gamma = 0
            this.fi = 0
            this.fidiag = 0
        this.calcularMatrizElemento()
    def hallarKb(this,psi=0):
        E = this.E
        I = this.Inercia
        A = this.Area
        L = this.Longitud
        if psi < 0.001:
            kb1 = 4*E*I/L
            kb2 = 2*E*I/L
            kb3 = 3*E*I/L
        else:
            kb1=((E*I)/(L))*((psi*(np.sin(psi)-psi*np.cos(psi)))/(2-2*np.cos(psi)-psi*np.sin(psi)))
            kb2=(E*I*(psi*(psi-np.sin(psi))))/(L*(2-2*np.cos(psi)-psi*np.sin(psi)))
            kb3=(E*I*(psi**2*np.sin(psi)))/(L*(np.sin(psi)-psi*np.cos(psi)))
        if this.Tipo == Tipo.UNO:
            kb = np.array([[E*A/L,0,0],[0,kb1,kb2],[0,kb2,kb1]])
        elif this.Tipo == Tipo.DOS:
            kb = np.array([[E*A/L,0,0],[0,0,0],[0,0,kb3]])
        elif this.Tipo == Tipo.TRES:
            kb = np.array([[E*A/L,0,0],[0,0,kb3],[0,0,0]])
        else:
            kb = np.array([[E*A/L,0,0],[0,0,0],[0,0,0]])
        return kb
    def determinarV0(this):
        this.kb0 = this.hallarKb()
        this.q0 = this.p0[np.ix_([3,2,5]),0].T
        this.v0 = np.dot(np.linalg.inv(this.kb0),this.q0)
    def calcularv(this):

        this.deltax0 = this.Longitud*np.cos(this.Angulo)
        this.deltay0 = this.Longitud*np.sin(this.Angulo)
        this.deltaux = this.Ue[3]-this.Ue[0]
        this.deltauy = this.Ue[4]-this.Ue[1]
        this.deltax = this.deltaux+this.deltax0
        this.deltay = this.deltauy+this.deltay0
        this.L = np.sqrt(this.deltax**2+this.deltay**2)
        this.theta = np.arcsin(this.deltay/this.L)
        this.deltatheta = this.theta-this.Angulo

        if this.Tipo == Tipo.UNO:
            v1 = this.L-this.Longitud
            v2 = this.Ue[2]-this.deltatheta
            v3 = this.Ue[5]-this.deltatheta
        elif this.Tipo == Tipo.DOS:
            v1 = this.L-this.Longitud
            v3 = this.Ue[5]-this.deltatheta
            v2 = -v3/2+this.v0[2][0]/2+this.v0[1][0]
        elif this.Tipo == Tipo.TRES:
            v1 = this.L-this.Longitud
            v2 = this.Ue[2]-this.deltatheta
            v3 = -v2/2+this.v0[1][0]/2+this.v0[2][0]
        else:
            v1 = this.L-this.Longitud
            v2 = 0
            v3 = 0
        this.v = np.array([[v1],[v2],[v3]])
    def calcularMatrizElemento(this):
        """
    Función para generar la matriz de rigidez del elemento
        """
        E = this.E
        v = this.seccion.material.v
        As = this.seccion.As
        A = this.Area
        I = this.Inercia
        t = this.Angulo

        L = this.Longitud

        alfa = this.alfa
        beta = this.beta
        gamma = this.gamma
        fi = this.fi
        fidiag = this.fidiag
        this.G = E / (2 + 2 * v)
        #TODO REforzar el concepto de mu
        mu = (12 * E * I) / (this.G * As * L ** 2) * this.defCortante

        c = np.cos(0)
        s = np.sin(0)
        k1 = (E * A / L) * (c ** 2) + alfa * E * I / ((1 + mu) * L ** 3) * s ** 2
        k2 = E * A / (L) * (s ** 2) + alfa * E * I / ((1 + mu) * L ** 3) * (c ** 2)
        k3 = (beta + mu) * E * I / ((1 + mu) * L)
        k4 = (E * A / L - alfa * E * I / ((1 + mu) * L ** 3)) * s * c
        k5 = gamma * E * I / ((1 + mu) * L ** 2) * c
        k6 = gamma * E * I / ((1 + mu) * L ** 2) * s
        #TODO revisar nuevas inplicacones del k7
        k7 = (beta / 2 - mu) * E * I / ((1 + mu) * L)
        this.Ke = np.array([[k1, k4, -k6 * fi, -k1, -k4, -k6 * fidiag], [k4, k2, k5 * fi, -k4, -k2, k5 * fidiag],
                            [-k6 * fi, k5 * fi, k3 * fi, k6 * fi, -k5 * fi, fi * k7 * fidiag],
                            [-k1, -k4, k6 * fi, k1, k4, k6 * fidiag], [-k4, -k2, -k5 * fi, k4, k2, -k5 * fidiag],
                            [-k6 * fidiag, k5 * fidiag, fidiag * k7 * fi, k6 * fidiag, -k5 * fidiag, k3 * fidiag]])
        #TODO revisar rigidez
        this.lbdaz = np.array([[1, 0, 0, 0, 0, 0], [0, 1, this.za, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, -this.zb], [0, 0, 0, 0, 0, 1]])
        #TODO ke loko
        this.Ke = np.dot(np.dot(np.dot(this.lbdaz.T, this.lbda.T), this.Ke), np.dot(this.lbdaz, this.lbda))

    #TODO como funciona el diccionario
    def crearDiccionario(this):
        """
    Función que genera la nomenclatura de grados de libertad correpondientes al elemento
        """
        this.diccionario = np.array(
            [this.nodoI.gdl[0], this.nodoI.gdl[1], this.nodoI.gdl[2], this.nodoF.gdl[0], this.nodoF.gdl[1],
             this.nodoF.gdl[2]])

    def matrizSuma(this, ngdl):
        """
    Función que simula la matriz de rigidez de la estructura, únicamente con el valor de las rigideces del elemento actual
        :param ngdl: Número de grados de libertad total de la estructura
        """
        this.kee = np.zeros([ngdl, ngdl])
        for i in range(0, 6):
            for j in range(0, 6):
                this.kee[this.diccionario[i], this.diccionario[j]] = this.Ke[i, j]

    def crearLambda(this):
        """
    Función que crea la matriz de transformación de coordenadas globales a locales Lambda del elemento
        """
        t = this.Angulo
        c = np.cos(t)
        s = np.sin(t)
        this.lbda = np.zeros([6, 6])
        this.lbda[0, 0] = c
        this.lbda[0, 1] = s
        this.lbda[1, 0] = -s
        this.lbda[1, 1] = c
        this.lbda[2, 2] = 1
        this.lbda[3, 3] = c
        this.lbda[3, 4] = s
        this.lbda[4, 3] = -s
        this.lbda[4, 4] = c
        this.lbda[5, 5] = 1
    def fuerzasBasicas(this):
        q1 = this.E*this.Area/this.Longitud*(this.v[0][0]-this.v0[0][0])
        q1 = np.min([q1,-1*10**-5])
        this.psi = np.sqrt(-q1*this.Longitud**2/this.E/this.Inercia)
        this.kb = this.hallarKb(this.psi)
        this.q = np.dot(this.kb,this.v-this.v0)
        c = np.cos(this.theta)
        s = np.sin(this.theta)
        l = this.L
        this.T = np.array([[-c,-s/l,-s/l],[-s,c/l,c/l],[0,1,0],[c,s/l,s/l],[s,-c/l,-c/l],[0,0,1]]).T

        this.lbd = np.zeros([6, 6])
        this.lbd[0, 0] = c
        this.lbd[0, 1] = s
        this.lbd[1, 0] = -s
        this.lbd[1, 1] = c
        this.lbd[2, 2] = 1
        this.lbd[3, 3] = c
        this.lbd[3, 4] = s
        this.lbd[4, 3] = -s
        this.lbd[4, 4] = c
        this.lbd[5, 5] = 1
    def matrizYFuerzas(this):
        matrizMaterial  = np.dot(this.T.T,np.dot(this.kb,this.T))
        c = np.cos(this.theta)
        s = np.sin(this.theta)
        l = this.L
        A = np.array([[1-c**2,-s*c,0,c**2-1,s*c,0],[-s*c,1-s**2,0,s*c,s**2-1,0],[0,0,0,0,0,0],[c**2-1,s*c,0,1-c**2,-s*c,0],[s*c,s**2-1,0,-s*c,1-s**2,0],[0,0,0,0,0,0]])
        B = np.array([[-2*s*c,c**2-s**2,0,2*s*c,s**2-c**2,0],[c**2-s**2,2*s*c,0,s**2-c**2,-2*s*c,0],[0,0,0,0,0,0],[2*s*c,s**2-c**2,0,-2*c*s,c**2-s**2,0],[s**2-c**2,-2*c*s,0,c**2-s**2,2*s*c,0],[0,0,0,0,0,0]])
        parteAxial = A*this.q[0][0]/l+B*(this.q[2][0]+this.q[1][0])/(l**2)
        this.matrizMaterial = matrizMaterial
        this.matrizGlobal = parteAxial
        this.Ke1 = parteAxial + matrizMaterial
        this.p1 = np.dot(this.T.T,this.q)+np.dot(this.lbd,this.p0)
        return this.Ke1,this.p1
    def calcularVectorDeFuerzas(this):
        """
    Función que calcula el vector de fuerzas del elemento en coordenadas globales
        """
        this.P0 = np.dot(np.dot(this.lbda.T, this.lbdaz.T), this.p0) #TODO Desglozar esto

    def definirCargas(this, pWx, pWy, pF, remplazar):
        """
    #TODO da pereza hacerlo ahora xdxd
        :param pWx:
        :param pWy:
        :param pF:
        :param remplazar:
        """
        this.wx = pWx
        this.wy = pWy
        this.f = pF
        wx = this.wx
        wy = this.wy
        l = this.Longitud
        f = this.f
        if this.Tipo == Tipo.UNO:
            p0 = np.array(
                [[-wx * l / 2], [wy * l / 2], [wy * l ** 2 / 12], [-wx * l / 2], [wy * l / 2], [-wy * l ** 2 / 12]])
            p0 = p0 + np.array([[0], [f], [2 * f * l / 9], [0], [f], [-2 * f * l / 9]])
        elif this.Tipo == Tipo.DOS:
            p0 = np.array([[-wx * l / 2], [3 * wy * l / 8], [0], [-wx * l / 2], [5 * wy * l / 8], [-wy * l ** 2 / 8]])
            p0 = p0 + np.array([[0], [2 * f / 3], [0], [0], [4 * f / 3], [-f * l / 3]])
        elif this.Tipo == Tipo.TRES:
            p0 = np.array([[-wx * l / 2], [5 * wy * l / 8], [wy * l ** 2 / 8], [-wx * l / 2], [3 * wy * l / 8], [0]])
            p0 = p0 + np.array([[0], [4 * f / 3], [f * l / 3], [0], [2 * f / 3], [0]])
        elif this.Tipo == Tipo.CUATRO:
            p0 = np.array([[-wx * l / 2], [wy * l / 2], [0], [-wx * l / 2], [wy * l / 2], [0]])
            p0 = p0 + np.array([[0], [f], [0], [0], [f], [0]])
        if remplazar:
            this.p0 = p0
            if not pWy == 0:
                this.cargasDistribuidas = np.append([], np.array([[0, -pWy], [this.Longitud, -pWy]]))
            if not pF == 0:
                this.cargasPuntuales = np.append([], np.array([this.Longitud / 3, -pF]))
                this.cargasPuntuales = np.append([], np.array([2 * this.Longitud / 3, -pF]))
        else:
            this.p0 = this.p0 + p0
            # TODO Mom come pick me up, I'm scared.
            if not pWy == 0:
                this.cargasDistribuidas = np.append(this.cargasDistribuidas,
                                                    np.array([[0, -pWy], [2, 1]]).reshape([2, 2]))
            if not pF == 0:
                this.cargasPuntuales = np.append(this.cargasPuntuales, np.array([this.Longitud / 3, -pF]))
                this.cargasPuntuales = np.append(this.cargasPuntuales, np.array([2 * this.Longitud / 3, -pF]))
        this.calcularVectorDeFuerzas()

    def factorZonasRigidas(this, apoyoIzquierdo, apoyoDerecho, za, zb):
        """
#TODO voy en orden
        :param apoyoIzquierdo:
        :param apoyoDerecho:
        :param za:
        :param zb:
        """
        this.za = za
        this.zb = zb
        this.apoyoIzquierdo = apoyoIzquierdo
        this.apoyoDerecho = apoyoDerecho

    def agregarCargaPuntual(this, f, x, remplazar):
        """
    Función que agrega una sola carga puntual a una distancia x del elemento (agregar una a una)
        :param f: Magnitud de la fuerza en cuestión en kiloPascales
        :param x: Distancia de la fuerza en cuestión desde el nodo inicial hasta el nodo final en metros
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        l = this.Longitud

        if this.Tipo == Tipo.UNO:
            p0 = np.array([[0], [f - f * x ** 2 * (3 * l - 2 * x) / l ** 3], [f * x * (l - x) ** 2 / l ** 2], [0],
                           [f * x ** 2 * (3 * l - 2 * x) / l ** 3], [-f * x ** 2 * (l - x) / l ** 2]])
        elif this.Tipo == Tipo.DOS:
            p0 = np.array([[0], [f * (l - x) ** 2 * (2 * l + x) / (2 * l ** 3)], [0], [0],
                           [f * x * (3 * l ** 2 - x ** 2) / (2 * l ** 3)], [-f * x * (l ** 2 - x ** 2) / (2 * l ** 2)]])
        elif this.Tipo == Tipo.TRES:
            p0 = np.array([[0], [f * (l - x) * (2 * l ** 2 + 2 * l * x - x ** 2) / (2 * l ** 3)],
                           [f * x * (l - x) * (2 * l - x) / (2 * l ** 2)], [0],
                           [f * x ** 2 * (3 * l - x) / (2 * l ** 3)], [0]])
        elif this.Tipo == Tipo.CUATRO:
            p0 = np.array([[0], [f * (l - x) / l], [0], [0], [f * x / l], [0]])
        if remplazar:
            this.p0 = p0
        else:
            this.p0 = this.p0 + p0
        this.calcularVectorDeFuerzas()

    def agregarDefectoDeFabricacion(this, e0, fi0, remplazar):
        """
    #Todo vamos en orden
        :param e0:
        :param fi0:
        :param remplazar:
        """
        L = this.Longitud
        E = this.seccion.material.E
        A = this.seccion.area
        I = this.seccion.inercia
        if this.Tipo == Tipo.UNO:
            p0 = np.array([[E * A * e0], [0], [E * I * fi0], [-E * A * e0], [0], [-E * I * fi0]])
        elif this.Tipo == Tipo.DOS:
            p0 = np.array([[E * A * e0], [-3 * E * I * fi0 / (2 * L)], [0], [-E * A * e0], [3 * E * I * fi0 / (2 * L)],
                           [-3 * E * I * fi0 / 2]])
        elif this.Tipo == Tipo.TRES:
            p0 = np.array([[E * A * e0], [3 * E * I * fi0 / (2 * L)], [3 * E * I * fi0 / 2], [-E * A * e0],
                           [-3 * E * I * fi0 / (2 * L)], [0]])
        elif this.Tipo == Tipo.CUATRO:
            p0 = np.array([[E * A * e0], [0], [0], [-E * A * e0], [0], [0]])
        if remplazar:
            this.p0 = p0
        else:
            this.p0 = this.p0 + p0
        this.calcularVectorDeFuerzas()

    def agregarCargaPorTemperatura(this, pDeltaT0, pDeltaTFh, remplazar):
        """
#TODO mk esto no acaba
        :param pDeltaT0:
        :param pDeltaTFh:
        :param remplazar:
        """
        this.deltaT0 = pDeltaT0
        this.deltaTFh = pDeltaTFh
        dt0 = this.deltaT0
        dtf = this.deltaTFh
        e0 = this.seccion.material.alfa * dt0
        fi0 = this.seccion.material.alfa * pDeltaTFh
        this.agregarDefectoDeFabricacion(e0, fi0, remplazar)

    def agregarCargaPresfuerzoAxial(this, q0, remplazar):
        """
#TODO mk me perdi arto
        :param q0:
        :param remplazar:
        """
        p0 = np.array([[-q0], [0], [0], [q0], [0], [0]])
        if remplazar:
            this.p0 = p0
        else:
            this.p0 = this.p0 + p0
        this.calcularVectorDeFuerzas()

    def agregarCargaPostensadoFlexionYAxial(this, f0, e1, e2, e3, remplazar):
        """
#TODO son muchos metodos
        :param f0:
        :param e1:
        :param e2:
        :param e3:
        :param remplazar:
        """
        L = this.Longitud
        c = e1
        e2 = -e2
        b = -2 / L * (np.sqrt(e1 - e2)) * (np.sqrt(e1 - e2) + np.sqrt(e3 - e2))
        a = (e3 - e1 - b * L) / (L ** 2)
        dedx = lambda x: 2 * a * x + b
        f0x = np.cos(np.arctan(dedx(0))) * f0
        if this.Tipo == Tipo.UNO:
            p0 = np.array([[-f0x], [-(a * L + b) * f0], [-(a * L ** 2 - 6 * c) * f0 / 6], [f0x], [(a * L + b) * f0],
                           [-(5 * a * L ** 2 + 6 * b * L + 6 * c) * f0 / 6]])
        elif this.Tipo == Tipo.DOS:
            p0 = np.array([[-f0x], [-(3 * a * L ** 2 + 4 * b * L + 6 * L) * f0 / (4 * L)], [0], [f0x],
                           [(3 * a * L ** 2 + 4 * b * L + 6 * c) * f0 / (4 * L)],
                           [-(3 * a * L ** 2 + 4 * b * L + 6 * c) * f0 / 4]])
        elif this.Tipo == Tipo.TRES:
            p0 = np.array(
                [[-f0x], [(a * L ** 2 + 2 * b * L + 6 * c) * f0 / (4 * L)], [(a * L ** 2 + 2 * b * L + 6 * c) * f0 / 4],
                 [f0x], [-(a * L ** 2 + 2 * b * L + 6 * c) * f0 / (4 * L)], [0]])
        elif this.Tipo == Tipo.CUATRO:
            p0 = np.array([[-f0x], [0], [0], [f0x], [0], [0]])
        if remplazar:
            this.p0 = p0
        else:
            this.p0 = this.p0 + p0
        this.calcularVectorDeFuerzas()

    def calcularF0(this, ngdl):
        """
    Función para calcular el vector de fuerzas generadas por cargas externas por elemento (teniendo en cuenta todos los gdl de la estructura)
        :param ngdl: Número de grados de libertad de la estructura
        """
        this.F0 = np.zeros([ngdl, 1])
        for i in range(0, 6):
            this.F0[this.diccionario[i], 0] = this.P0[i, 0]

    def calcularVectorDeFuerzasInternas(this, U):
        """
    Función para calcular el vector de fuerzas internas del elemento
        :param U: Vector de desplazamientos calculados para todos los grados de libertad de la estructura
        """
        this.Ue = U[this.diccionario]
        parcial = np.dot(this.Ke, this.Ue)
        this.P = np.reshape(parcial, [parcial.size, 1]) + this.P0
        this.p = np.dot(this.lbda, this.P)

    def solucionarFEM(this, P, n=50):
        """
    Solución de diagramas de fuerzas internas por medio de elementos finitos
        :param P: #TODO: Arturo
        :param n:
        """
        he = this.Longitud / (n - 1)
        K = tridiag(-1 / he, 2 / he, -1 / he, n)
        F = np.zeros([n, 1])
        for i in range(0, n):
            F[i, 0] = P(he * i) * he
        F = F - this.p[2] * K[:, 0].reshape([K[:, 0].size, 1])
        F = F + this.p[-1] * K[:, -1].reshape([K[:, -1].size, 1])
        F[0] = this.p[2]
        F[-1] = -this.p[-1]
        K[0, 0] = 1
        K[0, 1] = 0
        K[1, 0] = 0
        K[-1, -1] = 1
        K[-1, -2] = 0
        K[-2, -1] = 0
        this.DMomentos = np.dot(np.linalg.inv(K), F)
        this.DCortante = np.dot(K, this.DMomentos) - F


class Constraint:
    def __init__(this, tipo, nodoI, nodoF):
        """
TODO en orden mi perro
        :param tipo:
        :param nodoI:
        :param nodoF:
        """
        this.nodoI = nodoI
        this.nodoF = nodoF
        this.tipo = tipo
        if not (nodoF.x - nodoI.x == 0):
            this.Angulo = np.arctan((nodoF.y - nodoI.y) / (nodoF.x - nodoI.x))
        else:
            if (nodoF.y > nodoI.y):
                this.Angulo = np.pi / 2
            else:
                this.Angulo = -np.pi / 2
        this.r = np.zeros([3, 6])
        t = this.Angulo
        c = np.cos(t)
        s = np.sin(t)
        this.independientes = [this.nodoI.gdl[0], this.nodoI.gdl[1], this.nodoI.gdl[2], this.nodoF.gdl[2]]
        this.dependientes = [this.nodoF.gdl[0], this.nodoF.gdl[1]]


class SuperElemento:
    def __init__(this, SE, SF, gdl):
        """
TODO: En super orden perro
        :param SE:
        :param SF:
        :param gdl:
        """
        this.SE = SE
        this.SF = SF
        this.gdl = gdl

    def calcularKSUMA(this, n):
        a = np.zeros([n, n])
        a[np.ix_(this.gdl, this.gdl)] = this.SE
        return a

    def calcularF0(this, n):
        a = np.zeros([n, 1])
        a[np.ix_(this.gdl)] = this.SF
        return a


class Estructura:
    "Clase que representa una estructura."
    def __init__(this):
        """
    Método de inicialización de la estructura
        """
        this.nodos = np.array([])
        this.resortes = np.array([])
        this.elementos = np.array([])
        this.Ur = np.array([])
        this.constraints = np.array([])
        this.superelementos = np.array([])

    def agregarNodo(this, x, y, fix=[True, True, True]):
        """
    Función que agrega un nodo a la estructura
        :param x: Posición x del nodo
        :param y: Posición y del nodo
        :param fix: Condiciones de apoyo del nodo (True = restringido, False = libre)
        """
        this.nodos = np.append(this.nodos, Nodo(x, y, fix, this.nodos.size))
        this.actualizarGDL()
        this.actualizarElementos()
        this.actualizarResortes()
        this.Ur = np.zeros([this.restringidos.size, 1])
    def newton(this,param,semilla=None):
        try:
            if semilla == None:
                Ul = np.zeros([this.libres.size])
            else:
                Ul = semilla
        except:
            Ul = semilla
        Ur = np.zeros([this.restringidos.size])
        Fn = this.Fn[np.ix_(this.libres)]
        for i in range(0,param[0]):
            for i in this.elementos:
                U = np.append(Ul,Ur)
                i.Ue = U[np.ix_(i.diccionario)]
            Kll , P = this.determinacionDeEstado()
            A = np.dot(np.linalg.inv(Kll),(Fn-P))
            Ul = Ul + A.T
        return Ul.T
    def determinacionDeEstado(this):
        n=this.libres.size+this.restringidos.size
        Kll = np.zeros([n,n])
        Pl = np.zeros([n,1])
        for i in this.elementos:
            i.determinarV0()
            i.calcularv()
            i.fuerzasBasicas()
            [Ke, P] = i.matrizYFuerzas()
            Kll[np.ix_(i.diccionario,i.diccionario)] = Kll[np.ix_(i.diccionario,i.diccionario)] + Ke
            Pl[np.ix_(i.diccionario)] = Pl[np.ix_(i.diccionario)] + P
        return Kll[np.ix_(this.libres,this.libres)], Pl[np.ix_(this.libres)]

    def actualizarElementos(this):
        """TODO: arturo y esto porque (todos los metodos de actualizar)?
        Esta sirve para poder agregar cargas antes de definir la estructura completa, es un metodo que "sobra" pero tenerlo es ventajoso a la hora de definir una estructura. 
    Función que actualiza constantemente los atributos de los elementos en caso de existir una modificación
        """
        for i in range(0, this.elementos.size):
            parcial = this.elementos[i]
            this.elementos[i] = Elemento(this.elementos[i].seccion, this.nodos[this.elementos[i].nodoI.ID],
                                         this.nodos[this.elementos[i].nodoF.ID], this.elementos[i].Tipo,
                                         this.elementos[i].apoyoIzquierdo, this.elementos[i].apoyoDerecho,
                                         this.elementos[i].za, this.elementos[i].zb, this.elementos[i].defCortante)
            this.elementos[i].p0 = parcial.p0
            this.elementos[i].calcularVectorDeFuerzas()

    def actualizarResortes(this):
        """
    Función que actualiza constantemente los atributos de los resortes en caso de existir una modificación
        """
        for i in range(0, this.resortes.size):
            this.resortes[i] = Resorte(this.nodos[this.resortes[i].nodo.ID], this.resortes[i].rigidez,
                                       this.resortes[i].completo)

    def agregarElemento(this, seccion, nodoInicial, nodoFinal, tipo=Tipo.UNO, apoyoIzquierdo=0, za=0, apoyoDerecho=0, zb=0,
                        defCortante=True):
        """
    Función que permite añadir un nuevo elemento a la estructura
        :param seccion: Sección del elemento a añadir
        :param nodoInicial: Identificador del nodo inicial del elemento
        :param nodoFinal: Identificador del nodo final del elemento
        :param tipo: Tipo de elemento a crear (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
        :param apoyoIzquierdo: #Todo Vamos por partes
        :param za:
        :param apoyoDerecho:
        :param zb:
        :param defCortante:
        """
        this.elementos = np.append(this.elementos,
                                   Elemento(seccion, this.nodos[nodoInicial], this.nodos[nodoFinal], tipo,
                                            apoyoIzquierdo, apoyoDerecho, za, zb, defCortante))

    def agregarResorte(this, rigidez, nodo=-1, completo=False):
        """
    Función que permite añadir un nuevo resorte a la estructura
        :param rigidez: Vector con las magnitudes de las rigideces del resorte en kiloPascales (Kx,Ky,Km)
        :param nodo: Nodo sobre el que se va a agregar el resorte
        :param completo: #TOdo arturo
        """
        this.resortes = np.append(this.resortes, Resorte(this.nodos[nodo], rigidez, completo))

    def agregarSuperElementos(this, SE, SF, gdl):
        """
TODO por partes
        :param SE:
        :param SF:
        :param gdl:
        """
        supelemento = SuperElemento(SE, SF, gdl)
        this.superelementos = np.append(this.superelementos, supelemento)

    def definirConstraint(this, tipo, nodoInicial, nodoFinal):
        """
TODO aun no
        :param tipo:
        :param nodoInicial:
        :param nodoFinal:
        """
        nodoF = this.nodos[nodoFinal]
        nodoI = this.nodos[nodoInicial]
        constraint = Constraint(tipo, nodoI, nodoF)
        this.constraints = np.append(this.constraints, constraint)

    def crearMatrizDeRigidez(this):
        """
TOdo ahi vamos
        """
        n = this.nodos.size * 3
        this.KE = np.zeros([n, n])
        for i in range(0, this.elementos.size):
            this.elementos[i].matrizSuma(n)
            this.KE = this.KE + this.elementos[i].kee
        for i in range(0, this.resortes.size):
            this.resortes[i].calcularKe(n)
            this.KE = this.KE + this.resortes[i].Kee
        for i in range(0, this.superelementos.size):
            this.KE = this.KE + this.superelementos[i].calcularKSUMA(n)

    def definirCambiosTemperatura(this, inicial, interior, exterior, h, elemento=-1, remplazar=False):
        """
Todo casi
        :param inicial:
        :param interior:
        :param exterior:
        :param h:
        :param elemento:
        :param remplazar:
        """
        dta = 1 / 2 * ((exterior - inicial) + (interior - inicial))
        dtf = (interior - exterior)
        this.elementos[elemento].agregarCargaPorTemperatura(dta, dtf / h, remplazar)

    def agregarCargaPorTemperatura(this, deltaT0, deltaThf, elemento=-1, remplazar=False):
        """
Todo mierda
        :param deltaT0:
        :param deltaThf:
        :param elemento:
        :param remplazar:
        """
        this.elementos[elemento].agregarCargaPorTemperatura(deltaT0, deltaThf, remplazar)

    def agregarDefectoDeFabricacion(this, e0=0, fi0=0, elemento=-1, remplazar=False):
        """
Todo coño
        :param e0:
        :param fi0:
        :param elemento:
        :param remplazar:
        """
        this.elementos[elemento].agregarDefectoDeFabricacion(e0, fi0, remplazar)

    def agregarCargaPresfuerzoAxial(this, el, q0, elemento=-1, remplazar=False):
        """
TOdo cuanto mas
        :param el:
        :param q0:
        :param elemento:
        :param remplazar:
        """
        this.elementos[elemento].agregarCargaPresfuerzoAxial(q0, remplazar)

    def agregarCargaPostensadoFlexionYAxial(this, f0, e1, e2, e3, elemento=-1, remplazar=False):
        """
Todo marika ya
        :param f0:
        :param e1:
        :param e2:
        :param e3:
        :param elemento:
        :param remplazar:
        """
        this.elementos[elemento].agregarCargaPostensadoFlexionYAxial(f0, e1, e2, e3, remplazar)

    def agregarCargaElemento(this, wx=0, wy=0, ftercios=0, elemento=-1, remplazar=False):
        """
    Función para agregar cargas distribuidas o a tercios del elemento
        :param wx: Magnitud de la carga distribuida en dirección x en kiloNewtons/metros
        :param wy: Magnitud de la carga distribuida en dirección y en kiloNewtons/metros
        :param ftercios: Magnitud de las cargas aplicados a tercios de la longitud del elemento en kiloNewtons
        :param elemento: Identificador del elemento sobre el cual se aplica/n la/s carga/s
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        this.elementos[elemento].definirCargas(wx, wy, ftercios, remplazar)

    def agregarCargaNodo(this, nodo=-1, px=0, py=0, m=0, remplazar=False):
        """
    Función que permite agregar cargas a los nodos de la estructura
        :param nodo: Identificador del nodo sobre el cual se va a agregar la carga puntual
        :param px: Magnitud de la carga en x en kiloNewtons
        :param py: Magnitud de la carga en y en kiloNewtons
        :param m: Magnitud del momento aplicado en kiloNewtons-metros
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        this.nodos[nodo].definirCargas(px, py, m, remplazar)

    def agregarCargaPuntual(this, f, x, elemento=-1, remplazar=False):
        """
    Función que permite agregar una carga puntual a una distancia determinada de un elemento de la estructura
        :param f: Magnitud de la carga puntual en kiloNewtons
        :param x: Ubicación de la fuerza a aplicar desde el nodo inical hasta el nodo final en metros
        :param elemento: Identificador del elemento sobre el cual se aplica la nueva carga puntual
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        this.elementos[elemento].agregarCargaPuntual(f, x, remplazar)

    def agregarCargaDistribuida(this, WX=0, WY=0, elemento=-1, remplazar=False):
        """
    Función que permite agregar una carga distribuida sobre un elemento de la estructura
        :param WX: Magnitud de la carga distribuida en dirección x en kiloNewtons/metros
        :param WY: Magnitud de la carga distribuida en dirección y en kiloNewtons/metros
        :param elemento: Identificador del elemento sobre el cual se aplica/n la/s carga/s
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        s = np.sin(this.elementos[elemento].Angulo)
        c = np.cos(this.elementos[elemento].Angulo)
        this.elementos[elemento].definirCargas(-s * WY, c * WY, 0, remplazar)
        this.elementos[elemento].definirCargas(s * WX, c * WX, 0, False)

    def definirFactorPesoPropio(this, f=0, remplazar=False):
        """
    Función que permite incluir la carga agregada producida por el peso propio del elemento
        :param f:TODO no se que tan bien esta esto
        :param remplazar:
        """
        for i in range(0, this.elementos.size):
            sw = f * this.elementos[i].Area * this.elementos[i].seccion.material.gamma
            this.agregarCargaDistribuida(i, WY=sw, remplazar=remplazar)

    def calcularF0(this):
        """
    Función que calcula el vector de fuerzas externas globales de la estructura
        """
        n = this.nodos.size * 3
        this.F0 = np.zeros([n, 1])
        for i in range(0, this.elementos.size):
            this.elementos[i].calcularF0(n)
            this.F0 = this.F0 + this.elementos[i].F0
        for i in range(0, this.superelementos.size): #TODO superelementos
            this.F0 = this.F0 + this.superelementos[i].calcularF0(n)

    def calcularFn(this):
        """
    Función que calcula el vector de fuerzas nodales de la estructura
        """
        n = this.nodos.size * 3
        this.Fn = np.zeros([n, 1])
        for i in range(0, this.nodos.size):
            this.nodos[i].calcularFn(n)
            this.Fn = this.Fn + this.nodos[i].Fn

    def definirDesplazamientosRestringidos(this, desplazamientos):
        """#Todo: son indicadores o valores o ke
    Función que permite asignar desplazamientos restringidos a los grados de libertad restringidos
        :param desplazamientos: Vector de los desplazamientos restringidos (metros o Newtons-metros según el caso)
        """
        if desplazamientos.size == this.restringidos.size:
            this.Ur = desplazamientos
        else:
            'No se asignaron los desplazamientos restringidos porque el vector no tiene el mismo tamaño.'

    def definirDesplazamientoRestringido(this, nodo, gdl, valor):
        """
    TODO no entiendo
        :param nodo:
        :param gdl:
        :param valor:
        """
        if any(np.isin(this.restringidos, this.nodos[nodo].gdl[gdl])):
            for i in range(0, this.restringidos.size):
                if this.restringidos[i] == this.nodos[nodo].gdl[gdl]:
                    this.Ur[i] = valor
                    break
        else:
            print('No se asignaron los desplazamientos porque el gdl' + format(this.nodos[nodo].gdl[gdl]) + ' no hace parte de los grados de libertad restringidos, los grados de libertad restringidos disponibles son: ' + format(this.restringidos))

    def calcularSubmatrices(this):
        """
    Función para crear las submatrices de la matriz de rigidez de la estructura
        """
        this.Kll = this.KE[0:this.libres.size:1, 0:this.libres.size:1]
        this.Klr = this.KE[0:this.libres.size:1, this.libres.size:this.libres.size + this.restringidos.size:1]
        this.Krl = this.KE[this.libres.size:this.libres.size + this.restringidos.size:1, 0:this.libres.size:1]
        this.Krr = this.KE[this.libres.size:this.libres.size + this.restringidos.size:1,
                   this.libres.size:this.libres.size + this.restringidos.size:1]

    def calcularVectorDesplazamientosLibres(this):
        """
     Función que halla los desplazamientos de los grados de libertad libres de las estructura
        """
        if this.Ur.size == 0:
            this.Ur = np.zeros([this.restringidos.size, 1])
        this.Fl = this.Fn - this.F0
        this.Ul = np.dot(np.linalg.inv(this.Kll), (this.Fl[this.libres] - np.dot(this.Klr, this.Ur)))

    def calcularReacciones(this):
        """
    Función que calcula las reacciones de los grados de libertad restringidos de la estructura
        """
        this.R0 = this.F0[this.restringidos]
        this.Rn = np.dot(this.Krl, this.Ul) + np.dot(this.Krr, this.Ur) + this.R0

    def calcularVectoresDeFuerzasInternas(this):
        """
    Función que calcula las fuerzas internas de los elementos de la estructura
        """
        for i in range(0, this.elementos.size):
            this.elementos[i].calcularVectorDeFuerzasInternas(np.concatenate([this.Ul, this.Ur], axis=None))

    def hacerSuperElemento(this, gdlVisibles, gdlInvisibles):
        """
TOdo no aun
        :param gdlVisibles:
        :param gdlInvisibles:
        :return:
        """
        this.crearMatrizDeRigidez()
        this.calcularF0()
        this.calcularFn()
        this.calcularSubmatrices()
        this.Fl = this.Fn - this.F0
        klgc = np.dot(this.Kll[np.ix_(gdlVisibles, gdlInvisibles)],
                      (np.linalg.inv(this.Kll[np.ix_(gdlInvisibles, gdlInvisibles)])))
        a = this.Fl[np.ix_(gdlVisibles)] - np.dot(klgc, this.Fl[np.ix_(gdlInvisibles)])
        b = this.Kll[np.ix_(gdlVisibles, gdlVisibles)] - np.dot(klgc, this.Kll[np.ix_(gdlInvisibles, gdlVisibles)])
        return a, b

    def solucionar(this, verbose=True, dibujar=False, guardar=False, carpeta='Resultados',analisis='EL',iteraciones=100):
        """
    Función que resuelve el método matricial de rigidez de la estructura
        :param verbose: Opción para mostrar mensaje de análisis exitoso (True = mostrar, False = no mostrar)
        :param dibujar: Opción para realizar interfaz gráfica (True = mostrar, False = no mostrar)
        :param guardar: Opción para guardar los resultados del análisis (True = guardar, False = no guardar)
        :param carpeta: Dirección de la carpeta destinno
        """
        if analisis == 'EL':
            this.crearMatrizDeRigidez()
            this.calcularF0()
            this.calcularFn()
            this.calcularSubmatrices()
            this.calcularVectorDesplazamientosLibres()
            this.calcularVectoresDeFuerzasInternas()
            this.calcularReacciones()
            this.gdls = np.append(this.libres, this.restringidos)
            this.U = np.append(this.Ul, this.Ur)
        elif analisis == 'CR':
            this.solucionar(verbose=False, dibujar=False, guardar=False, carpeta='',analisis='EL',iteraciones=1)
            return this.newton([iteraciones])
        if verbose:
            print(
                'Se ha terminado de calcular, puedes examinar la variable de la estructura para consultar los resultados.')
        if dibujar:
            this.pintar()
        if guardar:
            this.guardarResultados(carpeta)

    def actualizarGDL(this):
        """
    #Todo analisar aqui ocurre la magia
        """
        count = 0
        this.libres = np.array([])
        this.restringidos = np.array([])
        for i in range(0, this.nodos.size):
            if this.nodos[i].restraints[0]:
                this.nodos[i].gdl[0] = count
                this.libres = np.append(this.libres, count)
                count = count + 1 * (this.nodos[i].restraints[0])
            if this.nodos[i].restraints[1]:
                this.nodos[i].gdl[1] = count
                this.libres = np.append(this.libres, count)
                count = count + 1 * (this.nodos[i].restraints[1])
            if this.nodos[i].restraints[2]:
                this.nodos[i].gdl[2] = count
                this.libres = np.append(this.libres, count)
                count = count + 1 * (this.nodos[i].restraints[2])
        for i in range(0, this.nodos.size):
            if not this.nodos[i].restraints[0]:
                this.nodos[i].gdl[0] = count
                this.restringidos = np.append(this.restringidos, count)
                count = count + 1
            if not this.nodos[i].restraints[1]:
                this.nodos[i].gdl[1] = count
                this.restringidos = np.append(this.restringidos, count)
                count = count + 1
            if not this.nodos[i].restraints[2]:
                this.nodos[i].gdl[2] = count
                this.restringidos = np.append(this.restringidos, count)
                count = count + 1
        this.libres = this.libres.astype(int)
        this.restringidos = this.restringidos.astype(int)

    def pintar(this):
        """
    Función que permite correr la interfaz de visualización de resultados de modelo, fuerzas internas y desplazamientos obtenidos
        """
        maxcoord = 0
        for i in this.elementos:
            if i.nodoI.x > maxcoord:
                maxcoord = i.nodoI.x
            if i.nodoI.y > maxcoord:
                maxcoord = i.nodoI.y
            if i.nodoF.x > maxcoord:
                maxcoord = i.nodoF.x
            if i.nodoF.y > maxcoord:
                maxcoord = i.nodoF.y

        margen = 100
        wiw = 600
        mult = (wiw - 2 * margen) / maxcoord
        import tkinter
        window = tkinter.Tk()
        window.title("Definicion de Estructura")
        canvas = tkinter.Canvas(window, width=wiw, height=wiw)
        count = 0
        tmañoNodo = 5
        r = 2
        for i in this.nodos:
            contador = 0
            for f in range(0, this.superelementos.size):
                k = this.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10),
                                                   margen + i.x * mult + 10 + 10 * (g + 1) / 2,
                                                   -margen + wiw - (i.y * mult + 10), fill="#C0C0C0", width=2)
                            canvas.create_line(margen + i.x * mult - 30, -margen + wiw - (i.y * mult),
                                               margen + i.x * mult - 20, -margen + wiw - (i.y * mult), fill="#C0C0C0",
                                               width=2)
                            contador = contador + 1
                            canvas.create_text(margen + i.x * mult - 40,
                                               -margen + wiw - i.y * mult + (contador - 1) * 8, fill="black",
                                               text='SE: ' + format(f), justify='center',
                                               font=("TkDefaultFont", tmañoNodo + 1))
                            flag = True
                            break
                    if flag:
                        break
        for i in this.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
            canvas.create_text(margen + xx * mult + 10, -margen + wiw - yy * mult - 10, fill="red", text=format(count))
            count = count + 1
            r = 3
            radio = tmañoNodo + r
            x = radio * np.cos(i.Angulo)
            y = np.tan(i.Angulo) * x
            j = i.nodoI
            k = i.nodoF
            if k.x < j.x:
                j = i.nodoF
                k = i.nodoI
            if i.Tipo == Tipo.DOS:
                canvas.create_oval(margen + j.x * mult + x - r, -margen + wiw - (j.y * mult + y) - r,
                                   margen + j.x * mult + x + r, -margen + wiw - (j.y * mult + y) + r, fill="lightblue")
            elif i.Tipo == Tipo.TRES:
                canvas.create_oval(margen + k.x * mult - x - r, -margen + wiw - (k.y * mult - y) - r,
                                   margen + k.x * mult - x + r, -margen + wiw - (k.y * mult - y) + r, fill="lightblue")
            elif i.Tipo == Tipo.CUATRO:
                canvas.create_oval(margen + j.x * mult + x - r, -margen + wiw - (j.y * mult + y) - r,
                                   margen + j.x * mult + x + r, -margen + wiw - (j.y * mult + y) + r, fill="lightblue")
                canvas.create_oval(margen + k.x * mult - x - r, -margen + wiw - (k.y * mult - y) - r,
                                   margen + k.x * mult - x + r, -margen + wiw - (k.y * mult - y) + r, fill="lightblue")
        r = 3
        for k in this.resortes:
            i = k.nodo
            canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult), margen + i.x * mult + 10,
                               -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
            for j in range(0, 3):
                canvas.create_line(margen + i.x * mult + 10 + 10 * j / 2, -margen + wiw - (i.y * mult + 10),
                                   margen + i.x * mult + 10 + 10 * j / 2, -margen + wiw - (i.y * mult - 10),
                                   fill="#52E3C4", width=2)
                canvas.create_line(margen + i.x * mult + 10 + 10 * j / 2, -margen + wiw - (i.y * mult - 10),
                                   margen + i.x * mult + 10 + 10 * (j + 1) / 2, -margen + wiw - (i.y * mult + 10),
                                   fill="#52E3C4", width=2)
            canvas.create_line(margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult + 10),
                               margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
            canvas.create_line(margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult), margen + i.x * mult + 30,
                               -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
        for i in this.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)

            if i.restraints == [False, False, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
            elif i.restraints == [True, False, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 1,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 1,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 5,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 5,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 9,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 9,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
            elif not i.restraints == [True, True, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   fill="blue", width=2)

                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult - 2 * tmañoNodo, -margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   fill="blue", width=2)

                canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult - tmañoNodo), margen + i.x * mult,
                                   -margen + wiw - (i.y * mult - 3 * tmañoNodo), fill="blue", width=2)
            canvas.create_text(margen + i.x * mult, -margen + wiw - i.y * mult, fill="black", text=format(i.ID),
                               justify='center', font=("TkDefaultFont", tmañoNodo + 1))
        canvas.pack()
        canvas.mainloop()

        window = tkinter.Tk()
        window.title("Reacciones")
        canvas = tkinter.Canvas(window, width=wiw, height=wiw)
        for i in this.nodos:
            contador = 0
            for f in range(0, this.superelementos.size):
                k = this.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10),
                                                   margen + i.x * mult + 10 + 10 * (g + 1) / 2,
                                                   -margen + wiw - (i.y * mult + 10), fill="#C0C0C0", width=2)
                            canvas.create_line(margen + i.x * mult - 30, -margen + wiw - (i.y * mult),
                                               margen + i.x * mult - 20, -margen + wiw - (i.y * mult), fill="#C0C0C0",
                                               width=2)
                            flag = True
                            break
                    if flag:
                        break
        for i in this.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
        for i in this.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)
            for j in this.restringidos:
                if j == i.gdl[1]:
                    canvas.create_line(margen + i.x * mult, -margen + wiw - i.y * mult - tmañoNodo, margen + i.x * mult,
                                       -margen + wiw - (i.y * mult + 50), fill="red", width=2)
                    canvas.create_text(margen + i.x * mult, -margen + wiw - (i.y * mult + 60),
                                       text=format(np.round(this.Rn[j - this.libres.size][0], 3)), fill="red")
                    canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50), margen + i.x*mult - 10,
                                       -margen + wiw - (i.y * mult + 40), fill="red", width=2)
                    canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50), margen + i.x * mult + 10,
                                       -margen + wiw - (i.y * mult + 40), fill="red", width=2)
                if j == i.gdl[0]:
                    canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - i.y * mult,
                                       margen + i.x * mult + 50, -margen + wiw - (i.y * mult), fill="green", width=2)
                    canvas.create_text(margen + i.x * mult + 80, -margen + wiw - (i.y * mult + 7),
                                       text=format(np.round(this.Rn[j - this.libres.size][0], 3)), fill="green")
                    canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult), margen + i.x * mult + 40,
                                       -margen + wiw - (i.y * mult + 10), fill="green", width=2)
                    canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult), margen + i.x * mult + 40,
                                       -margen + wiw - (i.y * mult - 10), fill="green", width=2)
                if j == i.gdl[2]:
                    canvas.create_text(margen + i.x * mult + 10, -margen + wiw - (i.y * mult - 13),
                                       text='M: ' + format(np.round(this.Rn[j - this.libres.size][0], 3)), fill="blue")
            canvas.create_text(margen + i.x * mult, -margen + wiw - i.y * mult, fill="black", text=format(i.ID),
                               justify='center', font=("TkDefaultFont", tmañoNodo + 1))
        canvas.pack()
        canvas.mainloop()

        window = tkinter.Tk()
        window.title("Desplazamientos")
        canvas = tkinter.Canvas(window, width=wiw, height=wiw)
        for i in this.nodos:
            contador = 0
            for f in range(0, this.superelementos.size):
                k = this.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10),
                                                   margen + i.x * mult + 10 + 10 * (g + 1) / 2,
                                                   -margen + wiw - (i.y * mult + 10), fill="#C0C0C0", width=2)
                            canvas.create_line(margen + i.x * mult - 30, -margen + wiw - (i.y * mult),
                                               margen + i.x * mult - 20, -margen + wiw - (i.y * mult), fill="#C0C0C0",
                                               width=2)
                            flag = True
                            break
                    if flag:
                        break
        for i in this.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
        for i in this.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)
            for j in this.gdls:
                if not this.U[j] == 0:
                    if j == i.gdl[1]:
                        canvas.create_line(margen + i.x * mult, -margen + wiw - i.y * mult - tmañoNodo,
                                           margen + i.x * mult, -margen + wiw - (i.y * mult + 50), fill="red", width=2)
                        canvas.create_text(margen + i.x * mult, -margen + wiw - (i.y * mult + 60),
                                           text=format(np.round(this.U[j], 4)), fill="red")
                        canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50),
                                           margen + i.x * mult - 10, -margen + wiw - (i.y * mult + 40), fill="red",
                                           width=2)
                        canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50),
                                           margen + i.x * mult + 10, -margen + wiw - (i.y * mult + 40), fill="red",
                                           width=2)
                    if j == i.gdl[0]:
                        canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - i.y * mult,
                                           margen + i.x * mult + 50, -margen + wiw - (i.y * mult), fill="green",
                                           width=2)
                        canvas.create_text(margen + i.x * mult + 70, -margen + wiw - (i.y * mult + 7),
                                           text=format(np.round(this.U[j], 4)), fill="green")
                        canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult),
                                           margen + i.x * mult + 40, -margen + wiw - (i.y * mult + 10), fill="green",
                                           width=2)
                        canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult),
                                           margen + i.x * mult + 40, -margen + wiw - (i.y * mult - 10), fill="green",
                                           width=2)
                    if j == i.gdl[2]:
                        canvas.create_text(margen + i.x * mult + 10, -margen + wiw - (i.y * mult - 13),
                                           text='R: ' + format(np.round(this.U[j], 4)), fill="blue")
            canvas.create_text(margen + i.x * mult, -margen + wiw - i.y * mult, fill="black", text=format(i.ID),
                               justify='center', font=("TkDefaultFont", tmañoNodo + 1))
        canvas.pack()
        canvas.mainloop()

    def guardarResultados(this, carpeta):
        """
    Función para generar y guardar un archivo con los resultados del análisis obtenido
        :param carpeta: Dirección de la carpeta destino en donde se guardarán los archivos creados
        """
        import os
        import shutil
        path = os.getcwd() + '/' + carpeta
        try:
            os.mkdir(path)
        except OSError:
            try:
                shutil.rmtree(path)
                os.mkdir(path)
            except OSError as e:
                print("Error: %s : %s" % (path, e.strerror))
            print("Se han guardado los resultados en: %s " % path)
        else:
            os.mkdir(path)
            print("Se han guardado los resultados en: %s " % path)
        np.savetxt(path + "/Rn.csv", this.Rn, delimiter=",")
        np.savetxt(path + "/Ul.csv", this.Ul, delimiter=",")
        np.savetxt(path + "/K.csv", this.KE, delimiter=",")
        np.savetxt(path + "/F0.csv", this.F0, delimiter=",")
        np.savetxt(path + "/Fn.csv", this.Fn, delimiter=",")
        for i in range(0, this.elementos.size):
            np.savetxt(path + '/Vector p0 Elemento (Locales) ' + format(i) + '.csv',
                       this.elementos[i].p0, delimiter=",")
            np.savetxt(path + '/Vector P0 Elemento (Globales) ' + format(i) + '.csv',
                       this.elementos[i].P0, delimiter=",")
            np.savetxt(path + '/Matriz Elemento ' + format(i) + '.csv', this.elementos[i].Ke, delimiter=",")
        temporal = np.array([])
        for i in range(0, this.elementos.size):
            temporal = np.append(temporal, [this.elementos[i].p])
        temporal = temporal.reshape(this.elementos.size, 6)
        np.savetxt(path + '/Vectores p Elementos.csv', temporal.T, delimiter=",")


def tridiag(a, b, c, n):
    """
Función auxiliar para crear una matriz tridiagonal (utilizada en FEM)
    :param a: Valor de la diagonal adyacente inferior
    :param b: Valor de la diagonal principal
    :param c: Valor de la diagonal adyacente superior
    :param n: Tamaño de la matriz nxn
    :return: La matriz tridiagonal creada a partir de los parámetros dados
    """
    va = np.zeros([1, n - 1])
    vb = np.zeros([1, n])
    vc = np.zeros([1, n - 1])
    va[0, :] = a
    vb[0, :] = b
    vc[0, :] = c
    return np.diag(va[0], -1) + np.diag(vb[0], 0) + np.diag(vc[0], 1)


