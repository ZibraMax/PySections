import numpy as np
from enum import Enum
from IPython.display import clear_output


class TipoSeccion(Enum):
    """Clase auxiliar para enumerar los tipos de secciones transversales (RECTANGULAR, CIRCULAR o GENERAL)
    """
    RECTANGULAR = 1
    CIRCULAR = 2
    GENERAL = 3


class Tipo(Enum):
    "Clase auxiliar para enumerar el tipo de elemento a utilizar (UNO, DOS, TRES o CUATRO)"
    UNO = 1
    DOS = 2
    TRES = 3
    CUATRO = 4


class Seccion:
    """Clase que representa la sección de los elementos a utilizar, asi como los materiales que la componen

        Args:
            nombre (str): Nombre definido para la sección a crear
            tipo (Tipo): Geometría de la sección transversal (Rectangular, Circular o General)
            propiedades (list): Dimensiones en metros relacionadas al tipo de geometría (Rectangular: base y altura, Circular: diámetro, General: área, inercia y área de cortante)
            material (Material): Tipo de material a utilizar
            qy (list, optional): Esto era algo de no lineal material pero no recuerdo. Defaults to None.
        """

    def __init__(self, nombre, tipo, propiedades, material, qy=None):
        """Clase que representa la sección de los elementos a utilizar, asi como los materiales que la componen

        Args:
            nombre (str): Nombre definido para la sección a crear
            tipo (Tipo): Geometría de la sección transversal (Rectangular, Circular o General)
            propiedades (list): Dimensiones en metros relacionadas al tipo de geometría (Rectangular: base y altura, Circular: diámetro, General: área, inercia y área de cortante)
            material (Material): Tipo de material a utilizar
            qy (list, optional): Esto era algo de no lineal material pero no recuerdo. Defaults to None.
        """
        self.qy = qy
        self.propiedadesGeometricas = propiedades
        self.material = material
        self.nombre = nombre
        if tipo == TipoSeccion.RECTANGULAR:
            self.area = propiedades[0] * propiedades[1]
            self.inercia = (propiedades[0] * propiedades[1] ** 3) / 12
            self.As = 5 / 6 * self.area
        if tipo == TipoSeccion.CIRCULAR:
            self.area = propiedades[0] ** 2 * np.pi / 4
            self.inercia = np.pi / 4 * (propiedades[0] / 2) ** 4
            self.As = 9 / 10 * self.area
        if tipo == TipoSeccion.GENERAL:
            if len(propiedades) >= 3:
                self.area = propiedades[0]
                self.inercia = propiedades[1]
                self.As = propiedades[2]
            else:
                self.area = propiedades[0]
                self.inercia = propiedades[1]
                self.As = 10 * 10 ** 10


class Resorte:
    """Clase que simula la rigidez de un resorte en un nodo de la estructura

        Args:
            nodo (Nodo): Nodo sobre el cual se quiere asignar el resorte
            rigidez (list):  Vector con las magnitudes de la rigideces asignadas al resorte en kiloNewtons / metros (Kx, Ky, Km)
            completo (bool): Sintetiza un resorte con 3 GDL si se especifica el parametro rigidez debe contener una matriz de numpy 3x3
        """

    def __init__(self, nodo, rigidez, completo):
        """Clase que simula la rigidez de un resorte en un nodo de la estructura

        Args:
            nodo (Nodo): Nodo sobre el cual se quiere asignar el resorte
            rigidez (list):  Vector con las magnitudes de la rigideces asignadas al resorte en kiloNewtons / metros (Kx, Ky, Km)
            completo (bool): Sintetiza un resorte con 3 GDL si se especifica el parametro rigidez debe contener una matriz de numpy 3x3
        """
        self.nodo = nodo
        self.completo = completo
        self.rigidez = rigidez
        if completo:
            self.Ke = self.rigidez
        else:
            self.Ke = np.array(
                [[self.rigidez[0], 0, 0], [0, self.rigidez[1], 0], [0, 0, self.rigidez[2]]])
        self.gdl = nodo.gdl

    def calcularKe(self, ngdl):
        """Función auxiliar para agregar los valores de rigidez del resorte a la matriz de rigidez de la estructura

        Args:
            ngdl (int): Número de grados de libertad de la estructura
        """
        self.Kee = np.zeros([ngdl, ngdl])
        for i in range(0, 3):
            for j in range(0, 3):
                self.Kee[self.gdl[i], self.gdl[j]] = self.Ke[i, j]


class Nodo:
    """Método de inicialización de nodos

        Args:
            pX (float): Coordenada x del nodo a crear
            pY (float):Coordenada y del nodo a crear
            pRestraints (list): Restricciones del nodo (x,y,m)
            pID (int): Identificador del nodo a crear (Número de nodo actual)
        """

    def __init__(self, pX, pY, pRestraints, pID):
        """Método de inicialización de nodos

        Args:
            pX (float): Coordenada x del nodo a crear
            pY (float):Coordenada y del nodo a crear
            pRestraints (list): Restricciones del nodo (x,y,m)
            pID (int): Identificador del nodo a crear (Número de nodo actual)
        """
        self.x = pX
        self.y = pY
        self.restraints = pRestraints
        self.gdl = np.array([-1, -1, -1])
        self.cargas = np.array([[0], [0], [0]])
        self.ID = pID

    def definirCargas(self, pX, pY, pM, remplazar):
        """Función para asignar cargas puntuales en los nodos creados

        Args:
            pX (float): Magnitud de carga puntual en x en kiloNewtons
            pY (float): Magnitud de carga puntual en y en kiloNewtons
            pM (float): Magnitud de momento puntual en kiloNewtons-metro
            remplazar (boll): Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        if remplazar:
            self.cargas = np.array([[pX], [pY], [pM]])
        else:
            self.cargas = self.cargas + np.array([[pX], [pY], [pM]])

    def calcularFn(self, ngdl):
        """Función para definir el vector de fuerzas nodales
        Args:
            ngdl (int): Número de grados de libertad de toda la estructura
        """
        self.Fn = np.zeros([ngdl, 1])
        for i in range(0, 3):
            self.Fn[self.gdl[i]] = self.cargas[i]


class Material:
    """Método de inicialización de los materiales

        Args:
            nombre (str): Nombre del material a crear
            E (float): Modulo de elasticidad del material a crear en kiloPascales
            v (float): Coeficiente de Poisson
            alfa (float): Coeficiente de expansión térmica en 1/°C
            gamma (float): Peso específico del material en kiloNewtons / metro cúbico
            sh (float, optional): Factor de reducción para no linealidad material. Defaults to 1.
        """

    def __init__(self, nombre, E, v, alfa, gamma, sh=1):
        """Método de inicialización de los materiales

        Args:
            nombre (str): Nombre del material a crear
            E (float): Modulo de elasticidad del material a crear en kiloPascales
            v (float): Coeficiente de Poisson
            alfa (float): Coeficiente de expansión térmica en 1/°C
            gamma (float): Peso específico del material en kiloNewtons / metro cúbico
            sh (float, optional): Factor de reducción para no linealidad material. Defaults to 1.
        """
        self.nombre = nombre
        self.E = E
        self.v = v
        self.alfa = alfa
        self.gamma = gamma
        self.sh = sh


class Elemento:
    "Clase para definir los elementos de la estructura y sus atributos"

    def __init__(self, pSeccion, pNodoI, pNodoF, pTipo, pApoyoIzquierdo, pApoyoDerecho, pZa, pZb, pDefCortante):
        """Método de inicialización de los elementos

        Args:
            pSeccion (Seccion): Sección del elemento (creada anteriormente)
            pNodoI (int): Indicador del nodo inicial (creado anteriormente)
            pNodoF (int): Indicador del nodo final (creado anteriormente)
            pTipo (Tipo): Tipo de elemento (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
            pApoyoIzquierdo (float): Longitud apoyo izquierdo para efectos de zonas rígidaz
            pApoyoDerecho (float): Longitud apoyo derecho para efectos de zonas rígidaz
            pZa (float): Factor de zonas rígidaz izquierdo
            pZb (float): Factor de zonas rigídas derecho
            pDefCortante (int): Parámetro binario para tener en cuenta deformaciones por cortante (1,0)
        """
        self.seccion = pSeccion
        self.Area = self.seccion.area
        self.Inercia = self.seccion.inercia
        self.E = self.seccion.material.E
        self.Tipo = pTipo
        self.nodoI = pNodoI
        self.nodoF = pNodoF
        self.cargasDistribuidas = np.array([[0, 0], [0, 0]])
        self.cargasPuntuales = np.array([])
        self.defCortante = pDefCortante
        self.factorZonasRigidas(pApoyoIzquierdo, pApoyoDerecho, pZa, pZb)
        if not (self.nodoF.x - self.nodoI.x == 0):
            self.Angulo = np.arctan(
                (self.nodoF.y - self.nodoI.y) / (self.nodoF.x - self.nodoI.x))
        else:
            if (self.nodoF.y > self.nodoI.y):
                self.Angulo = np.pi / 2
            else:
                self.Angulo = -np.pi / 2
        self.Longitud = ((self.nodoF.x - self.nodoI.x) ** 2 +
                         (self.nodoF.y - self.nodoI.y) ** 2) ** (1 / 2)
        la = self.za * self.apoyoIzquierdo
        lb = self.zb * self.apoyoDerecho
        self.Longitud = self.Longitud - la - lb
        self.crearDiccionario()
        self.crearLambda()
        self.wx = 0
        self.wy = 0
        self.f = 0
        self.p0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
        self.P0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
        if self.Tipo == Tipo.UNO:
            self.alfa = 12
            self.beta = 4
            self.gamma = 6
            self.fi = 1
            self.fidiag = 1
        elif self.Tipo == Tipo.DOS:
            self.alfa = 3
            self.beta = 3
            self.gamma = 3
            self.fi = 0
            self.fidiag = 1
        elif self.Tipo == Tipo.TRES:
            self.alfa = 3
            self.beta = 3
            self.gamma = 3
            self.fi = 1
            self.fidiag = 0
        else:
            self.alfa = 0
            self.beta = 0
            self.gamma = 0
            self.fi = 0
            self.fidiag = 0
        self.calcularMatrizElemento()

    def hallarKb(self, psi=0):
        """Función para hallar la matriz de rigidez básica del elemento

        Args:
            psi (int, optional): Factor psi, utilizado para realizar las aproximaciones lineales. Defaults to 0.

        Returns:
            np.ndarray: la matriz de rigidez básica
        """
        E = self.E
        I = self.Inercia
        A = self.Area
        L = self.Longitud
        if psi < 0.001:
            kb1 = 4*E*I/L
            kb2 = 2*E*I/L
            kb3 = 3*E*I/L
        else:
            kb1 = ((E*I)/(L))*((psi*(np.sin(psi)-psi*np.cos(psi))) /
                               (2-2*np.cos(psi)-psi*np.sin(psi)))
            kb2 = (E*I*(psi*(psi-np.sin(psi)))) / \
                (L*(2-2*np.cos(psi)-psi*np.sin(psi)))
            kb3 = (E*I*(psi**2*np.sin(psi)))/(L*(np.sin(psi)-psi*np.cos(psi)))
        if self.Tipo == Tipo.UNO:
            kb = np.array([[E*A/L, 0, 0], [0, kb1, kb2], [0, kb2, kb1]])
        elif self.Tipo == Tipo.DOS:
            kb = np.array([[E*A/L, 0, 0], [0, 0, 0], [0, 0, kb3]])
        elif self.Tipo == Tipo.TRES:
            kb = np.array([[E*A/L, 0, 0], [0, kb3, 0], [0, 0, 0]])
        else:
            kb = np.array([[E*A/L, 0, 0], [0, 0, 0], [0, 0, 0]])
        return kb

    def determinarV0(self):
        """Función que permite hallar los desplazamientos básicos iniciales del elemento
        """
        self.kb0 = self.hallarKb()
        # Matriz singular
        self.q0 = self.p0[np.ix_([3, 2, 5]), 0].T
        self.v0 = np.dot(np.linalg.pinv(self.kb0), self.q0)

    def calcularv(self):
        """Función que permite hallar los desplazamientos básicos del elemento deformado
        """
        self.deltax0 = self.Longitud*np.cos(self.Angulo)
        self.deltay0 = self.Longitud*np.sin(self.Angulo)
        self.deltaux = self.Ue[3]-self.Ue[0]
        self.deltauy = self.Ue[4]-self.Ue[1]
        self.deltax = self.deltaux+self.deltax0
        self.deltay = self.deltauy+self.deltay0
        self.L = np.sqrt(self.deltax**2+self.deltay**2)
        self.theta = np.arcsin(self.deltay/self.L)
        self.deltatheta = self.theta-self.Angulo
        if self.Tipo == Tipo.UNO:
            v1 = self.L-self.Longitud
            v2 = self.Ue[2]-self.deltatheta
            v3 = self.Ue[5]-self.deltatheta
        elif self.Tipo == Tipo.DOS:
            v1 = self.L-self.Longitud
            v3 = self.Ue[5]-self.deltatheta
            v2 = -v3/2+self.v0[2][0]/2+self.v0[1][0]
        elif self.Tipo == Tipo.TRES:
            v1 = self.L-self.Longitud
            v2 = self.Ue[2]-self.deltatheta
            v3 = -v2/2+self.v0[1][0]/2+self.v0[2][0]
        else:
            v1 = self.L-self.Longitud
            v2 = 0
            v3 = 0
        self.v = np.array([[v1], [v2], [v3]])

    def calcularMatrizElemento(self):
        """Función para generar la matriz de rigidez del elemento
        """
        E = self.E
        v = self.seccion.material.v
        As = self.seccion.As
        A = self.Area
        I = self.Inercia
        t = self.Angulo

        L = self.Longitud

        alfa = self.alfa
        beta = self.beta
        gamma = self.gamma
        fi = self.fi
        fidiag = self.fidiag
        self.G = E / (2 + 2 * v)
        mu = (12 * E * I) / (self.G * As * L ** 2) * self.defCortante

        c = np.cos(0)
        s = np.sin(0)
        k1 = (E * A / L) * (c ** 2) + alfa * E * \
            I / ((1 + mu) * L ** 3) * s ** 2
        k2 = E * A / (L) * (s ** 2) + alfa * E * I / \
            ((1 + mu) * L ** 3) * (c ** 2)
        k3 = (beta + mu) * E * I / ((1 + mu) * L)
        k4 = (E * A / L - alfa * E * I / ((1 + mu) * L ** 3)) * s * c
        k5 = gamma * E * I / ((1 + mu) * L ** 2) * c
        k6 = gamma * E * I / ((1 + mu) * L ** 2) * s
        k7 = (beta / 2 - mu) * E * I / ((1 + mu) * L)
        self.Ke = np.array([[k1, k4, -k6 * fi, -k1, -k4, -k6 * fidiag], [k4, k2, k5 * fi, -k4, -k2, k5 * fidiag],
                            [-k6 * fi, k5 * fi, k3 * fi, k6 *
                                fi, -k5 * fi, fi * k7 * fidiag],
                            [-k1, -k4, k6 * fi, k1, k4, k6 * fidiag], [-k4, -
                                                                       k2, -k5 * fi, k4, k2, -k5 * fidiag],
                            [-k6 * fidiag, k5 * fidiag, fidiag * k7 * fi, k6 * fidiag, -k5 * fidiag, k3 * fidiag]])
        self.lbdaz = np.array([[1, 0, 0, 0, 0, 0], [0, 1, self.za, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, -self.zb], [0, 0, 0, 0, 0, 1]])
        self.Ke = np.dot(np.dot(np.dot(self.lbdaz.T, self.lbda.T),
                         self.Ke), np.dot(self.lbdaz, self.lbda))

    def crearDiccionario(self):
        """
    Función que genera la nomenclatura de grados de libertad correpondientes al elemento
        """
        self.diccionario = np.array(
            [self.nodoI.gdl[0], self.nodoI.gdl[1], self.nodoI.gdl[2], self.nodoF.gdl[0], self.nodoF.gdl[1],
             self.nodoF.gdl[2]])

    def matrizSuma(self, ngdl):
        """
    Función que simula la matriz de rigidez de la estructura, únicamente con el valor de las rigideces del elemento actual
        :param ngdl: Número de grados de libertad total de la estructura
        """
        self.kee = np.zeros([ngdl, ngdl])
        for i in range(0, 6):
            for j in range(0, 6):
                self.kee[self.diccionario[i],
                         self.diccionario[j]] = self.Ke[i, j]

    def crearLambda(self):
        """
    Función que crea la matriz de transformación de coordenadas globales a locales Lambda del elemento
        """
        t = self.Angulo
        c = np.cos(t)
        s = np.sin(t)
        self.lbda = np.zeros([6, 6])
        self.lbda[0, 0] = c
        self.lbda[0, 1] = s
        self.lbda[1, 0] = -s
        self.lbda[1, 1] = c
        self.lbda[2, 2] = 1
        self.lbda[3, 3] = c
        self.lbda[3, 4] = s
        self.lbda[4, 3] = -s
        self.lbda[4, 4] = c
        self.lbda[5, 5] = 1

    def fuerzasBasicas(self):
        """Función asociada a ls matriz de rigidez básica y las matrices de transformación geométrica (lambda y T)
        """
        Re, v, q, kb, ve, vp = estadoPlasticidadConcentrada(
            self.v, self.seccion.material.sh, self.seccion.qy, self.E*self.Inercia, self.Longitud, self.E*self.Area, tipo=self.Tipo, v0=self.v0)
        self.kb = kb
        self.ve = ve
        self.vp = vp
        self.q = q
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        l = self.L
        self.T = np.array([[-c, -s/l, -s/l], [-s, c/l, c/l],
                          [0, 1, 0], [c, s/l, s/l], [s, -c/l, -c/l], [0, 0, 1]]).T

        self.lbd = np.zeros([6, 6])
        self.lbd[0, 0] = c
        self.lbd[0, 1] = s
        self.lbd[1, 0] = -s
        self.lbd[1, 1] = c
        self.lbd[2, 2] = 1
        self.lbd[3, 3] = c
        self.lbd[3, 4] = s
        self.lbd[4, 3] = -s
        self.lbd[4, 4] = c
        self.lbd[5, 5] = 1

    def fuerzasBasicasEL(self):
        q1 = self.E*self.Area/self.Longitud*(self.v[0][0]-self.v0[0][0])
        q1 = np.min([q1, -1*10**-5])

        self.psi = np.sqrt(-q1*self.Longitud**2/self.E/self.Inercia)
        self.kb = self.hallarKb(self.psi)
        self.q = self.kb @ (self.v-self.v0)
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        l = self.L
        self.T = np.array([[-c, -s/l, -s/l], [-s, c/l, c/l],
                          [0, 1, 0], [c, s/l, s/l], [s, -c/l, -c/l], [0, 0, 1]]).T

        self.lbd = np.zeros([6, 6])
        self.lbd[0, 0] = c
        self.lbd[0, 1] = s
        self.lbd[1, 0] = -s
        self.lbd[1, 1] = c
        self.lbd[2, 2] = 1
        self.lbd[3, 3] = c
        self.lbd[3, 4] = s
        self.lbd[4, 3] = -s
        self.lbd[4, 4] = c
        self.lbd[5, 5] = 1

    def matrizYFuerzas(self):
        """Función asociada al cálculo de matriz de rigidez y fuerzas internas de los elementos con efectos de carga axial
        :return: Ke y p, matriz de rigidez y vector de fuerzas respectivamente
        """
        matrizMaterial = np.dot(self.T.T, np.dot(self.kb, self.T))
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        l = self.L
        A = np.array([[1-c**2, -s*c, 0, c**2-1, s*c, 0], [-s*c, 1-s**2, 0, s*c, s**2-1, 0], [0, 0, 0, 0, 0, 0],
                     [c**2-1, s*c, 0, 1-c**2, -s*c, 0], [s*c, s**2-1, 0, -s*c, 1-s**2, 0], [0, 0, 0, 0, 0, 0]])
        B = np.array([[-2*s*c, c**2-s**2, 0, 2*s*c, s**2-c**2, 0], [c**2-s**2, 2*s*c, 0, s**2-c**2, -2*s*c, 0], [0, 0, 0, 0, 0, 0],
                     [2*s*c, s**2-c**2, 0, -2*c*s, c**2-s**2, 0], [s**2-c**2, -2*c*s, 0, c**2-s**2, 2*s*c, 0], [0, 0, 0, 0, 0, 0]])
        parteAxial = A*self.q[0][0]/l+B*(self.q[2][0]+self.q[1][0])/(l**2)
        self.matrizMaterial = matrizMaterial
        self.matrizGlobal = parteAxial
        self.Ke1 = parteAxial + matrizMaterial
        # Pensar en un malparido metodo numerico
        self.p1 = np.dot(self.T.T, self.q)+np.dot(self.lbd, self.p0)
        return self.Ke1, self.p1

    def calcularVectorDeFuerzas(self):
        """Función que calcula el vector de fuerzas del elemento en coordenadas globales
        """
        self.P0 = np.dot(np.dot(self.lbda.T, self.lbdaz.T), self.p0)

    def definirCargas(self, pWx, pWy, pF, remplazar):
        """Función que define las cargas (distribuidas en kiloNewtons / metros o puntuales en Newtons cada L/3) aplicadas sobre el elemento
        :param pWx: Carga distribuida en la dirección +x en kiloNewtons / metros
        :param pWy: Carga distribuida en la dirección -y en kiloNewtons / metros
        :param pF: Carga aplicada cada L/3 en la dirección -y en kiloNewtons
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.wx = pWx
        self.wy = pWy
        self.f = pF
        wx = self.wx
        wy = self.wy
        l = self.Longitud
        f = self.f
        if self.Tipo == Tipo.UNO:
            p0 = np.array(
                [[-wx * l / 2], [wy * l / 2], [wy * l ** 2 / 12], [-wx * l / 2], [wy * l / 2], [-wy * l ** 2 / 12]])
            p0 = p0 + np.array([[0], [f], [2 * f * l / 9],
                               [0], [f], [-2 * f * l / 9]])
        elif self.Tipo == Tipo.DOS:
            p0 = np.array([[-wx * l / 2], [3 * wy * l / 8], [0],
                          [-wx * l / 2], [5 * wy * l / 8], [-wy * l ** 2 / 8]])
            p0 = p0 + np.array([[0], [2 * f / 3], [0], [0],
                               [4 * f / 3], [-f * l / 3]])
        elif self.Tipo == Tipo.TRES:
            p0 = np.array([[-wx * l / 2], [5 * wy * l / 8], [wy *
                          l ** 2 / 8], [-wx * l / 2], [3 * wy * l / 8], [0]])
            p0 = p0 + \
                np.array([[0], [4 * f / 3], [f * l / 3], [0], [2 * f / 3], [0]])
        elif self.Tipo == Tipo.CUATRO:
            p0 = np.array([[-wx * l / 2], [wy * l / 2], [0],
                          [-wx * l / 2], [wy * l / 2], [0]])
            p0 = p0 + np.array([[0], [f], [0], [0], [f], [0]])
        if remplazar:
            self.p0 = p0
            if not pWy == 0:
                self.cargasDistribuidas = np.append(
                    [], np.array([[0, -pWy], [self.Longitud, -pWy]]))
            if not pF == 0:
                self.cargasPuntuales = np.append(
                    [], np.array([self.Longitud / 3, -pF]))
                self.cargasPuntuales = np.append(
                    [], np.array([2 * self.Longitud / 3, -pF]))
        else:
            self.p0 = self.p0 + p0
            if not pWy == 0:
                self.cargasDistribuidas = np.append(self.cargasDistribuidas,
                                                    np.array([[0, -pWy], [2, 1]]).reshape([2, 2]))
            if not pF == 0:
                self.cargasPuntuales = np.append(
                    self.cargasPuntuales, np.array([self.Longitud / 3, -pF]))
                self.cargasPuntuales = np.append(
                    self.cargasPuntuales, np.array([2 * self.Longitud / 3, -pF]))
        self.calcularVectorDeFuerzas()

    def factorZonasRigidas(self, apoyoIzquierdo, apoyoDerecho, za, zb):
        """
        :param apoyoIzquierdo:
        :param apoyoDerecho:
        :param za:
        :param zb:
        """
        self.za = za
        self.zb = zb
        self.apoyoIzquierdo = apoyoIzquierdo
        self.apoyoDerecho = apoyoDerecho

    def agregarCargaPuntual(self, f, x, remplazar):
        """
    Función que agrega una sola carga puntual a una distancia x del elemento (agregar una a una)
        :param f: Magnitud de la fuerza en cuestión en kiloPascales
        :param x: Distancia de la fuerza en cuestión desde el nodo inicial hasta el nodo final en metros
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        l = self.Longitud

        if self.Tipo == Tipo.UNO:
            p0 = np.array([[0], [f - f * x ** 2 * (3 * l - 2 * x) / l ** 3], [f * x * (l - x) ** 2 / l ** 2], [0],
                           [f * x ** 2 * (3 * l - 2 * x) / l ** 3], [-f * x ** 2 * (l - x) / l ** 2]])
        elif self.Tipo == Tipo.DOS:
            p0 = np.array([[0], [f * (l - x) ** 2 * (2 * l + x) / (2 * l ** 3)], [0], [0],
                           [f * x * (3 * l ** 2 - x ** 2) / (2 * l ** 3)], [-f * x * (l ** 2 - x ** 2) / (2 * l ** 2)]])
        elif self.Tipo == Tipo.TRES:
            p0 = np.array([[0], [f * (l - x) * (2 * l ** 2 + 2 * l * x - x ** 2) / (2 * l ** 3)],
                           [f * x * (l - x) * (2 * l - x) / (2 * l ** 2)], [0],
                           [f * x ** 2 * (3 * l - x) / (2 * l ** 3)], [0]])
        elif self.Tipo == Tipo.CUATRO:
            p0 = np.array([[0], [f * (l - x) / l], [0], [0], [f * x / l], [0]])
        if remplazar:
            self.p0 = p0
        else:
            self.p0 = self.p0 + p0
        self.calcularVectorDeFuerzas()

    def agregarDefectoDeFabricacion(self, e0, fi0, remplazar):
        """Función que simula los defectos de fabricación del elemento y los tiene en cuenta para el vector de fuerzas
        :param e0: Elongación del elemento
        :param fi0: Curvatura del elemetno (1 / metros)
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        L = self.Longitud
        E = self.seccion.material.E
        A = self.seccion.area
        I = self.seccion.inercia
        if self.Tipo == Tipo.UNO:
            p0 = np.array([[E * A * e0], [0], [E * I * fi0],
                          [-E * A * e0], [0], [-E * I * fi0]])
        elif self.Tipo == Tipo.DOS:
            p0 = np.array([[E * A * e0], [-3 * E * I * fi0 / (2 * L)], [0], [-E * A * e0], [3 * E * I * fi0 / (2 * L)],
                           [-3 * E * I * fi0 / 2]])
        elif self.Tipo == Tipo.TRES:
            p0 = np.array([[E * A * e0], [3 * E * I * fi0 / (2 * L)], [3 * E * I * fi0 / 2], [-E * A * e0],
                           [-3 * E * I * fi0 / (2 * L)], [0]])
        elif self.Tipo == Tipo.CUATRO:
            p0 = np.array([[E * A * e0], [0], [0], [-E * A * e0], [0], [0]])
        if remplazar:
            self.p0 = p0
        else:
            self.p0 = self.p0 + p0
        self.calcularVectorDeFuerzas()

    def agregarCargaPorTemperatura(self, pDeltaT0, pDeltaTFh, remplazar):
        """Función que simula los efectos de la temperatura en el elemento y los tiene en cuenta para el vector de fuerzas
        :param pDeltaT0: Variación de la temperatura en el elemento (°C)
        :param pDeltaTFh: Relación entre el gradiente de temperatura en el elemento y la altura de este (°C / metros)
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.deltaT0 = pDeltaT0
        self.deltaTFh = pDeltaTFh
        dt0 = self.deltaT0
        dtf = self.deltaTFh
        e0 = self.seccion.material.alfa * dt0
        fi0 = self.seccion.material.alfa * pDeltaTFh
        self.agregarDefectoDeFabricacion(e0, fi0, remplazar)

    def agregarCargaPresfuerzoAxial(self, q0, remplazar):
        """Función que simula los efectos de una carga de presfuerzo axial sobre el elemento
        :param q0: Fuerza de presfuerzo en kiloNewtons
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        p0 = np.array([[-q0], [0], [0], [q0], [0], [0]])
        if remplazar:
            self.p0 = p0
        else:
            self.p0 = self.p0 + p0
        self.calcularVectorDeFuerzas()

    def agregarCargaPostensadoFlexionYAxial(self, f0, e1, e2, e3, remplazar):
        """Función que simula los efectos de una carga de postensado a flexión y axial sobre el elemento.
        :param f0: fuerza de presfuerzo aplicada (puede incluir pérdidas) en kiloNewtons
        :param e1: Excentricidad del cable al comienzo del elemento en metros
        :param e2: Excentricidad del cable a la distancia donde ocurre el mínimo del elemento en metros
        :param e3: Excentricidad del cable al final del elemento en metros
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        L = self.Longitud
        c = e1
        e2 = -e2
        b = -2 / L * (np.sqrt(e1 - e2)) * (np.sqrt(e1 - e2) + np.sqrt(e3 - e2))
        a = (e3 - e1 - b * L) / (L ** 2)
        def dedx(x): return 2 * a * x + b
        f0x = np.cos(np.arctan(dedx(0))) * f0
        if self.Tipo == Tipo.UNO:
            p0 = np.array([[-f0x], [-(a * L + b) * f0], [-(a * L ** 2 - 6 * c) * f0 / 6], [f0x], [(a * L + b) * f0],
                           [-(5 * a * L ** 2 + 6 * b * L + 6 * c) * f0 / 6]])
        elif self.Tipo == Tipo.DOS:
            p0 = np.array([[-f0x], [-(3 * a * L ** 2 + 4 * b * L + 6 * L) * f0 / (4 * L)], [0], [f0x],
                           [(3 * a * L ** 2 + 4 * b * L + 6 * c) * f0 / (4 * L)],
                           [-(3 * a * L ** 2 + 4 * b * L + 6 * c) * f0 / 4]])
        elif self.Tipo == Tipo.TRES:
            p0 = np.array(
                [[-f0x], [(a * L ** 2 + 2 * b * L + 6 * c) * f0 / (4 * L)], [(a * L ** 2 + 2 * b * L + 6 * c) * f0 / 4],
                 [f0x], [-(a * L ** 2 + 2 * b * L + 6 * c) * f0 / (4 * L)], [0]])
        elif self.Tipo == Tipo.CUATRO:
            p0 = np.array([[-f0x], [0], [0], [f0x], [0], [0]])
        if remplazar:
            self.p0 = p0
        else:
            self.p0 = self.p0 + p0
        self.calcularVectorDeFuerzas()

    def calcularF0(self, ngdl):
        """
    Función para calcular el vector de fuerzas generadas por cargas externas por elemento (teniendo en cuenta todos los gdl de la estructura)
        :param ngdl: Número de grados de libertad de la estructura
        """
        self.F0 = np.zeros([ngdl, 1])
        for i in range(0, 6):
            self.F0[self.diccionario[i], 0] = self.P0[i, 0]

    def calcularVectorDeFuerzasInternas(self, U):
        """
    Función para calcular el vector de fuerzas internas del elemento
        :param U: Vector de desplazamientos calculados para todos los grados de libertad de la estructura
        """
        self.Ue = U[self.diccionario]
        parcial = np.dot(self.Ke, self.Ue)
        self.P = np.reshape(parcial, [parcial.size, 1]) + self.P0
        self.p = np.dot(self.lbda, self.P)


class Constraint:
    "Clase que define los constraints presentes en una estructura"

    def __init__(self, tipo, nodoI, nodoF):
        """Métoddo que inicializa los constraints del elemento
        :param tipo: Tipo de elemento (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
        :param nodoI: Nodo inicial (objeto)
        :param nodoF: Nodo final (objeto)
        """
        self.nodoI = nodoI
        self.nodoF = nodoF
        self.tipo = tipo
        if not (nodoF.x - nodoI.x == 0):
            self.Angulo = np.arctan((nodoF.y - nodoI.y) / (nodoF.x - nodoI.x))
        else:
            if (nodoF.y > nodoI.y):
                self.Angulo = np.pi / 2
            else:
                self.Angulo = -np.pi / 2
        self.r = np.zeros([3, 6])
        t = self.Angulo
        c = np.cos(t)
        s = np.sin(t)
        self.independientes = [
            self.nodoI.gdl[0], self.nodoI.gdl[1], self.nodoI.gdl[2], self.nodoF.gdl[2]]
        self.dependientes = [self.nodoF.gdl[0], self.nodoF.gdl[1]]


class SuperElemento:
    "Clase que define los superelementos presentes en una estructura"

    def __init__(self, SE, SF, gdl):
        """Método de inicialización de los elementos
        :param SE:
        :param SF:
        :param gdl: Grados de libertad de la estructura
        """
        self.SE = SE
        self.SF = SF
        self.gdl = gdl

    def calcularKSUMA(self, n):
        """Función para calcular la matriz de rigidez condensada de la estructura
        :param n:
        :return:
        """
        a = np.zeros([n, n])
        a[np.ix_(self.gdl, self.gdl)] = self.SE
        return a

    def calcularF0(self, n):
        """Función para calcular el vector de fuerzas condensado de la estructura
        :param n:
        :return:
        """
        a = np.zeros([n, 1])
        a[np.ix_(self.gdl)] = self.SF
        return a


class Estructura:
    "Clase que representa una estructura."

    def __init__(self):
        """Método de inicialización de la estructura
        """
        self.nodos = np.array([])
        self.resortes = np.array([])
        self.elementos = np.array([])
        self.Ur = np.array([])
        self.constraints = np.array([])
        self.superelementos = np.array([])

    def agregarNodo(self, x, y, fix=[True, True, True]):
        """Función que agrega un nodo a la estructura
        :param x: Posición x del nodo en metros
        :param y: Posición y del nodo en metros
        :param fix: Condiciones de apoyo del nodo (True = restringido, False = libre)
        """
        self.nodos = np.append(self.nodos, Nodo(x, y, fix, self.nodos.size))
        self.actualizarGDL()
        self.actualizarElementos()
        self.actualizarResortes()
        self.Ur = np.zeros([self.restringidos.size, 1])

    def newton(self, param, semilla=None, control='carga'):
        """Función que realiza el método iterativo de Newton
        :param param: lista de parametrso del metodo de Newton TODO especificar los parametros en la documentacion
        :param semilla: Semilla inicial del metodo de newton ndarray
        :param control: Tipo de algotimo de solucion, puede ser carga o desplazamiento
        """
        self.calcularFn()
        try:
            if semilla == None:
                Ul = np.zeros([self.libres.size])
            else:
                Ul = semilla
        except:
            Ul = semilla
        # Alerta de que no se usaran restringidos np.any() creo
        Ur = np.zeros([self.restringidos.size])
        Fn = self.Fn[np.ix_(self.libres)]
        self.RECORDU = []
        self.RECORDF = []
        if control == 'carga':
            # Inicializacion
            e1 = param[0]
            e2 = param[1]
            e3 = param[2]
            dli = param[3]
            Nd = param[4]
            Nj = param[5]
            gamma = param[6]
            dlmax = param[7]
            dlmin = param[8]
            li = 0
            incremento = param[9]
            gdl = param[11]
            for i in range(0, Nd):

                # Aasigan los desplazamientos
                for e in self.elementos:
                    U = np.append(Ul, Ur)
                    e.Ue = U[np.ix_(e.diccionario)]
                Kll, P = self.determinacionDeEstado()
                if i == 0:
                    Kll0 = np.copy(Kll)
                if i > 0:
                    if incremento == 'constante':
                        dli = param[3]
                        dlmin = param[3]
                        dlmax = param[3]
                    elif incremento == 'bergan':
                        num = np.dot(Fn.T, np.dot(np.linalg.pinv(Kll0), Fn))
                        den = np.dot(Fn.T, np.dot(np.linalg.pinv(Kll), Fn))
                        dli = param[3]*(np.dot(num, 1/den))**gamma
                    elif incremento == 'numiter':
                        dli = param[3]*((i)/Nd)**gamma
                    else:
                        dli = param[3]
                        dlmin = param[3]
                        dlmax = param[3]
                dli = np.max(
                    [np.max(np.array([dlmin, np.min(np.array([dlmax, dli]))]))*param[10], dli])

                li = li + dli
                F = li*Fn
                # Pensar en un while mejor!
                for j in range(1, Nj):
                    R = F-P
                    if np.linalg.norm(R) > e1:
                        DUl = np.dot(np.linalg.pinv(Kll), R)
                        if np.linalg.norm(DUl) > e2:
                            if np.linalg.norm(np.dot(DUl.T, R))*0.5 > e3:
                                Ul = Ul + DUl.T
                                for e in self.elementos:
                                    U = np.append(Ul, Ur)
                                    e.Ue = U[np.ix_(e.diccionario)]
                                Kll, P = self.determinacionDeEstado()
                self.RECORDU = np.append(self.RECORDU, U[np.ix_(gdl)])
                self.RECORDF = np.append(self.RECORDF, P[np.ix_(gdl)])
            return Ul.T
        elif control == 'desplazamiento':
            print('TODO')
            for i in range(0, param[0]):
                for i in self.elementos:
                    U = np.append(Ul, Ur)
                    i.Ue = U[np.ix_(i.diccionario)]
                Kll, P = self.determinacionDeEstado()
                A = np.dot(np.linalg.pinv(Kll), (Fn-P))
                Ul = Ul + A.T
            return Ul.T

    def determinacionDeEstado(self):
        """Función para realizar determinación de estado de la estructura.
        :return: Kll y Pl, matriz de rigidez y vector de fuerzas de la estructura
        """
        n = self.libres.size+self.restringidos.size
        Kll = np.zeros([n, n])
        Pl = np.zeros([n, 1])
        for i in self.elementos:
            i.determinarV0()
            i.calcularv()
            if i.seccion.qy == None:
                i.fuerzasBasicasEL()
            else:
                i.fuerzasBasicas()
            [Ke, P] = i.matrizYFuerzas()
            Kll[np.ix_(i.diccionario, i.diccionario)] = Kll[np.ix_(
                i.diccionario, i.diccionario)] + Ke
            Pl[np.ix_(i.diccionario)] = Pl[np.ix_(i.diccionario)] + P
        return Kll[np.ix_(self.libres, self.libres)], Pl[np.ix_(self.libres)]

    def actualizarElementos(self):
        """Función que actualiza constantemente los atributos de los elementos en caso de existir una modificación
        """
        for i in range(0, self.elementos.size):
            parcial = self.elementos[i]
            self.elementos[i] = Elemento(self.elementos[i].seccion, self.nodos[self.elementos[i].nodoI.ID],
                                         self.nodos[self.elementos[i].nodoF.ID], self.elementos[i].Tipo,
                                         self.elementos[i].apoyoIzquierdo, self.elementos[i].apoyoDerecho,
                                         self.elementos[i].za, self.elementos[i].zb, self.elementos[i].defCortante)
            self.elementos[i].p0 = parcial.p0
            self.elementos[i].calcularVectorDeFuerzas()

    def actualizarResortes(self):
        """Función que actualiza constantemente los atributos de los resortes en caso de existir una modificación
        """
        for i in range(0, self.resortes.size):
            self.resortes[i] = Resorte(self.nodos[self.resortes[i].nodo.ID], self.resortes[i].rigidez,
                                       self.resortes[i].completo)

    def agregarElemento(self, seccion, nodoInicial, nodoFinal, tipo=Tipo.UNO, apoyoIzquierdo=0, za=0, apoyoDerecho=0, zb=0,
                        defCortante=True):
        """Función que permite añadir un nuevo elemento a la estructura
        :param seccion: Sección del elemento a añadir
        :param nodoInicial: Identificador del nodo inicial del elemento
        :param nodoFinal: Identificador del nodo final del elemento
        :param tipo: Tipo de elemento a crear (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
        :param apoyoIzquierdo:
        :param za:
        :param apoyoDerecho:
        :param zb:
        :param defCortante:
        """
        self.elementos = np.append(self.elementos,
                                   Elemento(seccion, self.nodos[nodoInicial], self.nodos[nodoFinal], tipo,
                                            apoyoIzquierdo, apoyoDerecho, za, zb, defCortante))

    def agregarResorte(self, rigidez, nodo=-1, completo=False):
        """Función que permite añadir un nuevo resorte a la estructura
        :param rigidez: Vector con las magnitudes de las rigideces del resorte en kiloPascales (Kx,Ky,Km)
        :param nodo: Nodo sobre el que se va a agregar el resorte
        :param completo:
        """
        self.resortes = np.append(self.resortes, Resorte(
            self.nodos[nodo], rigidez, completo))

    def agregarSuperElementos(self, SE, SF, gdl):
        """Función que permite realizar un superelemento con la estructura
        :param SE:
        :param SF:
        :param gdl: Número de grados de libertad de la estructura
        """
        supelemento = SuperElemento(SE, SF, gdl)
        self.superelementos = np.append(self.superelementos, supelemento)

    def definirConstraint(self, tipo, nodoInicial, nodoFinal):
        """Función que permite definir un constraint en la estructura
        :param tipo: Tipo de elemento (Tipo.UNO, Tipo.DOS, Tipo.TRES, Tipo.CUATRO)
        :param nodoInicial: Nodo inicial (objeto)
        :param nodoFinal: Nodo final (objeto)
        """
        nodoF = self.nodos[nodoFinal]
        nodoI = self.nodos[nodoInicial]
        constraint = Constraint(tipo, nodoI, nodoF)
        self.constraints = np.append(self.constraints, constraint)

    def crearMatrizDeRigidez(self):
        """Función que permite calcular la matriz de rigidez de la estructura
        """
        n = self.nodos.size * 3
        self.KE = np.zeros([n, n])
        for i in range(0, self.elementos.size):
            self.elementos[i].matrizSuma(n)
            self.KE = self.KE + self.elementos[i].kee
        for i in range(0, self.resortes.size):
            self.resortes[i].calcularKe(n)
            self.KE = self.KE + self.resortes[i].Kee
        for i in range(0, self.superelementos.size):
            self.KE = self.KE + self.superelementos[i].calcularKSUMA(n)

    def definirCambiosTemperatura(self, inicial, interior, exterior, h, elemento=-1, remplazar=False):
        """Función que permite calcular las fuerzas asociadas a los cambios de temperatura en el elemento (CASO 1)
        :param inicial: Temperatura ambiente a la que fue construida la estructura en °C
        :param interior: Temperatura interna de la estructura en °C
        :param exterior: Temperatura externa de la estructura en °C
        :param h: Altura de la sección transversal del elemento en metros
        :param elemento: Elemento sobre el cual se quieren definir los cambios de temperatura
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        dta = 1 / 2 * ((exterior - inicial) + (interior - inicial))
        dtf = (interior - exterior)
        self.elementos[elemento].agregarCargaPorTemperatura(
            dta, dtf / h, remplazar)

    def agregarCargaPorTemperatura(self, deltaT0, deltaThf, elemento=-1, remplazar=False):
        """Función que permite calcular las fuerzas asociadas a los cambios de temperatura en el elemento (CASO 2)
        :param deltaT0: Variación de la temperatura en °C
        :param deltaThf: Gradiente de temperatura (°C / metros)
        :param elemento: Elemento sobre el cual se quieren definir los cambios de temperatura
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].agregarCargaPorTemperatura(
            deltaT0, deltaThf, remplazar)

    def agregarDefectoDeFabricacion(self, e0=0, fi0=0, elemento=-1, remplazar=False):
        """Función que permite calcular las fuerzas asociadas a los defectos de fabricación en el elemento
        :param e0: Elongación del elemento
        :param fi0: Curvatura del elemetno (1 / metros)
        :param elemento: Elemento sobre el cual se quieren definir los defectos de fabricación
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].agregarDefectoDeFabricacion(
            e0, fi0, remplazar)

    def agregarCargaPresfuerzoAxial(self, el, q0, elemento=-1, remplazar=False):
        """Función que permite calcular las fuerzas asociadas a una carga de presfuerzo axial en el elemento
        :param el:
        :param q0: Fuerza de presfuerzo en kiloNewtons
        :param elemento: Elemento sobre el cual se quieren definir las cargas de presfuerzo
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].agregarCargaPresfuerzoAxial(q0, remplazar)

    def agregarCargaPostensadoFlexionYAxial(self, f0, e1, e2, e3, elemento=-1, remplazar=False):
        """Función que permite calcular las fuerzas asociadas a una carga de postensado a flexión y axial en el elemento
        :param f0: fuerza de presfuerzo aplicada (puede incluir pérdidas) en kiloNewtons
        :param e1: Excentricidad del cable al comienzo del elemento en metros
        :param e2: Excentricidad del cable a la distancia donde ocurre el mínimo del elemento en metros
        :param e3: Excentricidad del cable al final del elemento en metros
        :param elemento: Elemento sobre el cual se quieren definir las cargas de postensado
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].agregarCargaPostensadoFlexionYAxial(
            f0, e1, e2, e3, remplazar)

    def agregarCargaElemento(self, wx=0, wy=0, ftercios=0, elemento=-1, remplazar=False):
        """
    Función para agregar cargas distribuidas o a tercios del elemento
        :param wx: Magnitud de la carga distribuida en dirección x en kiloNewtons/metros
        :param wy: Magnitud de la carga distribuida en dirección y en kiloNewtons/metros
        :param ftercios: Magnitud de las cargas aplicados a tercios de la longitud del elemento en kiloNewtons
        :param elemento: Identificador del elemento sobre el cual se aplica/n la/s carga/s
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].definirCargas(wx, wy, ftercios, remplazar)

    def agregarCargaNodo(self, nodo=-1, px=0, py=0, m=0, remplazar=False):
        """
    Función que permite agregar cargas a los nodos de la estructura
        :param nodo: Identificador del nodo sobre el cual se va a agregar la carga puntual
        :param px: Magnitud de la carga en x en kiloNewtons
        :param py: Magnitud de la carga en y en kiloNewtons
        :param m: Magnitud del momento aplicado en kiloNewtons-metros
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.nodos[nodo].definirCargas(px, py, m, remplazar)

    def agregarCargaPuntual(self, f, x, elemento=-1, remplazar=False):
        """
    Función que permite agregar una carga puntual a una distancia determinada de un elemento de la estructura
        :param f: Magnitud de la carga puntual en kiloNewtons
        :param x: Ubicación de la fuerza a aplicar desde el nodo inical hasta el nodo final en metros
        :param elemento: Identificador del elemento sobre el cual se aplica la nueva carga puntual
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        self.elementos[elemento].agregarCargaPuntual(f, x, remplazar)

    def agregarCargaDistribuida(self, WX=0, WY=0, elemento=-1, remplazar=False):
        """
    Función que permite agregar una carga distribuida sobre un elemento de la estructura
        :param WX: Magnitud de la carga distribuida en dirección x en kiloNewtons/metros
        :param WY: Magnitud de la carga distribuida en dirección y en kiloNewtons/metros
        :param elemento: Identificador del elemento sobre el cual se aplica/n la/s carga/s
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        s = np.sin(self.elementos[elemento].Angulo)
        c = np.cos(self.elementos[elemento].Angulo)
        self.elementos[elemento].definirCargas(-s * WY, c * WY, 0, remplazar)
        self.elementos[elemento].definirCargas(s * WX, c * WX, 0, False)

    def definirFactorPesoPropio(self, f=0, remplazar=False):
        """Función que permite incluir la carga agregada producida por el peso propio del elemento
        :param f: Factor de multipliación de peso propio (utilizado para combinaciones de carga)
        :param remplazar: Opción para remplazar las cargas anteriores o no (True = remplaza, False = agrega)
        """
        for i in range(0, self.elementos.size):
            sw = f * self.elementos[i].Area * \
                self.elementos[i].seccion.material.gamma
            self.agregarCargaDistribuida(i, WY=sw, remplazar=remplazar)

    def calcularF0(self):
        """Función que calcula el vector de fuerzas externas globales de la estructura
        """
        n = self.nodos.size * 3
        self.F0 = np.zeros([n, 1])
        for i in range(0, self.elementos.size):
            self.elementos[i].calcularF0(n)
            self.F0 = self.F0 + self.elementos[i].F0
        for i in range(0, self.superelementos.size):
            self.F0 = self.F0 + self.superelementos[i].calcularF0(n)

    def calcularFn(self):
        """Función que calcula el vector de fuerzas nodales de la estructura
        """
        n = self.nodos.size * 3
        self.Fn = np.zeros([n, 1])
        for i in range(0, self.nodos.size):
            self.nodos[i].calcularFn(n)
            self.Fn = self.Fn + self.nodos[i].Fn

    def definirDesplazamientosRestringidos(self, desplazamientos):
        """Función que permite asignar desplazamientos conocidos a los grados de libertad restringidos
        :param desplazamientos: Vector de los desplazamientos restringidos (metros o radianes según el caso)
        """
        if desplazamientos.size == self.restringidos.size:
            self.Ur = desplazamientos
        else:
            'No se asignaron los desplazamientos restringidos porque el vector no tiene el mismo tamaño.'

    def definirDesplazamientoRestringido(self, nodo, gdl, valor):
        """Función que permite asignar un desplazamiento conocido a uno de los grados de libertad restringidos
        :param nodo: Identificador del nodo sobre el que se va a asignar el desplazamiento
        :param gdl: Grado de libertad del nodo sobre el que se va a asignar el desplazamiento
        :param valor: Magnitud del desplazamiento asignado (en metros o radianes según el caso)
        """
        if any(np.isin(self.restringidos, self.nodos[nodo].gdl[gdl])):
            for i in range(0, self.restringidos.size):
                if self.restringidos[i] == self.nodos[nodo].gdl[gdl]:
                    self.Ur[i] = valor
                    break
        else:
            print('No se asignaron los desplazamientos porque el gdl' + format(
                self.nodos[nodo].gdl[gdl]) + ' no hace parte de los grados de libertad restringidos, los grados de libertad restringidos disponibles son: ' + format(self.restringidos))

    def calcularSubmatrices(self):
        """Función para crear las submatrices de la matriz de rigidez de la estructura
        """
        self.Kll = self.KE[0:self.libres.size:1, 0:self.libres.size:1]
        self.Klr = self.KE[0:self.libres.size:1,
                           self.libres.size:self.libres.size + self.restringidos.size:1]
        self.Krl = self.KE[self.libres.size:self.libres.size +
                           self.restringidos.size:1, 0:self.libres.size:1]
        self.Krr = self.KE[self.libres.size:self.libres.size + self.restringidos.size:1,
                           self.libres.size:self.libres.size + self.restringidos.size:1]

    def calcularVectorDesplazamientosLibres(self):
        """Función que halla los desplazamientos de los grados de libertad libres de las estructura
        """
        if self.Ur.size == 0:
            self.Ur = np.zeros([self.restringidos.size, 1])
        self.Fl = self.Fn - self.F0
        self.Ul = np.dot(np.linalg.pinv(self.Kll),
                         (self.Fl[self.libres] - np.dot(self.Klr, self.Ur)))

    def calcularReacciones(self):
        """Función que calcula las reacciones de los grados de libertad restringidos de la estructura
        """
        self.R0 = self.F0[self.restringidos]
        self.Rn = np.dot(self.Krl, self.Ul) + \
            np.dot(self.Krr, self.Ur) + self.R0

    def calcularVectoresDeFuerzasInternas(self):
        """Función que calcula las fuerzas internas de los elementos de la estructura
        """
        for i in range(0, self.elementos.size):
            self.elementos[i].calcularVectorDeFuerzasInternas(
                np.concatenate([self.Ul, self.Ur], axis=None))

    def hacerSuperElemento(self, gdlVisibles, gdlInvisibles):
        """Función para realizar un superelemento en la estructura
        :param gdlVisibles: Conjunto de grados de libertad que se quieren mantener
        :param gdlInvisibles: Conjunto de grados de libertad que se quieren condensar
        :return: El vector de fuerzas y la matriz de rigidez condensados
        """
        self.crearMatrizDeRigidez()
        self.calcularF0()
        self.calcularFn()
        self.calcularSubmatrices()
        self.Fl = self.Fn - self.F0
        klgc = np.dot(self.Kll[np.ix_(gdlVisibles, gdlInvisibles)],
                      (np.linalg.pinv(self.Kll[np.ix_(gdlInvisibles, gdlInvisibles)])))
        a = self.Fl[np.ix_(gdlVisibles)] - np.dot(klgc,
                                                  self.Fl[np.ix_(gdlInvisibles)])
        b = self.Kll[np.ix_(gdlVisibles, gdlVisibles)] - \
            np.dot(klgc, self.Kll[np.ix_(gdlInvisibles, gdlVisibles)])
        return a, b

    def solucionar(self, verbose=True, dibujar=False, guardar=False, carpeta='Resultados', analisis='EL', param=[]):
        """Función que resuelve el método matricial de rigidez de la estructura
        :param verbose: Opción para mostrar mensaje de análisis exitoso (True = mostrar, False = no mostrar)
        :param dibujar: Opción para realizar interfaz gráfica (True = mostrar, False = no mostrar)
        :param guardar: Opción para guardar los resultados del análisis (True = guardar, False = no guardar)
        :param carpeta: Dirección de la carpeta destinno
        """
        if analisis == 'EL':
            self.crearMatrizDeRigidez()
            self.calcularF0()
            self.calcularFn()
            self.calcularSubmatrices()
            self.calcularVectorDesplazamientosLibres()
            self.calcularVectoresDeFuerzasInternas()
            self.calcularReacciones()
            self.gdls = np.append(self.libres, self.restringidos)
            self.U = np.append(self.Ul, self.Ur)
        elif analisis == 'CR':
            #self.solucionar(verbose=False, dibujar=False, guardar=False, carpeta='',analisis='EL',param=[])
            return self.newton(param)
        if verbose:
            print(
                'Se ha terminado de calcular, puedes examinar la variable de la estructura para consultar los resultados.')
        if dibujar:
            self.pintar()
        if guardar:
            self.guardarResultados(carpeta)

    def actualizarGDL(self):
        """Función que actualiza los grados de libertad de la estructura
        """
        count = 0
        self.libres = np.array([])
        self.restringidos = np.array([])
        for i in range(0, self.nodos.size):
            if self.nodos[i].restraints[0]:
                self.nodos[i].gdl[0] = count
                self.libres = np.append(self.libres, count)
                count = count + 1 * (self.nodos[i].restraints[0])
            if self.nodos[i].restraints[1]:
                self.nodos[i].gdl[1] = count
                self.libres = np.append(self.libres, count)
                count = count + 1 * (self.nodos[i].restraints[1])
            if self.nodos[i].restraints[2]:
                self.nodos[i].gdl[2] = count
                self.libres = np.append(self.libres, count)
                count = count + 1 * (self.nodos[i].restraints[2])
        for i in range(0, self.nodos.size):
            if not self.nodos[i].restraints[0]:
                self.nodos[i].gdl[0] = count
                self.restringidos = np.append(self.restringidos, count)
                count = count + 1
            if not self.nodos[i].restraints[1]:
                self.nodos[i].gdl[1] = count
                self.restringidos = np.append(self.restringidos, count)
                count = count + 1
            if not self.nodos[i].restraints[2]:
                self.nodos[i].gdl[2] = count
                self.restringidos = np.append(self.restringidos, count)
                count = count + 1
        self.libres = self.libres.astype(int)
        self.restringidos = self.restringidos.astype(int)

    def pintar(self):
        """Función que permite correr la interfaz de visualización de resultados de modelo, fuerzas internas y desplazamientos obtenidos
        """
        maxcoord = 0
        for i in self.elementos:
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
        for i in self.nodos:
            contador = 0
            for f in range(0, self.superelementos.size):
                k = self.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult - 10),
                                                   margen + i.x * mult +
                                                   10 + 10 * (g + 1) / 2,
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
        for i in self.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
            canvas.create_text(margen + xx * mult + 10, -margen +
                               wiw - yy * mult - 10, fill="red", text=format(count))
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
        for k in self.resortes:
            i = k.nodo
            canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult), margen + i.x * mult + 10,
                               -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
            for j in range(0, 3):
                canvas.create_line(margen + i.x * mult + 10 + 10 * j / 2, -margen + wiw - (i.y * mult + 10),
                                   margen + i.x * mult + 10 + 10 * j /
                                   2, -margen + wiw - (i.y * mult - 10),
                                   fill="#52E3C4", width=2)
                canvas.create_line(margen + i.x * mult + 10 + 10 * j / 2, -margen + wiw - (i.y * mult - 10),
                                   margen + i.x * mult + 10 + 10 *
                                   (j + 1) / 2, -margen +
                                   wiw - (i.y * mult + 10),
                                   fill="#52E3C4", width=2)
            canvas.create_line(margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult + 10),
                               margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
            canvas.create_line(margen + i.x * mult + 10 + 10, -margen + wiw - (i.y * mult), margen + i.x * mult + 30,
                               -margen + wiw - (i.y * mult), fill="#52E3C4", width=2)
        for i in self.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -
                                    margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)

            if i.restraints == [False, False, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + tmañoNodo, -margen +
                                   wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
            elif i.restraints == [True, False, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 0 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + tmañoNodo, -margen +
                                   wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 1,
                                   -margen + wiw -
                                   (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 1,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 5,
                                   -margen + wiw -
                                   (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 5,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
                canvas.create_oval(margen + i.x * mult - tmañoNodo - r + 9,
                                   -margen + wiw -
                                   (i.y * mult - 2 * tmañoNodo - 2 * r + 2) - r,
                                   margen + i.x * mult - tmañoNodo + r + 9,
                                   -margen + wiw - (i.y * mult - 2 * tmañoNodo - 2 * r + 2) + r, fill="blue", width=0)
            elif not i.restraints == [True, True, True]:
                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   fill="blue", width=2)

                canvas.create_line(margen + i.x * mult - tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult - 2 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - 3 * tmañoNodo),
                                   fill="blue", width=2)
                canvas.create_line(margen + i.x * mult + tmañoNodo * 2, -margen + wiw - (i.y * mult - 2 * tmañoNodo),
                                   margen + i.x * mult + 2 * tmañoNodo, -
                                   margen + wiw - (i.y * mult - 3 * tmañoNodo),
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
        for i in self.nodos:
            contador = 0
            for f in range(0, self.superelementos.size):
                k = self.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult - 10),
                                                   margen + i.x * mult +
                                                   10 + 10 * (g + 1) / 2,
                                                   -margen + wiw - (i.y * mult + 10), fill="#C0C0C0", width=2)
                            canvas.create_line(margen + i.x * mult - 30, -margen + wiw - (i.y * mult),
                                               margen + i.x * mult - 20, -margen + wiw - (i.y * mult), fill="#C0C0C0",
                                               width=2)
                            flag = True
                            break
                    if flag:
                        break
        for i in self.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
        for i in self.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -
                                    margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)
            for j in self.restringidos:
                if j == i.gdl[1]:
                    canvas.create_line(margen + i.x * mult, -margen + wiw - i.y * mult - tmañoNodo, margen + i.x * mult,
                                       -margen + wiw - (i.y * mult + 50), fill="red", width=2)
                    canvas.create_text(margen + i.x * mult, -margen + wiw - (i.y * mult + 60),
                                       text=format(np.round(self.Rn[j - self.libres.size][0], 3)), fill="red")
                    canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50), margen + i.x*mult - 10,
                                       -margen + wiw - (i.y * mult + 40), fill="red", width=2)
                    canvas.create_line(margen + i.x * mult, -margen + wiw - (i.y * mult + 50), margen + i.x * mult + 10,
                                       -margen + wiw - (i.y * mult + 40), fill="red", width=2)
                if j == i.gdl[0]:
                    canvas.create_line(margen + i.x * mult + tmañoNodo, -margen + wiw - i.y * mult,
                                       margen + i.x * mult + 50, -margen + wiw - (i.y * mult), fill="green", width=2)
                    canvas.create_text(margen + i.x * mult + 80, -margen + wiw - (i.y * mult + 7),
                                       text=format(np.round(self.Rn[j - self.libres.size][0], 3)), fill="green")
                    canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult), margen + i.x * mult + 40,
                                       -margen + wiw - (i.y * mult + 10), fill="green", width=2)
                    canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult), margen + i.x * mult + 40,
                                       -margen + wiw - (i.y * mult - 10), fill="green", width=2)
                if j == i.gdl[2]:
                    canvas.create_text(margen + i.x * mult + 10, -margen + wiw - (i.y * mult - 13),
                                       text='M: ' + format(np.round(self.Rn[j - self.libres.size][0], 3)), fill="blue")
            canvas.create_text(margen + i.x * mult, -margen + wiw - i.y * mult, fill="black", text=format(i.ID),
                               justify='center', font=("TkDefaultFont", tmañoNodo + 1))
        canvas.pack()
        canvas.mainloop()

        window = tkinter.Tk()
        window.title("Desplazamientos")
        canvas = tkinter.Canvas(window, width=wiw, height=wiw)
        for i in self.nodos:
            contador = 0
            for f in range(0, self.superelementos.size):
                k = self.superelementos[f]
                flag = False
                for j in range(0, 3):
                    for h in range(0, 6):
                        if i.gdl[j] == k.gdl[h]:
                            for g in range(-6, -3):
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult + 10),
                                                   margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw - (i.y * mult - 10), fill="#C0C0C0", width=2)
                                canvas.create_line(margen + i.x * mult + 10 + 10 * g / 2,
                                                   -margen + wiw -
                                                   (i.y * mult - 10),
                                                   margen + i.x * mult +
                                                   10 + 10 * (g + 1) / 2,
                                                   -margen + wiw - (i.y * mult + 10), fill="#C0C0C0", width=2)
                            canvas.create_line(margen + i.x * mult - 30, -margen + wiw - (i.y * mult),
                                               margen + i.x * mult - 20, -margen + wiw - (i.y * mult), fill="#C0C0C0",
                                               width=2)
                            flag = True
                            break
                    if flag:
                        break
        for i in self.elementos:
            canvas.create_line(margen + i.nodoI.x * mult, -margen + wiw - i.nodoI.y * mult, margen + i.nodoF.x * mult,
                               -margen + wiw - i.nodoF.y * mult, fill="gray", width=2)
            xx = i.nodoF.x - (i.nodoF.x - i.nodoI.x) / 2
            yy = i.nodoF.y - (i.nodoF.y - i.nodoI.y) / 2
        for i in self.nodos:
            canvas.create_rectangle(margen + i.x * mult - tmañoNodo, -margen + wiw - (i.y * mult - tmañoNodo),
                                    margen + i.x * mult + tmañoNodo, -
                                    margen + wiw - (i.y * mult + tmañoNodo),
                                    fill="#F1C531", width=1)
            for j in self.gdls:
                if not self.U[j] == 0:
                    if j == i.gdl[1]:
                        canvas.create_line(margen + i.x * mult, -margen + wiw - i.y * mult - tmañoNodo,
                                           margen + i.x * mult, -margen + wiw - (i.y * mult + 50), fill="red", width=2)
                        canvas.create_text(margen + i.x * mult, -margen + wiw - (i.y * mult + 60),
                                           text=format(np.round(self.U[j], 4)), fill="red")
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
                                           text=format(np.round(self.U[j], 4)), fill="green")
                        canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult),
                                           margen + i.x * mult + 40, -margen + wiw - (i.y * mult + 10), fill="green",
                                           width=2)
                        canvas.create_line(margen + i.x * mult + 50, -margen + wiw - (i.y * mult),
                                           margen + i.x * mult + 40, -margen + wiw - (i.y * mult - 10), fill="green",
                                           width=2)
                    if j == i.gdl[2]:
                        canvas.create_text(margen + i.x * mult + 10, -margen + wiw - (i.y * mult - 13),
                                           text='R: ' + format(np.round(self.U[j], 4)), fill="blue")
            canvas.create_text(margen + i.x * mult, -margen + wiw - i.y * mult, fill="black", text=format(i.ID),
                               justify='center', font=("TkDefaultFont", tmañoNodo + 1))
        canvas.pack()
        canvas.mainloop()

    def guardarResultados(self, carpeta):
        """Función para generar y guardar un archivo con los resultados del análisis obtenido
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
        np.savetxt(path + "/Rn.csv", self.Rn, delimiter=",")
        np.savetxt(path + "/Ul.csv", self.Ul, delimiter=",")
        np.savetxt(path + "/K.csv", self.KE, delimiter=",")
        np.savetxt(path + "/F0.csv", self.F0, delimiter=",")
        np.savetxt(path + "/Fn.csv", self.Fn, delimiter=",")
        for i in range(0, self.elementos.size):
            np.savetxt(path + '/Vector p0 Elemento (Locales) ' + format(i) + '.csv',
                       self.elementos[i].p0, delimiter=",")
            np.savetxt(path + '/Vector P0 Elemento (Globales) ' + format(i) + '.csv',
                       self.elementos[i].P0, delimiter=",")
            np.savetxt(path + '/Matriz Elemento ' + format(i) +
                       '.csv', self.elementos[i].Ke, delimiter=",")
        temporal = np.array([])
        for i in range(0, self.elementos.size):
            temporal = np.append(temporal, [self.elementos[i].p])
        temporal = temporal.reshape(self.elementos.size, 6)
        np.savetxt(path + '/Vectores p Elementos.csv',
                   temporal.T, delimiter=",")


def tridiag(a, b, c, n):
    """Función auxiliar para crear una matriz tridiagonal (utilizada en FEM)
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


def estadoPlasticidadConcentrada(vt, sh, qy, EI, l, EA, tipo, v0, q=[[0], [0], [0]]):
    qy = np.array(qy)
    vt = np.array(vt)
    v0 = np.array(v0)
    q = np.array(q)
    error = 1
    i = 1
    while error > 1*10**-10 and i <= 50:
        q1 = np.min([q[0][0], -1*10**-5])
        psi = np.sqrt(((-q1)*l**2)/(EI))
        fe = _fe(psi, l, EI, EA, tipo)
        fp = _fp(q, qy, EI, l, sh, EA)
        if np.abs(q[0][0]) > np.abs(qy[0][0]):
            fe = np.zeros([3, 3])
            fp = np.zeros([3, 3])
        kb = np.linalg.pinv(fe + fp)
        ve = fe @ q
        vp = fp @ (q - np.abs(qy)*np.sign(q))
        v = vp + ve
        Re = vt - v0 - v
        dq = kb @ Re
        if np.abs(q[0][0]) > np.abs(qy[0][0]):
            q = [[qy[0][0]], [0], [0]]
            ve = [[l/EA*qy[0][0]], [0], [0]]
            vp = vt - ve
            v = vp + ve
            Re = np.zeros([3, 1])
            kb = np.zeros([3, 3])
            break
        q = q + dq
        i += 1
        error = np.linalg.norm(Re)
        print('Error q: ' + format(error) + ' iteracion ' + format(i))
    return Re, v, q, kb, ve, vp


def _fp(q, qy, EI, l, sh, EA=1, sh2=None):
    alpha0 = 1*(1-(np.abs(q[0][0]) <= np.abs(qy[0][0])))
    alpha1 = 1*(1-(np.abs(q[1][0]) <= np.abs(qy[1][0])))
    alpha2 = 1*(1-(np.abs(q[2][0]) <= np.abs(qy[2][0])))
    kbc2 = (6*EI/l)*sh
    if sh2 == None:
        sh2 = sh
    kbc3 = (6*EI/l)*sh2
    return np.array([[alpha0*l/EA, 0, 0], [0, alpha1/kbc2, 0], [0, 0, alpha2/kbc3]])


def _fe(psi, l, EI, EA, tipo=Tipo.UNO):
    L = l
    if psi < 0.001:
        kb1 = 4*EI/L
        kb2 = 2*EI/L
        kb3 = 3*EI/L
    else:
        kb1 = ((EI)/(L))*((psi*(np.sin(psi)-psi*np.cos(psi))) /
                          (2-2*np.cos(psi)-psi*np.sin(psi)))
        kb2 = (EI*(psi*(psi-np.sin(psi)))) / \
            (L*(2-2*np.cos(psi)-psi*np.sin(psi)))
        kb3 = (L*(np.sin(psi)-psi*np.cos(psi)))/(EI*(psi**2*np.sin(psi)))
    fe1 = (kb1)/(kb1**2-kb2**2)
    fe2 = -(kb2)/(kb1**2-kb2**2)
    fe3 = kb3
    if tipo == Tipo.UNO:
        fe = np.array([[L/EA, 0, 0], [0, fe1, fe2], [0, fe2, fe1]])
    elif tipo == Tipo.DOS:
        fe = np.array([[L/EA, 0, 0], [0, 0, 0], [0, 0, fe3]])
    elif tipo == Tipo.TRES:
        fe = np.array([[L/EA, 0, 0], [0, fe3, 0], [0, 0, 0]])
    else:
        fe = np.array([[L/EA, 0, 0], [0, 0, 0], [0, 0, 0]])
    return fe


def calcularPsi(q, l, EI):
    q1 = np.min([q[0][0], -1*10**-5])
    return
