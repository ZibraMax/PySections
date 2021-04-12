import quadpy


def estadoPlasticidadConcentradaFeo(vt, sh, qy, EI, l, EA=1, v0=[[0], [0], [0]]):
    qy = np.array(qy)
    vt = np.array(vt)
    v0 = np.array(v0)
    q = np.array([np.array([[0], [0], [0]])], dtype=np.ndarray)
    v = np.array([])
    psi = calcularPsi(q[-1], l, EI)
    fp = np.array([_fp(q[-1], qy, EI, l, sh)])
    fe = np.array([_fe(psi, l, EI, EA)])
    flex = fe[-1] + fp[-1]
    kb = np.array([np.linalg.inv(flex)])
    signos = np.sign(q[-1])
    vp = np.array([np.dot(fp[-1], (q[-1] - qy * (signos)))])
    v = np.array([np.array([[0], [0], [0]])], dtype=np.ndarray)
    ve = np.array([np.array([[0], [0], [0]])])
    i = 0
    while i < 30:
        Re = vt - v[-1] - v0
        dq = np.dot(kb[-1], Re)
        q = np.append(q, [q[-1] + dq], axis=0)
        fp = np.append(fp, [_fp(q[-1], qy, EI, l, sh)], axis=0)
        psi = calcularPsi(q[-1], l, EI)
        fe = np.append(fe, [_fe(psi, l, EI, EA)], axis=0)
        flex = fe[-1] + fp[-1]
        kb = np.append(kb, [np.linalg.inv(flex)], axis=0)
        signos = np.sign(q[-1])
        vp = np.append(vp, [np.dot(fp[-1], (q[-1] - qy * (signos)))], axis=0)
        ve = np.append(ve, [np.dot(fe[-1], q[-1])], axis=0)
        v = np.append(v, [ve[-1] + vp[-1]], axis=0)
        i += 1
    return Re, v, q, kb, ve, vp


def estadoPlasticidadConcentrada(vt, sh, qy, EI, l, EA=1, v0=[[0], [0], [0]], q=[[0], [0], [0]]):
    qy = np.array(qy)
    vt = np.array(vt)
    v0 = np.array(v0)
    q = np.array(q)
    error = 1
    i = 1
    while error > 1 * 10 ** -10:
        psi = calcularPsi(q, l, EI)
        fe = _fe(psi, l, EI, EA)
        fp = _fp(q, qy, EI, l, sh)
        kb = np.linalg.pinv(fe + fp)
        ve = fe @ q
        vp = fp @ (q - np.abs(qy) * np.sign(q))
        v = vp + ve
        Re = vt - v0 - v
        dq = kb @ Re
        q = q + dq
        i += 1
        error = np.linalg.norm(Re)
        print('Error q: ' + format(error) + ' iteracion ' + format(i))
    return Re, v, q, kb, ve, vp


def estadoPlasticidadDistribuida(vt, l, v0=[[0], [0], [0]], q=[[0], [0], [0]], n=5):
    vt = np.array(vt)
    v0 = np.array(v0)
    q = np.array(q)
    i = 0
    error = 1
    while error > 1 * 10 ** -10:
        v, kb = _vkb(l, q, n)
        Re = vt - v0 - v
        dq = kb @ Re
        q = q + dq
        i += 1
        error = np.max(Re)
        print('Error q: ' + format(error) + ' iteracion ' + format(i))
    return Re, v, q, kb


def _vkb(L, q, n):
    s = quadpy.line_segment.gauss_lobatto(n)
    X = (np.array(s.points) / 2 + 1 / 2) * L
    W = np.array(s.weights) / 2 * L

    v = np.zeros([3, 1])
    kb = np.zeros([3, 3])
    for i in range(0, len(X)):
        x = X[i]
        b = np.array([[1, 0, 0], [0, x / L - 1, x / L]])
        St = b @ q
        e = np.zeros([2, 1])
        fibras = crearFibras()
        error = 1
        j = 1
        while error > 1 * 10 ** -6 and j < 30:
            S, Ks = _estadoSeccion(e, fibras)
            Rs = St - S
            de = np.linalg.pinv(Ks) @ Rs
            e = e + de
            error = np.linalg.norm(de)
            clear_output(wait=True)
            print('Error s: ' + format(error) + ' iteracion ' + format(i) + ',' + format(j))
            j += 1
        v = v + W[i] * (b.T @ e)
        kb = kb + W[i] * (b.T @ np.linalg.pinv(Ks) @ b)
    kb = np.linalg.pinv(kb)
    return v, kb


def _estadoSeccion(e, fibras):
    n = fibras.shape[0]
    ea = e[0][0]
    phi = e[1][0]
    ys = fibras[:, 0].reshape(n, 1)
    ai = fibras[:, 1].reshape(n, 1)

    epsilon = ea - ys * phi  # Pueden fallar
    sigma, Et = _esfdeft(epsilon)
    Sm = sigma * ai
    C = (Et * ai).T[0]  # Puede Fallar
    km = np.diag(C)
    As = np.array([np.zeros([n]) + 1, -ys.T[0]]).T

    S = As.T @ Sm

    Ks = As.T @ km @ As
    return S, Ks


def _esfdeft(epsilon, sh=0.015):  # CURVA ESFUERZO DEFORMACION
    sh = 1 - sh
    ey = 0.001725
    et = 200000000 - sh * 200000000 * (np.abs(epsilon) > ey)
    s = et * (epsilon - ey * np.sign(epsilon) * (np.abs(epsilon) > ey)) + ey * np.sign(epsilon) * 200000000 * (
                np.abs(epsilon) > ey)
    return s, et


# Cambair la seccion transversar es cuestion del usuario
def crearFibras():
    tf = 0.940 * 2.54 / 100
    th = 0.590 * 2.54 / 100
    a = 14.670 * 2.54 / 100
    b = 14.48 * 2.54 / 100 - 2 * tf

    tf = 0.01143 * 2
    th = 0.01397
    a = 0.3099
    b = 0.09229 * 3

    a1 = a * tf / 2
    a2 = b * th / 3

    d1 = tf / 2
    d2 = b / 3
    fibras = [[-(d2 / 2 + d2 + d1 + d1 / 2), a1],
              [-(d2 / 2 + d2 + d1 / 2), a1],
              [-(d2), a2],
              [0, a2],
              [(d2), a2],
              [(d2 / 2 + d2 + d1 / 2), a1],
              [(d2 / 2 + d2 + d1 + d1 / 2), a1]]
    return np.array(fibras)


vt = [[-0.001076], [0.05186], [-0.01091]]
l = 4.5

Re, v, q, kb = estadoPlasticidadDistribuida(vt, l, v0=[[0], [0], [0]], n=5)
print('Vector {v}')
print(v)
print('\nMatriz [kb]')
print(kb)
print('\nVector {Re}')
print(Re)
print('\nVector q {q}')
print(q)
