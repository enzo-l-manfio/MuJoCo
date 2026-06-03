import sympy as sp



t = sp.symbols('t')

theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta3 = sp.Function('theta3')(t)

theta = sp.Matrix([theta1, theta2, theta3])

c1 = sp.cos(theta1)
s1 = sp.sin(theta1)
c2 = sp.cos(theta2)
s2 = sp.sin(theta2)
s3 = sp.sin(theta3)
c3 = sp.cos(theta3)
c23 = sp.cos(theta2 + theta3)
s23 = sp.sin(theta2 + theta3)

L1, L2, L3 = sp.symbols('L1 L2 L3')

m1, m2, m3 = sp.symbols('m1 m2 m3')

R1 = sp.Matrix([[c1, -s1, 0],
                [s1, c1, 0],
                [0, 0, 1]])

R2 =  R1 @ sp.Matrix([[1, 0, 0],
                      [0, c2, s2],
                      [0, -s2, c2]])

R3 = R2 @ sp.Matrix([[1, 0, 0],
                [0, c3, s3],
                [0, -s3, c3]])

T1 = R1
T2 = R1 @ R2
T3 = R2 @ R3

J1 = sp.Matrix([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])

J2 = L2/2 * sp.Matrix([[c1 * s2, c1 * c2, 0],
                [-s1 * s2, s1 * c2, 0],
                [0, -s2, 0]])

u1 = L2 * s2 + L3 * s23/2
u2 = L2 * c2 + L3 * c23/2
u3 = L3 * c23/2

J3 = sp.Matrix([[c1 * u1, s1 * u2, s1*u3],
                [s1 * u1, -c1 * u2, -c1 * u3],
                [0, -u2, -L3 * s23/2]])

J_efetuador = sp.Matrix([[c1 * u1, s1 * u2, s1*u3],
                         [s1 * u1, -c1 * u2, -c1 * u3],
                         [0, -u2, -L3 * s23]])

derivada_J_efetuador = sp.diff(J_efetuador, t)

funcao_derivada_J_efetuador = sp.lambdify((theta1, theta2, theta3, sp.diff(theta1, t), sp.diff(theta2, t), sp.diff(theta3, t)), derivada_J_efetuador)


I_xx_1, I_yy_1, I_zz_1 = sp.symbols('I_xx_1 I_yy_1 I_zz_1')
tensor_inercia_local_1 = sp.diag(I_xx_1, I_yy_1, I_zz_1)
I_xx_2, I_yy_2, I_zz_2 = sp.symbols('I_xx_2 I_yy_2 I_zz_2')
tensor_inercia_local_2 = sp.diag(I_xx_2, I_yy_2, I_zz_2)
I_xx_3, I_yy_3, I_zz_3 = sp.symbols('I_xx_3 I_yy_3 I_zz_3')
tensor_inercia_local_3 = sp.diag(I_xx_3, I_yy_3, I_zz_3)

tensores_inercia_globais = [T1 @ tensor_inercia_local_1 @ T1.T,
                            T2 @ tensor_inercia_local_2 @ T2.T,
                            T3 @ tensor_inercia_local_3 @ T3.T]

inercias_translacionais_globais = [m1* J1.T @ J1, m2* J2.T @ J2, m3* J3.T @ J3]

matriz_inercia = sp.zeros(3)
for i in range(3):
    matriz_inercia += tensores_inercia_globais[i] + inercias_translacionais_globais[i]

matriz_inercia = sp.simplify(matriz_inercia)

derivada_theta = sp.diff(theta, t)
derivada_matriz_inercia = sp.diff(matriz_inercia, t)

energia_cinetica = 0.5 * derivada_theta.T @ matriz_inercia @ derivada_theta
derivada_energia_cinetica = sp.Matrix([sp.diff(energia_cinetica, theta_i) for theta_i in theta])

termo_coriolis = derivada_matriz_inercia @ derivada_theta - derivada_energia_cinetica

termo_coriolis = sp.simplify(termo_coriolis)

funcao_coriolis = sp.lambdify((theta1, theta2, theta3, sp.diff(theta1, t), sp.diff(theta2, t), sp.diff(theta3, t)), termo_coriolis)



if __name__ == "__main__":

    g = sp.symbols('g')

    P1 = m1*g*L1/2
    P2 = m2*g*(L1 + L2/2 * c2)
    P3 = m3*g*(L1 + u2)

    energia_potencial = P1 + P2 + P3

    termo_gravitacional = sp.diff(energia_potencial, theta)
    termo_gravitacional = sp.simplify(termo_gravitacional)

    print("Matriz de Inércia:")
    sp.pprint(matriz_inercia)

    print("\nTermo Gravitacional:")
    sp.pprint(termo_gravitacional)

    print("\nTermo de Coriolis:")
    sp.pprint(termo_coriolis)

    with open('dinamica.tex', 'w') as f:
        f.write("Matriz de Inércia:\n")
        f.write(sp.latex(matriz_inercia))
        f.write("\n\nTermo de Coriolis:\n")
        f.write(sp.latex(termo_coriolis))
        f.write("\n\nTermo Gravitacional:\n")
        f.write(sp.latex(termo_gravitacional))
