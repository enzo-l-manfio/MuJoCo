import numpy as np
from dataclasses import dataclass
#from expressoes_simbolicas import funcao_coriolis, funcao_derivada_J_efetuador

@dataclass
class Link:
    comprimento: float
    massa: float
    tensor_inercia: np.ndarray


class Manipulador3DOF:
    
    def __init__(self, links, theta_i = np.array([0.0, 0.0, 0.0])):
        self.links = links
        self.L1 = links[0].comprimento
        self.L2 = links[1].comprimento
        self.L3 = links[2].comprimento
        self.tensores_inercia = [link.tensor_inercia for link in links]
        self.massas = [link.massa for link in links]

        #Parâmetros trigonomtericos
        self.atualizar_posicao(theta_i)


    def atualizar_posicao(self, theta):
        self.theta = theta
        theta_23 = theta[1] + theta[2]
        self.s1 = np.sin(theta[0])
        self.c1 = np.cos(theta[0])
        self.s2 = np.sin(theta[1])
        self.c2 = np.cos(theta[1])
        self.c3 = np.cos(theta[2])
        self.s3 = np.sin(theta[2])
        self.s23 = np.sin(theta_23)
        self.c23 = np.cos(theta_23)


    def cinematica_direta_efetuador(self):

        u_1 = self.L2 * self.s2 + self.L3 * self.s23

        x = self.s1 * u_1
        y = -self.c1 * u_1
        z = self.L1 + self.L2*self.c2 + self.L3 * self.c23
        return np.array([x, y, z])

    def Jacobiana_Efetuador(self):

        u_1 = self.L2 * self.s2 + self.L3 * self.s23
        u_2 = self.L2 * self.c2 + self.L3 * self.c23
        u_3 = self.L3 * self.c23

        return np.array([
            [ self.c1 * u_1,  self.s1 * u_2,  self.s1 * u_3],
            [ self.s1 * u_1, -self.c1 * u_2, -self.c1 * u_3],
            [      0,     -u_1,      -self.L3 * self.s23]
        ])
    
    '''
    def derivada_Jacobiana_Efetuador(self, velocidades_angulares):
        return funcao_derivada_J_efetuador(self.theta[0], self.theta[1], self.theta[2], velocidades_angulares[0], velocidades_angulares[1], velocidades_angulares[2])
    '''

    def cinematica_direta_link1(self):
        return np.array([0, 0, self.L1])
    
    def cinematica_direta_link2(self):
    
        x = self.L2 * self.s2 * self.s1 / 2
        y = self.L2 * self.s2 * self.c1 / 2
        z = self.L1 + self.L2 * self.c2 / 2
        return np.array([x, y, z])
    
    def rotacao_global_para_link1(self):
        return np.array([[self.c1, -self.s1, 0],
                         [self.s1, self.c1, 0],
                         [0, 0, 1]])
    
    def rotacao_global_para_link2(self):
        R1 = self.rotacao_global_para_link1()
        R2 = np.array([[1, 0, 0],
                       [0, self.c2, self.s2],
                       [0, -self.s2, self.c2]])
        return R1 @ R2
    
    def rotacao_global_para_link3(self):
        R2 = self.rotacao_global_para_link2()
        R3 = np.array([[1, 0, 0],
                       [0, self.c3, self.s3],
                       [0, -self.s3, self.c3]])
        return R2 @ R3

    #Jacobianas das posições dos CMs de cada link
    def Jacobiana_Link1(self):
        return np.zeros((3, 3))  #O eixo de rotação do link 1 passa pelo seu CM, logo não o movimenta
    
    def Jacobiana_Link2(self):

        return self.L2/2 * np.array([[self.s2*self.c1, self.c2*self.c1, 0],
                                     [-self.s2*self.s1, self.c2*self.s1, 0],
                                     [0, -self.s2, 0]])

    def Jacobiana_Link3(self):
        #O CM do link 3 está no centro do link, logo sua jacobiana é a mesma do efetuador, mas com o comprimento do link 3 reduzido pela metade
        self.L3 = self.L3/2
        jacobiana_link3 = self.Jacobiana_Efetuador()
        self.L3 = self.L3*2
        return jacobiana_link3

    def cinematica_inversa_numerica(self, posicao_efetuador):

        theta = np.array([0.1, 0.1, 0.1])
        for _ in range(10000):
            posicao_calculada = self.cinematica_direta_efetuador(theta)
            erro = posicao_efetuador - posicao_calculada
            if np.linalg.norm(erro) < 1e-4:
                break
            J = self.Jacobiana_Efetuador()
            try:
                inversa_J = np.linalg.inv(J)
                theta += inversa_J @ erro
                theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
            except np.linalg.LinAlgError:
                print("Jacobian is singular at this configuration.")
                break
        return theta
    
    def cinematica_inversa_analitica(self, posicao_efetuador):
        x, y, z = posicao_efetuador
        theta1 = np.arctan2(x, -y)

        r = np.sqrt(x**2 + y**2)
        s = z - self.L1
        d = np.sqrt(r**2 + s**2)

        if d > self.L2 + self.L3 or d < abs(self.L2 - self.L3):
            raise ValueError("A posição desejada está fora do alcance do manipulador.")
        
        cos_theta3 = (d**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)

        if abs(cos_theta3) > 1:
            raise ValueError("A posição desejada está fora do alcance do manipulador.")
        
        theta3 = np.arccos(cos_theta3)

        cos_alpha2 = (self.L3**2 - self.L2**2 - d**2) / -(2 * self.L2 * d)

        if abs(cos_alpha2) > 1:
            raise ValueError("A posição desejada está fora do alcance do manipulador.")
        
        theta2 = np.pi/2 - np.arccos(cos_alpha2) - np.arctan2(s, r)


        return np.array([theta1, theta2, theta3])
    

    def velocidades_angulares(self, velocidades_lineares):
        J = self.Jacobiana_Efetuador()
        try:
            inversa_J = np.linalg.inv(J)
            return inversa_J @ velocidades_lineares
        except np.linalg.LinAlgError:
            return np.zeros(3)
        
    def aceleracoes_angulares(self, velocidades_angulares, aceleracoes_lineares):
        J = self.Jacobiana_Efetuador()
        dJ_dt = self.derivada_Jacobiana_Efetuador(velocidades_angulares)
        try:
            inversa_J = np.linalg.inv(J)
            return inversa_J @ (aceleracoes_lineares - dJ_dt @ velocidades_angulares)
        except np.linalg.LinAlgError:
            return np.zeros(3)
        
    
    def matriz_inercia(self):
        matriz_inercia = np.zeros((3, 3))

        Rg1 = self.rotacao_global_para_link1()
        Rg2 = self.rotacao_global_para_link2()
        Rg3 = self.rotacao_global_para_link3()

        J1 = self.Jacobiana_Link1()
        J2 = self.Jacobiana_Link2()
        J3 = self.Jacobiana_Link3()
        
        tensores_inercia_frame_global = [Rg1 @ self.tensores_inercia[0] @ Rg1.T,
                                         Rg2 @ self.tensores_inercia[1] @ Rg2.T,
                                         Rg3 @ self.tensores_inercia[2] @ Rg3.T]
        
        inercias_translacionais = [self.massas[0] * J1.T @ J1,
                                   self.massas[1] * J2.T @ J2,
                                   self.massas[2] * J3.T @ J3]
        
        for i in range(3):
            matriz_inercia += tensores_inercia_frame_global[i] + inercias_translacionais[i]

        return matriz_inercia


    '''
    def termo_coriolis(self, w):

        return funcao_coriolis(self.theta[0], self.theta[1], self.theta[2], w[0], w[1], w[2]).squeeze()
    '''
    
    def termo_gravitacional(self):
        g = 9.81
        G1 = 0
        G2 = self.L2*self.s2*(self.massas[1] + 2*self.massas[2]) + self.L3*self.massas[2]*self.s23
        G3 = self.massas[2] * self.L3 * self.s23
        return -g/2 * np.array([G1, G2, G3])
    
    def torque_necessario(self, velocidades_angulares, aceleracoes_angulares):
        M = self.matriz_inercia()
        C = self.termo_coriolis(velocidades_angulares)
        G = self.termo_gravitacional()
        newton = M @ aceleracoes_angulares

        return newton + C + G

    


if __name__ == "__main__":
    
    LinkGeral = Link(comprimento=0.5, massa=5.9, tensor_inercia=np.diag([0.14, 0.14, 0.0068]))

    manipulador = Manipulador3DOF(links=[LinkGeral, LinkGeral, LinkGeral], theta_i=np.array([0.0, 0.0, np.pi/2]))

    posicao_efetuador = manipulador.cinematica_direta_efetuador()

    print("Posição do efetuador:", posicao_efetuador)   

