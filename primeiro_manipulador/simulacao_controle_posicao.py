import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import dinamica as din
import matplotlib.pyplot as plt

TensorLinkGeral = np.diag([0.14, 0.14, 0.0068])
massaLinkGeral = 5.9
L = 0.5
LinkGeral = din.Link(comprimento=L, massa=massaLinkGeral, tensor_inercia=TensorLinkGeral)

manipulador = din.Manipulador3DOF(links=[LinkGeral, LinkGeral, LinkGeral], theta_i=np.array([0.0, 0.0, 0.0]))

def trajetoria(t):

    if t >= 1.0:
        theta1 = -np.pi/14
        theta2 = 0.0
        theta3 = np.pi/10
    else:
        theta1 = 0.0
        theta2 = 0.0
        theta3 = 0.0

    return np.array([theta1, theta2, theta3])


xml_path = "manipulador_atuadores_posicao.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

tempo  = 0.0
dt = sim.model.opt.timestep

posicao_inical = trajetoria(tempo)

sim.data.qpos[0] = posicao_inical[0]
sim.data.qpos[1] = posicao_inical[1]
sim.data.qpos[2] = posicao_inical[2]


Kp = 0.5
Kd = 0.015
Ki = 0.0

erro_juntas_anterior = np.zeros(3)
integral_erro = np.zeros(3)

durante = 1000

erros_x = np.array([])
erros_y = np.array([])
erros_z = np.array([])

posicao_junta1_log = []
posicao_junta2_log = []
posicao_junta3_log = []

for _ in range(durante):

    theta1 = sim.data.qpos[0]
    theta2 = sim.data.qpos[1]
    theta3 = sim.data.qpos[2]

    posicao_junta1_log.append(theta1)
    posicao_junta2_log.append(theta2)
    posicao_junta3_log.append(theta3)

    theta_atual = np.array([theta1, theta2, theta3])
    manipulador.atualizar_posicao(theta_atual)

    posicao_referencia_juntas = trajetoria(tempo)

    erro_juntas = posicao_referencia_juntas - theta_atual
    derivada_erro = (erro_juntas - erro_juntas_anterior) / dt
    integral_erro += erro_juntas * dt

    erro_juntas_anterior = erro_juntas

    sinal_controle = Kp * erro_juntas + Kd * derivada_erro + Ki * integral_erro

    comando = theta_atual + sinal_controle

    sim.data.ctrl[0] = comando[0]
    sim.data.ctrl[1] = comando[1]
    sim.data.ctrl[2] = comando[2]


    erros_x = np.append(erros_x, erro_juntas[0])
    erros_y = np.append(erros_y, erro_juntas[1])
    erros_z = np.append(erros_z, erro_juntas[2])

    print(f"Tempo: {tempo:.2f} s, Erro Juntas 1: {erro_juntas[0]:.4f}, Erro Juntas 2: {erro_juntas[1]:.4f}, Erro Juntas 3: {erro_juntas[2]:.4f}")

    sim.step()
    viewer.render()
    tempo += dt


t = np.arange(0, durante*dt, dt)

trajetoria_desejada_1 = [trajetoria(ti)[0] for ti in t]
trajetoria_desejada_2 = [trajetoria(ti)[1] for ti in t]
trajetoria_desejada_3 = [trajetoria(ti)[2] for ti in t]

figure, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(t, trajetoria_desejada_1, label='Trajetória Desejada 1')
axs[0].plot(t, posicao_junta1_log, label='Posição Junta 1')
axs[0].set_xlabel('Tempo (s)')
axs[0].set_ylabel('ângulo (rad)')
axs[0].set_title(f'kp = {Kp}, kd = {Kd}, ki = {Ki}')
axs[0].legend('upper right')
axs[0].grid()

axs[1].plot(t, trajetoria_desejada_2, label='Trajetória Desejada 2')
axs[1].plot(t, posicao_junta2_log, label='Posição Junta 2')
axs[1].set_xlabel('Tempo (s)')
axs[1].set_ylabel('ângulo (rad)')
axs[1].legend('upper right')
axs[1].grid()

axs[2].plot(t, trajetoria_desejada_3, label='Trajetória Desejada 3')
axs[2].plot(t, posicao_junta3_log, label='Posição Junta 3')
axs[2].set_xlabel('Tempo (s)')
axs[2].set_ylabel('ângulo (rad)')
axs[2].legend('upper right')
axs[2].grid()

n = 9
plt.savefig(f"erros_juntas_step[{n}].png")
plt.show()