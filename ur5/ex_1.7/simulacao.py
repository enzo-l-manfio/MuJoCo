import os
import numpy as np
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt
import pinocchio as pin



dir_atual = os.path.dirname(__file__)
dir_anterior = os.path.dirname(dir_atual)

xml_path = os.path.join(dir_anterior, "universal_robots_ur5e_modificado/scene.xml")
mj_model = mj.MjModel.from_xml_path(xml_path)
mj_data = mj.MjData(mj_model)
mj.mj_kinematics(mj_model, mj_data)

xml_path_robot = os.path.join(dir_anterior, "universal_robots_ur5e_modificado/ur5e.xml")
pin_model, *_ = pin.buildModelsFromMJCF(xml_path_robot)
pin_data = pin_model.createData()

def referencia_pos_step(t):
    theta1 = 0.0
    theta3 = 0.0
    theta4 = 0.0
    theta6 = 0.0;
    if t < 2.0:
        theta2 = 0.0
        theta5 = 0.0
    else:
        theta2 = -0.4
        theta5 = 0.5
    return np.array([theta1, theta2, theta3, theta4, theta5, theta6])

def referencia_ac_step(t):
    return np.zeros(6)

def referencia_pos_senoidal(t):
    theta1 = 0.0
    theta3 = 0.0
    theta4 = 0.0
    theta6 = 0.0
    if t <= 4:
        theta2 = 0.2*np.sin(2*np.pi*t)
        theta5 = 0.4*np.sin(3*np.pi*t)
        theta = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    else:
        theta = referencia_pos_senoidal(4)

    return theta

def referencia_ac_senoidal(t):
    ac = np.zeros(6)
    if t <= 4:
        ac[1] = -((2*np.pi)**2) * 0.2*np.sin(2*np.pi*t)
        ac[4] = -((3*np.pi)**2) * 0.4*np.sin(3*np.pi*t)
    return ac


referencia_pos = referencia_pos_step
referencia_ac = referencia_ac_step

Kp = np.diag([300, 300, 300, 300, 300, 300])

def CalcularKd(M):
    kd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(6):
        kd[i] = np.sqrt(Kp[i][i]*M[i][i])
    return 2*np.diag(kd)


log_posicao_juntas = []
log_Kd = []
erro_juntas_anterior = np.zeros(6)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    while mj_data.time <= 8.0:

        mj.mj_kinematics(mj_model, mj_data)
        pin.crba(pin_model, pin_data, mj_data.qpos[:6])
        t = mj_data.time

        posicao_juntas_referencia = referencia_pos(t)
        erro_juntas = posicao_juntas_referencia - mj_data.qpos[:6]
        derivada_erro = (erro_juntas - erro_juntas_anterior) / mj_model.opt.timestep
        erro_juntas_anterior = erro_juntas

        mj.mj_crb(mj_model, mj_data)
        matriz_inercia = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, matriz_inercia, mj_data.qM)

        G = pin.computeGeneralizedGravity(pin_model, pin_data, posicao_juntas_referencia)


        Kd = CalcularKd(matriz_inercia)
        log_Kd.append(Kd)


        control_signal = matriz_inercia@referencia_ac(t) +  Kp @ erro_juntas + Kd @ derivada_erro + G

        mj_data.ctrl[:6] = control_signal

        log_posicao_juntas.append(np.copy(mj_data.qpos[:6]))
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()


intervalo = np.arange(0, mj_data.time, mj_model.opt.timestep)
trajetoria_referencia = [referencia_pos(ti) for ti in intervalo]


figure_1, axs1 = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    for j in range(2):
        junta = i + j*3
        axs1[i][j].plot(intervalo, [pos[junta] for pos in trajetoria_referencia])
        axs1[i][j].plot(intervalo, [pos[junta] for pos in log_posicao_juntas])
        axs1[i][j].set_title(f'Junta {junta + 1}')
        axs1[i][j].grid()

plt.savefig('Juntas.pdf')

figure_2, axs2 = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    for j in range(2):
        junta = i + j*3
        axs2[i][j].plot(intervalo, [kd[junta][junta] for kd in log_Kd])
        axs2[i][j].set_title(f'Junta {junta + 1}')
        axs2[i][j].grid()

plt.savefig('Kd.pdf')

plt.show()
