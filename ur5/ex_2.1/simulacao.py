import os
import numpy as np
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt



dir_atual = os.path.dirname(__file__)
dir_anterior = os.path.dirname(dir_atual)

xml_path = os.path.join(dir_anterior, "universal_robots_ur5e_modificado/scene.xml")
mj_model = mj.MjModel.from_xml_path(xml_path)
mj_data = mj.MjData(mj_model)
mj.mj_kinematics(mj_model, mj_data)

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

def referencia_vel_step(t):
    return np.zeros(6)

def referencia_ac_step(t):
    return np.zeros(6)

def referencia_pos_senoidal(t):
    theta = np.zeros(6)
    if t <= 4:
        theta[1] = 0.2*np.sin(2*np.pi*t)
        theta[4] = 0.4*np.sin(3*np.pi*t)
    else:
        theta = referencia_pos_senoidal(4)

    return theta

def referencia_vel_senoidal(t):
    vel = np.zeros(6)
    if t <=4 :
        vel[1] = 0.4*np.pi*np.sin(2*np.pi*t)
        vel[4] = 1.2*np.pi*np.sin(3*np.pi*t)
    return vel

def referencia_ac_senoidal(t):
    ac = np.zeros(6)
    if t <= 4:
        ac[1] = -((2*np.pi)**2) * 0.2*np.sin(2*np.pi*t)
        ac[4] = -((3*np.pi)**2) * 0.4*np.sin(3*np.pi*t)
    return ac


referencia_pos = referencia_pos_senoidal
referencia_vel = referencia_vel_senoidal
referencia_ac = referencia_ac_senoidal

Kp = np.diag([300, 300, 300, 300, 300, 300])
Kd = np.diag([20, 20, 20, 20, 20, 20])


log_posicao_juntas = []
log_Kd = []
erro_juntas_anterior = np.zeros(6)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    while mj_data.time <= 8.0:

        mj.mj_kinematics(mj_model, mj_data)
        t = mj_data.time

        erro_juntas = referencia_pos(t) - mj_data.qpos[:6]
        erro_velocidade = referencia_vel(t) - mj_data.qvel[:6]

        mj.mj_crb(mj_model, mj_data)
        M = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, M, mj_data.qM)

        a = referencia_ac(t) + Kp@erro_juntas + Kd@erro_velocidade

        control_signal = M@a + mj_data.qfrc_bias

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


plt.show()
