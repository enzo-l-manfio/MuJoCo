import numpy as np
import mujoco as mj
import os
import mujoco.viewer
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_dir = os.path.dirname(script_dir)
xml_path = os.path.join(xml_dir, "manipulador_atuadores_torque.xml")

mj_model = mj.MjModel.from_xml_path(xml_path)
mj_data = mj.MjData(mj_model)


mj_model.opt.timestep = 0.0001

pos_inicial = np.array([0.0, np.pi/6, np.pi/6])

mj_data.qpos[:3] = np.copy(pos_inicial)


mj.mj_kinematics(mj_model, mj_data)


def referencia_pos_step(t):

    theta = np.copy(pos_inicial)
    if t >= 2 :
        theta[0] -= 0.4
        theta[2] += 0.5

    return theta

def referencia_ac_step(t):
    return np.zeros(3)

def referencia_vel_step(t):
    vel = np.zeros(3)
    return vel

def referencia_pos_senoidal(t):
    theta = np.copy(pos_inicial)
    if t <= 2:
        theta[0] += 0.2*np.sin(2*np.pi * t)
        theta[2] += 0.4*np.sin(3*np.pi * t)
    return theta

def referencia_vel_senoidal(t):
    vel = np.zeros(3)
    if t <=2 :
        vel[0] += 0.4*np.pi*np.cos(2*np.pi * t)
        vel[2] += 1.2*np.pi*np.cos(3*np.pi * t)
    return vel

def referencia_ac_senoidal(t):
    ac = np.zeros(3)
    if t <= 2:
        ac[0] = -((2 * np.pi) ** 2) * 0.2 * np.sin(2 * np.pi * t)
        ac[2] = -((3 * np.pi) ** 2) * 0.4 * np.sin(3 * np.pi * t)
    return ac

referencia_pos = referencia_pos_step
referencia_ac = referencia_ac_step
referencia_vel = referencia_vel_step

k_d = lambda k_p: 2 * np.sqrt(k_p)
Ks = [(10, k_d(10)),
      (20, k_d(20)),
      (30, k_d(30)),
      (100, k_d(100)),
      (200, k_d(200)),
      (300, k_d(300)),
      (1000, k_d(1000)),
      (2000, k_d(2000)),
      (3000, k_d(3000)),]



for kp, kd in Ks:

    mj.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:3] = np.copy(pos_inicial)
    mj_data.qvel[:3] = referencia_vel(0)
    mj.mj_forward(mj_model, mj_data)

    erro_juntas_anterior = np.zeros(3)

    log_posicao_juntas = []
    log_kd = []

    Kp = kp * np.diag([1, 1, 1])
    Kd = kd * np.diag([1, 1, 1])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        while mj_data.time <= 4.0:
            mj.mj_kinematics(mj_model, mj_data)
            t = mj_data.time
            mj.mj_forward(mj_model, mj_data)


            erro_juntas = referencia_pos(t) - mj_data.qpos[:3]

            erro_vel = referencia_vel(t) - mj_data.qvel 

            ac_referencia = referencia_ac(t)

            mj.mj_crb(mj_model, mj_data)
            matriz_inercia = np.zeros((mj_model.nv, mj_model.nv))
            mujoco.mj_fullM(mj_model, matriz_inercia, mj_data.qM)

            aceleracao_referencia = referencia_ac(t)

            a = ac_referencia + Kp@erro_juntas + Kd@erro_vel

            control_signal = matriz_inercia@a + mj_data.qfrc_bias

            mj_data.ctrl[:3] = 0.001* control_signal

            log_posicao_juntas.append(np.copy(mj_data.qpos[:3]))
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
        

    t = np.linspace(0, mj_data.time, len(log_posicao_juntas))

    trajetoria_referencia = [referencia_pos(ti) for ti in t]
    figure, axs = plt.subplots(3, 1, figsize=(10, 8))

    for i in range(3):

        axs[i].plot(t, [pos[i] for pos in trajetoria_referencia], label='referencia')
        axs[i].plot(t, [pos[i] for pos in log_posicao_juntas], label='real')
        if i == 2:
            axs[i].set_xlabel('Tempo (s)')
        axs[i].set_ylabel('ângulo (rad)')
        axs[i].set_title(f'Kp = {Kp[i][i]} Kd = Crítico')
        axs[i].legend('upper right')
        axs[i].grid()

    n_imagem = 6
    plt.savefig(f'Kp_{Kp[0][0]}.pdf')
    plt.show()