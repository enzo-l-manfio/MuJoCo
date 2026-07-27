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


def referencia_step(t):

    theta = np.copy(pos_inicial)
    if t >= 2 :
        theta[0] -= 0.4
        theta[2] += 0.5

    return theta

def referencia_senoidal(t):
    theta = np.copy(pos_inicial)
    if t <= 2:
        theta[0] += 0.2*np.sin(2*np.pi * t)
        theta[2] += 0.4*np.sin(3*np.pi * t)
    return theta


referencia = referencia_senoidal


Kps = [10 * np.diag([1, 1, 1]),
       20 * np.diag([1, 1, 1]),
       30 * np.diag([1, 1, 1])]

def CalcularKd(M, Kp):
    kd = [0.0, 0.0, 0.0]
    for i in range(3):
        kd[i] = 2*np.sqrt(1000*Kp[i][i]*M[i][i])
    return np.diag(kd) /1000

for Kp in Kps:

    mj.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:3] = np.copy(pos_inicial)
    mj.mj_forward(mj_model, mj_data)

    erro_juntas_anterior = np.zeros(3)

    log_posicao_juntas = []
    log_kd = []

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        while mj_data.time <= 4.0:
            
            mj.mj_kinematics(mj_model, mj_data)
            t = mj_data.time

            posicao_juntas_referencia = referencia(t)

            erro_juntas = posicao_juntas_referencia - mj_data.qpos[:3]
            derivada_erro = (erro_juntas - erro_juntas_anterior) / mj_model.opt.timestep

            mj.mj_crb(mj_model, mj_data)
            matriz_inercia = np.zeros((mj_model.nv, mj_model.nv))
            mujoco.mj_fullM(mj_model, matriz_inercia, mj_data.qM)

            Kd = CalcularKd(matriz_inercia, Kp)
            log_kd.append(Kd)

            control_signal = Kp @ erro_juntas + Kd @ derivada_erro
            erro_juntas_anterior = erro_juntas

            mj_data.ctrl[:3] = control_signal

            log_posicao_juntas.append(np.copy(mj_data.qpos[:3]))
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
        

    t = np.linspace(0, mj_data.time, len(log_posicao_juntas))

    trajetoria_referencia = [referencia(ti) for ti in t]
    figure, axs = plt.subplots(3, 2, figsize=(10, 8))

    for i in range(3):

        axs[i][0].plot(t, [pos[i] for pos in trajetoria_referencia], label='referencia')
        axs[i][0].plot(t, [pos[i] for pos in log_posicao_juntas], label='real')
        if i == 2:
            axs[i][0].set_xlabel('Tempo (s)')
        axs[i][0].set_ylabel('ângulo (rad)')
        axs[i][0].set_title(f'Kp = {Kp[i][i]} Kd = Amortecimento Crítico')
        axs[i][0].legend('upper right')
        axs[i][0].grid()

        axs[i][1].plot(t, [kd[i][i] for kd in log_kd])
        if i == 2:
            axs[i][1].set_xlabel('Tempo (s)')
        axs[i][1].set_ylabel('Kd (kNm)')
        axs[i][1].set_title(f'Kd para junta {i + 1}')
        axs[i][1].legend('upper right')
        axs[i][1].grid()

    plt.savefig(f'Kp_{Kp[0][0]}.pdf')
    plt.show()