import numpy as np
import mujoco as mj
import os
import mujoco.viewer
import matplotlib.pyplot as plt
from time import perf_counter

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
xml_path = os.path.join(script_dir, "manipulador_atuadores_torque.xml")

mj_model = mj.MjModel.from_xml_path(xml_path)
mj_data = mj.MjData(mj_model)

mj_model.opt.timestep = 0.001

pos_inicial = np.array([0.0, np.pi/6, np.pi/6])

mj_data.qpos[:3] = np.copy(pos_inicial)

mj.mj_kinematics(mj_model, mj_data)

def referencia_step(t):

    theta = np.copy(pos_inicial)
    if t < 2 :
        theta[0] -= 0.4
        theta[2] += 0.5

    return theta

def referencia_senoidal(t):
    theta = np.copy(pos_inicial)
    if t <= 2:
        theta[0] += 0.2*np.sin(2*np.pi * t)
        theta[2] += 0.4*np.sin(3*np.pi * t)
    return theta

Kp = np.diag([30, 30, 30])
Kd = np.diag([0.15, 0.15, 0.15])


erro_juntas_anterior = np.zeros(3)

log_posicao_juntas = [pos_inicial]

n = 0

start=0
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    while mj_data.time <= 4.0:
        end = perf_counter()
        print(end-start)
        start = perf_counter()
        
        mj.mj_kinematics(mj_model, mj_data)
        t = mj_data.time

        posicao_juntas_referencia = referencia_step(t)

        erro_juntas = posicao_juntas_referencia - mj_data.qpos[:3]
        derivada_erro = (erro_juntas - erro_juntas_anterior) / mj_model.opt.timestep
        control_signal = Kp @ erro_juntas + Kd @ derivada_erro
        erro_juntas_anterior = erro_juntas

        mj_data.ctrl[:3] = control_signal

        log_posicao_juntas.append(np.copy(mj_data.qpos[:3]))
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        n+=1
        if n >10 :
            exit(1)
    



t = np.arange(0, mj_data.time, mj_model.opt.timestep)

trajetoria_referencia = [referencia_step(ti) for ti in t]
figure, axs = plt.subplots(3, 1, figsize=(10, 8))

for i in range(3):

    axs[i].plot(t, [pos[i] for pos in trajetoria_referencia], label='referencia')
    axs[i].plot(t, [pos[i] for pos in log_posicao_juntas], label='real')
    axs[i].set_xlabel('Tempo (s)')
    axs[i].set_ylabel('ângulo (rad)')
    axs[i].set_title(f'Kp = {Kp[i][i]} Kd = {Kd[i][i]} q0 = {pos_inicial[i]:.4f}')
    axs[i].legend('upper right')
    axs[i].grid()

plt.show()