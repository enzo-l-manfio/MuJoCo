import os
import numpy as np
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt


dir_atual = os.path.dirname(__file__)
dir_anterior = os.path.dirname(dir_atual)

xml_path = os.path.join(dir_anterior, "universal_robots_ur5e_modificado/scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
mj.mj_kinematics(model, data)


def referencia(t):
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

Kp = np.diag([600, 600, 600, 600, 600, 600]) #N.m/rad
Kd = np.diag([40, 40, 40, 40, 40, 40]) #N.m.s/rad

log_posicao_juntas = []
erro_juntas_anterior = np.zeros(6)


with mujoco.viewer.launch_passive(model, data) as viewer:

    while data.time < 8.0:

        mj.mj_kinematics(model, data)
        t = data.time

        posicao_juntas_referencia = referencia(t)
        erro_juntas = posicao_juntas_referencia - data.qpos[:6]
        derivada_erro = (erro_juntas - erro_juntas_anterior) / model.opt.timestep
        erro_juntas_anterior = erro_juntas

        control_signal = Kp @ erro_juntas + Kd @ derivada_erro

        data.ctrl[:6] = control_signal

        log_posicao_juntas.append(np.copy(data.qpos[:6]))

        mujoco.mj_step(model, data)
        viewer.sync()


intervalo = np.arange(0, data.time, model.opt.timestep)
trajetoria_referencia = [referencia(ti) for ti in intervalo]

figure, axs = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    for j in range(2):
        junta = i + j*3
        axs[i][j].plot(intervalo, [pos[junta] for pos in trajetoria_referencia])
        axs[i][j].plot(intervalo, [pos[junta] for pos in log_posicao_juntas])
        axs[i][j].set_title(f'Junta {junta + 1}; Kp = {Kp[junta][junta]}; Kd = {Kd[junta][junta]}')
        axs[i][j].grid()

plt.show()


