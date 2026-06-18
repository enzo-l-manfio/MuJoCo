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

def referencia_step(t):
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

def referencia_senoidal(t):
    theta1 = 0.0
    theta3 = 0.0
    theta4 = 0.0
    theta6 = 0.0
    if t <= 4:
        theta2 = 0.2*np.sin(2*np.pi*t)
        theta5 = 0.4*np.sin(3*np.pi*t)
        theta = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    else:
        theta = referencia_senoidal(4)

    return theta

Kp = np.diag([300, 300, 300, 300, 300, 300])

def CalcularKd(M):
    m = Kp@M
    return 2*np.diag([np.sqrt(m[i][i]) for i in range(6)])


log_posicao_juntas = []
erro_juntas_anterior = np.zeros(6)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    while mj_data.time <= 8.0:

        mj.mj_kinematics(mj_model, mj_data)
        pin.crba(pin_model, pin_data, mj_data.qpos[:6])
        t = mj_data.time

        posicao_juntas_referencia = referencia_step(t)
        erro_juntas = posicao_juntas_referencia - mj_data.qpos[:6]
        derivada_erro = (erro_juntas - erro_juntas_anterior) / mj_model.opt.timestep
        erro_juntas_anterior = erro_juntas


        Kd = CalcularKd(pin_data.M)
        control_signal = Kp @ erro_juntas + Kd @ derivada_erro

        mj_data.ctrl[:6] = control_signal

        log_posicao_juntas.append(np.copy(mj_data.qpos[:6]))
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()


intervalo = np.arange(0, mj_data.time, mj_model.opt.timestep)
trajetoria_referencia = [referencia_step(ti) for ti in intervalo]

figure, axs = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    for j in range(2):
        junta = i + j*3
        axs[i][j].plot(intervalo, [pos[junta] for pos in trajetoria_referencia])
        axs[i][j].plot(intervalo, [pos[junta] for pos in log_posicao_juntas])
        axs[i][j].set_title(f'Junta {junta + 1}')
        axs[i][j].grid()

plt.show()
