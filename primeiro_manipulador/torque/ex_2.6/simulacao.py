import numpy as np
import mujoco as mj
import os
import mujoco.viewer
import matplotlib.pyplot as plt
import pinocchio as pin


script_dir = os.path.dirname(os.path.abspath(__file__))
xml_dir = os.path.dirname(script_dir)
xml_path = os.path.join(xml_dir, "manipulador_atuadores_torque.xml")

mj_model = mj.MjModel.from_xml_path(xml_path)
mj_data = mj.MjData(mj_model)

pin_model, *_ = pin.buildModelsFromMJCF(xml_path)
pin_data = pin_model.createData()

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

referencia_pos = referencia_pos_senoidal
referencia_ac = referencia_ac_senoidal
referencia_vel = referencia_vel_senoidal




mj.mj_resetData(mj_model, mj_data)
mj_data.qpos[:3] = np.copy(pos_inicial)
mj_data.qvel[:3] = referencia_vel(0)
mj.mj_forward(mj_model, mj_data)

log_posicao_juntas = []

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    while mj_data.time <= 4.0:
        mj.mj_kinematics(mj_model, mj_data)
        t = mj_data.time
        mj.mj_forward(mj_model, mj_data)

        mj_data.ctrl[:3] = 0.001* pin.rnea(pin_model, pin_data, referencia_pos(t), referencia_vel(t), referencia_ac(t))

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
    axs[i].set_title(f'Realimentação')
    axs[i].legend('upper right')
    axs[i].grid()

plt.savefig(f'senoidal.pdf')
plt.show()