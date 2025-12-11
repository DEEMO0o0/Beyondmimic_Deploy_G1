import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

import onnxruntime

import numpy as np
import mujoco

xml_path = "./unitree_description/mjcf/g1_liao.xml"
# xml_path:  "/home/ym/Whole_body_tracking/unitree_description/g1_xml.xml"

# Total simulation time
simulation_duration = 300.0
# Simulation time step
simulation_dt = 0.002
# Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
control_decimation = 10
def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version).

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Component a: v * (2.0 * q_w^2 - 1.0)
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    
    # Component b: cross(q_vec, v) * q_w * 2.0
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    # Component c: q_vec * dot(q_vec, v) * 2.0
    # For efficient computation, handle different dimensionalities
    if q_vec.ndim == 2:
        # For 2D case: use matrix multiplication for better performance
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        # For general case: use Einstein summation
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c
import numpy as np

def matrix_to_quaternion_simple(matrix):
    """
    简化的矩阵转四元数实现
    """
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    
    trace = m00 + m11 + m22
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])
def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b,init_to_world):
    """
    与IsaacLab中subtract_frame_transforms完全相同的实现（一维版本）
    计算从坐标系A到坐标系B的相对变换
    
    参数:
        pos_a: 坐标系A的位置 (3,)
        quat_a: 坐标系A的四元数 (4,) [w, x, y, z]格式
        pos_b: 坐标系B的位置 (3,)
        quat_b: 坐标系B的四元数 (4,) [w, x, y, z]格式
        
    返回:
        rel_pos: B相对于A的位置 (3,)
        rel_quat: B相对于A的旋转四元数 (4,) [w, x, y, z]格式
    """
    # 计算相对位置: pos_B_to_A = R_A^T * (pos_B - pos_A)
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    
    # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    rel_quat = quaternion_multiply(matrix_to_quaternion_simple(init_to_world), quat_b)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), rel_quat)
    
    # 确保四元数归一化（与IsaacLab保持一致）
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    
    return rel_pos, rel_quat

def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])
def get_all_body_poses(d, m):
    """
    获取MuJoCo模型中所有连杆在世界坐标系下的位置和姿态
    
    参数:
        d: mujoco.MjData 对象
        m: mujoco.MjModel 对象
        
    返回:
        body_poses: 字典，键为连杆名称，值为包含位置、四元数、旋转矩阵等信息的字典
    """
    body_poses = {}
    
    # 遍历所有body（从1开始，跳过世界body，body_id=0）
    for body_id in range(1, m.nbody):
        # 获取连杆名称
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        if body_name:  # 确保名称不为空（有些body可能没有名称）
            # 获取世界坐标系下的位置和姿态
            position = d.body(body_id).xpos.copy()      # 位置 (3,)
            quaternion = d.body(body_id).xquat.copy()   # 四元数 (4,)
            rotation_matrix = d.body(body_id).xmat.reshape(3, 3).copy()  # 旋转矩阵 (3,3)
            
            body_poses[body_name] = {
                'body_id': body_id,
                'position': position,
                'quaternion': quaternion,
                'rotation_matrix': rotation_matrix,
                'xmat_flat': d.body(body_id).xmat.copy()  # 平坦化的旋转矩阵 (9,)
            }
    
    return body_poses

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# 与xml文件里定义的执行器顺序保持一致
joint_xml = [
        "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]



def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

if __name__ == "__main__":
    # get config file name from command line

    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", default=,type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = "/home/ym/Whole_body_tracking/configs/g1.yaml"

    # 测试时motion_file和policy_path两个文件路径都要改
    motion_file = "./motion.npz"
    motion =  np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    i = 0

    policy_path ="./policy.onnx"

    num_actions = 29
    # 观测向量 154 维 = 参考动作 58 + 躯干相对姿态 6 + 根角速度 3 + 关节位置(减去默认) 29 + 关节转速 29 + 上一动作 29
    num_obs = 154
    import onnx
    model = onnx.load(policy_path)
    # 把 ONNX 模型里自带的元数据配置一次性读出来，并按 XML 关节顺序重排
    for prop in model.metadata_props:
        # 得到网络输出的关节顺序列表
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        # 把元数据里的默认关节零位转化为float数组，再按XML关节顺序重排，后面 PD 和初始化就能直接用
        if prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        if prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
            # stiffness_array = np.array([])
            
        if prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])        
        
        if prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    action = np.zeros(num_actions, dtype=np.float32)
    # target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    # policy = torch.jit.load(policy_path)

    # 把onnx模型加载进内存，由 ONNX Runtime负责后端计算，返回的对象 policy 相当于一个“已编译好的函数”            
    policy = onnxruntime.InferenceSession(policy_path)
    # 所用模型只有1个输入张量，所以取第0维，返回其名称，就是个字符串
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    action_buffer = np.zeros((num_actions,), dtype=np.float32) # 存last_action
    # 把仿真初始状态设为默认关节位置，并设置躯干高度0.8m，防止初始落地碰撞
    timestep = 0
    motioninput = np.concatenate((motioninputpos[timestep,:],motioninputvel[timestep,:]), axis=0)
    # 这里应该是获取基座body在参考运动初始帧的位姿，索引顺序还要再确定下
    motionposcurrent = motionpos[timestep,9,:]
    motionquatcurrent = motionquat[timestep,9,:]
    # 获取默认关节零位，设置初始姿态
    target_dof_pos = joint_pos_array.copy()
    d.qpos[2] = 0.8
    d.qpos[7:] = target_dof_pos
    # target_dof_pos = joint_pos_array_seq
    # ---------- 顺时针旋转 90° (测试初始yaw不为0的情况)----------
    import math
    yaw_plus90 = math.radians(90)                 # +90°
    half_yaw = yaw_plus90 / 2.0
    d.qpos[3:7] = [math.cos(half_yaw), 0, 0, -math.sin(half_yaw)]  # [w,x,y,z]
    # ------------------------------------

    # 获取anchor body在mujoco中的body索引
    body_name = "torso_link"  # robot_ref_body_index=3 motion_ref_body_index=7
    # body_name = "pelvis"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
        
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        
        start = time.time()
        # 固定相机：世界系位置 + 看向世界系原点 + 世界系上方向
        viewer.cam.lookat[:] = [0, 0, 0.8]      # 注视点（躯干高度）
        viewer.cam.distance = 3.0               # 相机到注视点距离
        viewer.cam.elevation = -20              # 仰角（度）
        viewer.cam.azimuth = 90                 # 方位角（度）
        viewer.cam.fixedcamid = -1              # 必须=-1，表示用自由相机
        
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # 在前两个控制周期，把参考运动和仿真的yaw对齐
            # 得到参考->机器人的初始yaw对齐矩阵
            if timestep < 2:
                ref_motion_quat = motionquat[timestep,9,:]
                yaw_motion_quat = yaw_quat(ref_motion_quat)
                yaw_motion_matrix = np.zeros(9)
                mujoco.mju_quat2Mat(yaw_motion_matrix, yaw_motion_quat)
                yaw_motion_matrix = yaw_motion_matrix.reshape(3,3)
                
                robot_quat = d.xquat[body_id]
                yaw_robot_quat = yaw_quat(robot_quat)
                yaw_robot_matrix = np.zeros(9)
                mujoco.mju_quat2Mat(yaw_robot_matrix, yaw_robot_quat)
                yaw_robot_matrix = yaw_robot_matrix.reshape(3,3)
                init_to_world =  yaw_robot_matrix @ yaw_motion_matrix.T

            mujoco.mj_step(m, d)
            # 计算传入仿真的最终力矩tau，也是底层PD控制，500hz
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml
            d.ctrl[:] = tau
            counter += 1

            # 上层50hz策略控制回路
            if counter % control_decimation == 0:
                # Apply control signal here.
                # 取robot_anchor_pos/quat
                position = d.xpos[body_id]
                quaternion = d.xquat[body_id]
                # 取command
                if timestep >= motioninputpos.shape[0]:
                    timestep = motioninputpos.shape[0] - 1      # 停在最后一帧
                motioninput = np.concatenate((motioninputpos[timestep,:],motioninputvel[timestep,:]),axis=0)
                # anchor_pos/quat
                motionposcurrent = motionpos[timestep,9,:]
                motionquatcurrent = motionquat[timestep,9,:]
                # 计算机器人->参考的旋转误差(本体坐标系)，
                # 其中init_to_world把“参考动作世界系”水平转到“仿真机器人世界系”
                anchor_quat = subtract_frame_transforms_mujoco(position,quaternion,motionposcurrent,motionquatcurrent,init_to_world)[1]
                # 取旋转误差矩阵的前两列(和论文对应)
                anchor_ori = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
                anchor_ori = anchor_ori.reshape(-1,)
                # create observation
                # 手动拼观测向量，用 offset 指针一段段把数据塞进 obs 数组
                offset = 0
                obs[offset:offset + 58] = motioninput
                offset += 58
                obs[offset:offset + 6] = anchor_ori  
                offset += 6
                # 计算角速度(本体系)并塞进观测
                angvel = quat_rotate_inverse_np(d.qpos[3:7], d.qvel[3 : 6])
                obs[offset:offset + 3] = d.qvel[3 : 6]
                offset += 3
                # 取q_joint, v_joint塞入观测
                # 这里就肯定了，只有29个关节的位置信息，没有浮动基部分
                qpos_xml = d.qpos[7 : 7 + num_actions]  # joint positions
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                # q_joint取的其实是q_joint - q_default
                obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq  # joint positions
                offset += num_actions
                qvel_xml = d.qvel[6 : 6 + num_actions]  # joint positions
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                obs[offset:offset + num_actions] = qvel_seq  # joint velocities
                offset += num_actions   
                # 将last_action塞入观测
                obs[offset:offset + num_actions] = action_buffer
                # 把准备好的观测向量转化为[1，154]的张量(浅拷贝，只是映射)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # 策略推理，第一个参数指：只要名叫actions的输出张量
                # 第二个参数dict，包含观测和当前帧序号
                action = policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([timestep], dtype=np.float32).reshape(1,1)})[0]
                # 
                action = np.asarray(action).reshape(-1) # 摊平
                action_buffer = action.copy() # 存入a_last
                target_dof_pos = action * action_scale + joint_pos_array_seq # 套公式计算期望q(网络joint顺序)
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml]) # 按xml重排
                i += 1
                if i > 0:# 1000
                    timestep += 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
