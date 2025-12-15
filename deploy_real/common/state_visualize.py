import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import transforms3d as tf3
import mujoco
import pandas as pd
import os
from openpyxl import load_workbook
from datetime import datetime
import sys


class RobotStateObserver:
    def __init__(self, robot_name, model=None, data=None):
        self.name = robot_name
        self.j_num = 6
        self.base_save_addr = os.path.join(sys.path[0], "sim2real_pic")
        # self.save_addr = '/home/aius/Reinforcement_Learning_Project/humanoid-gym/sim2real_pic/real_data'  # 图表保存地址
        # self.excel_path = '/sim2real_pic/2_13_推力训练/real_data.xlsx'

        # 创建带时间戳的子目录
        save_dir = create_timestamped_dir(self.base_save_addr)
        # 构建完整保存路径
        excel_filename = "real_data.xlsx"
        self.excel_path = os.path.join(save_dir, excel_filename)
        self.save_addr = os.path.join(save_dir, 'real_data')  # 图表保存地址

        # 初始化数据字典来存储状态历史记录
        self.data = {
            'root_vel': [],
            'root_pos':[],
            'root_height': [],
            'joint_pos': [],
            'joint_vel':[],
            'joint_torque': [],
            'lfoot_z': [],
            'rfoot_z': [],
            'l_grf': [],
            'r_grf': [],
            'obs':[],
            'action':[],
            'omega':[],
            'eu_ang':[],
            # 可以根据需要添加更多状态
        }
        self.limits = {
            'root_height': [],  # 高度范围
            'joint_pos': [],  # 关节位置范围，以弧度为单位
            'joint_vel': [],  # 关节速度范围
            'joint_torque': [[-360, -360, -120, -360, -120, -120, -360, -360, -120, -360, -120, -120],
                             [360, 360, 120, 360, 120, 120, 360, 360, 120, 360, 120, 120]]  # 关节扭矩范围
        }
        self.ref = {
            'root_height': 0.78,
            'root_vel': 0.5
        }

        self.select_robot()
        if model is not None:
            # self.robot_interface = RobotInterface(model, data, rfoot_body_name=self.rfoot_name,
            #                                       lfoot_body_name=self.lfoot_name)
            self.robot_interface = None
        # self.get_limit()

    def update(self):
        # 获取机器人状态并追加到数据字典中
        # self.data['root_vel'].append(self.robot_interface.get_root_body_vel())
        self.data['root_height'].append(self.robot_interface.get_object_xpos_by_name(self.root_name, 'OBJ_BODY')[2])
        self.data['root_pos'].append(self.robot_interface.get_object_xpos_by_name(self.root_name, 'OBJ_BODY').copy())
        self.data['root_vel'].append(self.robot_interface.get_qvel())
        self.data['joint_pos'].append(self.robot_interface.get_act_joint_positions())
        self.data['joint_vel'].append(self.robot_interface.get_act_joint_velocities())
        self.data['joint_torque'].append(self.robot_interface.get_act_joint_torques())
        self.data['lfoot_z'].append(self.robot_interface.get_object_xpos_by_name(self.lfoot_name, 'OBJ_BODY')[2])
        self.data['rfoot_z'].append(self.robot_interface.get_object_xpos_by_name(self.rfoot_name, 'OBJ_BODY')[2])
        self.data['r_grf'].append(self.robot_interface.get_rfoot_grf())
        self.data['l_grf'].append(self.robot_interface.get_lfoot_grf())

    def obs_action_check(self, obs, action, omega=None, eu_ang=None):
        self.data['obs'].append(obs)
        self.data['omega'].append(omega)
        self.data['eu_ang'].append(eu_ang)
        self.data['action'].append(action)

    def update_real_data(self, q, dq, ang_vel, euler_xyz, target_q, tau_est,velocity=None,position=None):
        # 获取机器人状态并追加到数据字典中
        self.data['joint_pos'].append(q)
        self.data['joint_vel'].append(dq)
        self.data['omega'].append(ang_vel)
        self.data['eu_ang'].append(euler_xyz)
        self.data['action'].append(target_q)
        self.data['joint_torque'].append(tau_est)
        if velocity is not None:
            self.data['root_vel'].append(velocity)
            self.data['root_pos'].append(position)  # 记录位置数据

    def get_limit(self):
        # 获取机器人的各种限制
        self.limits['joint_pos'] = self.robot_interface.get_act_joint_range()
        print(self.limits['joint_pos'])

    def plot_save_real_data(self):
        self.plot_joint_pos(save_name='_1_joint_pos')
        self.plot_joint_vel(save_name='_2_joint_vel')
        self.plot_joint_torque(save_name='_3_joint_torque')
        self.plot_base_ang(save_name='_4_base_ang')
        self.plot_base_ang_vel(save_name='_5_base_ang_vel')
        self.plot_action(save_name='_6_joint_target_q')
        self.plot_base_lin_vel(save_name='_7_base_lin_vel')
        self.plot_base_pos(save_name='_8_base_pos')
        # plt.show()

    def plot(self):
        self.plot_joint_pos(save_name='_1_joint_pos')
        self.plot_joint_vel(save_name='_2_joint_vel')
        self.plot_joint_torque(save_name='_3_joint_torque')
        # self.plot_action()
        self.plot_base_pos(save_name='_4_base_pos')
        self.plot_base_ang(save_name='_5_base_ang')
        self.plot_base_lin_vel(save_name='_6_base_lin_vel')
        self.plot_base_ang_vel(save_name='_7_base_ang_vel')

        self.plot_root_height(save_name='_8_base_height')
        self.plot_foot_z(save_name='_9_feet_height')
        self.plot_grf(save_name='_10_grf')
        # plt.show()

    def plot_action(self, save_name=None):
        action = self.data['action']
        # print(action)
        nested_out = np.array(transpose_list_of_lists(action))
        plt.figure()
        plt.suptitle("Target q")

        for idx, p in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(p, label=f"Joint {idx + 1}")  # 添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            # 使用 self.joint_names 作为标题（如果存在且索引在范围内）
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                plt.title(self.joint_names[idx])
            elif idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Action")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        joint_action_data = {'Time Step': range(len(action))}
        for idx, p in enumerate(nested_out):
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                joint_action_data[self.joint_names[idx]] = p
            else:
                joint_action_data[f'Joint_{idx}'] = p
        save_data_to_excel({'Joint Target q': joint_action_data}, self.excel_path)


    def plot_base_pos(self, save_name=None):
        # 假设 self.data['root_pos'] 是一个列表的列表，表示 x, y, z 位置数据
        base_pos = self.data['root_pos']
        base_pos_tr = np.array(transpose_list_of_lists(base_pos))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'x': 'blue',
            'y': 'green',
            'z': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=base_pos_tr,
            title="Base Position",
            ylabel="Distance (m)",
            colors=colors,
            labels=['x', 'y', 'z'],
            save_name=save_name
        )

        # 保存数据到 Excel
        base_pos_data = {
            'Time Step': range(len(base_pos)),
            'X Position': base_pos_tr[0],
            'Y Position': base_pos_tr[1],
            'Z Position': base_pos_tr[2]
        }
        save_data_to_excel({'Base Position': base_pos_data}, self.excel_path)

    def plot_base_ang(self, save_name=None):
        # 假设 self.data['eu_ang'] 是一个列表的列表，表示 roll, pitch, yaw 数据
        base_ang = self.data['eu_ang']
        base_ang_tr = np.array(transpose_list_of_lists(base_ang))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'roll': 'blue',
            'pitch': 'green',
            'yaw': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=base_ang_tr,
            title="Base Euler Angles",
            ylabel="Angle (radians)",
            colors=colors,
            labels=['roll', 'pitch', 'yaw'],
            save_name=save_name
        )
        # 保存数据到 Excel
        base_ang_data = {
            'Time Step': range(len(base_ang)),
            'Roll': base_ang_tr[0],
            'Pitch': base_ang_tr[1],
            'Yaw': base_ang_tr[2]
        }
        save_data_to_excel({'Base Euler Angles': base_ang_data}, self.excel_path)

    def plot_base_lin_vel(self, save_name=None):
        # 假设 self.data['root_vel'] 是一个列表的列表，表示 x, y, z 速度数据
        base_lin_vel = self.data['root_vel']
        base_lin_vel_tr = np.array(transpose_list_of_lists(base_lin_vel))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'x': 'blue',
            'y': 'green',
            'z': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=base_lin_vel_tr,
            title="Base Linear Velocities",
            ylabel="Velocity (m/s)",
            colors=colors,
            labels=['x', 'y', 'z'],
            save_name=save_name
        )

        # 保存数据到 Excel
        base_lin_vel_data = {
            'Time Step': range(len(base_lin_vel)),
            'X Linear Velocity': base_lin_vel_tr[0],
            'Y Linear Velocity': base_lin_vel_tr[1],
            'Z Linear Velocity': base_lin_vel_tr[2]
        }
        save_data_to_excel({'Base Linear Velocity': base_lin_vel_data}, self.excel_path)

    def plot_base_ang_vel(self, save_name=None):
        # 假设 self.data['omega'] 是一个列表的列表，表示 roll, pitch, yaw 角速度数据
        base_ang_vel = self.data['omega']
        base_ang_vel_tr = np.array(transpose_list_of_lists(base_ang_vel))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'roll': 'blue',
            'pitch': 'green',
            'yaw': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=base_ang_vel_tr,
            title="Base Angular Velocities",
            ylabel="Angular Velocity (rad/s)",
            colors=colors,
            labels=['roll', 'pitch', 'yaw'],
            save_name=save_name
        )

        # 保存数据到 Excel
        base_ang_vel_data = {
            'Time Step': range(len(base_ang_vel)),
            'X Angular Velocity': base_ang_vel_tr[0],
            'Y Angular Velocity': base_ang_vel_tr[1],
            'Z Angular Velocity': base_ang_vel_tr[2]
        }
        save_data_to_excel({'Base Angular Velocity': base_ang_vel_data}, self.excel_path)

    def plot_joint_pos(self, save_name=None):
        pos = self.data['joint_pos']
        nested_out = np.array(transpose_list_of_lists(pos))
        # if self.name == 'G1_whole':
        #     nested_out = g1_whole_rearrange_array(nested_out)
        plt.figure()
        plt.suptitle("Joint Positions")
        for idx, p in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(p, label=f"Joint {idx + 1}")  # 添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            # 使用 self.joint_names 作为标题（如果存在且索引在范围内）
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                plt.title(self.joint_names[idx])
            elif idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Position")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_pos_data = {'Time Step': range(len(pos))}
        for idx, p in enumerate(nested_out):
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                joint_pos_data[self.joint_names[idx]] = p
            else:
                joint_pos_data[f'Joint_{idx}'] = p
        save_data_to_excel({'Joint Positions': joint_pos_data}, self.excel_path)

    def plot_joint_vel(self, save_name=None):
        vel = self.data['joint_vel']
        nested_out = np.array(transpose_list_of_lists(vel))
        # if self.name == 'G1_whole':
        #     nested_out = g1_whole_rearrange_array(nested_out)
        plt.figure()
        plt.suptitle("Joint Velocities")
        for idx, p in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(p, label=f"Joint {idx + 1}")  # 添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            # 使用 self.joint_names 作为标题（如果存在且索引在范围内）
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                plt.title(self.joint_names[idx])
            elif idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Velocities")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_vel_data = {'Time Step': range(len(vel))}
        for idx, p in enumerate(nested_out):
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                joint_vel_data[self.joint_names[idx]] = p
            else:
                joint_vel_data[f'Joint_{idx}'] = p
        save_data_to_excel({'Joint Velocities': joint_vel_data}, self.excel_path)

    def plot_joint_torque(self, save_name=None):
        """
        Plot joint torque data.
        """
        torque = self.data['joint_torque']
        nested_out = np.array(transpose_list_of_lists(torque))
        # if self.name == 'G1_whole':
        #     nested_out = g1_whole_rearrange_array(nested_out)
        plt.figure()
        plt.suptitle("Joint Torque", fontsize=16)

        for idx, t in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(t, label=f"Joint {idx + 1}")  # 绘制曲线并添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            # 使用 self.joint_names 作为标题（如果存在且索引在范围内）
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                plt.title(self.joint_names[idx])
            elif idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Torque")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_torque_data = {'Time Step': range(len(torque))}
        for idx, t in enumerate(nested_out):
            if hasattr(self, 'joint_names') and idx < len(self.joint_names):
                joint_torque_data[self.joint_names[idx]] = t
            else:
                joint_torque_data[f'Joint_{idx}'] = t
        save_data_to_excel({'Joint Torques': joint_torque_data}, self.excel_path)

    def plot_root_height(self, save_name=None):
        ref = self.ref['root_height']
        h = self.data['root_height']

        plt.figure()
        plot_with_tracking(h, ref, "Root Height")

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        root_height_data = {
            'Time Step': range(len(h)),
            'Base Height': h,  # 修改为 Base Height
            'Ref Base Height': ref  # 修改为 Ref Base Height
        }
        save_data_to_excel({'Base Height': root_height_data}, self.excel_path)

    def plot_foot_z(self, save_name=None):
        # 创建一个新的图形
        plt.figure()

        l_z = np.array(self.data['lfoot_z'])
        r_z = np.array(self.data['rfoot_z'])

        # 获取时间轴，假设时间轴是数据的索引
        time_steps = np.arange(len(l_z))

        # 绘制左右足随时间的变化
        plt.plot(time_steps, l_z, label='Left Foot', color='blue')
        plt.plot(time_steps, r_z, label='Right Foot', color='red')

        # 设置图表的标签和标题
        plt.xlabel('Time Step')
        plt.ylabel('Position (X)')
        plt.title('Foot Z Position Over Time')

        # 添加图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # 保存图表
        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        foot_z_data = {
            'Time Step': time_steps,
            'Left Height': l_z,  # 修改为 Left Height
            'Right Height': r_z,  # 修改为 Right Height
            'Ref Feet Height': np.zeros_like(l_z)  # 如果需要参考值，可以添加 Ref Feet Height
        }
        save_data_to_excel({'Feet Height': foot_z_data}, self.excel_path)

    def plot_grf(self, save_name=None):
        r_grf = self.data['r_grf']  # 右脚地面反作用力数据
        l_grf = self.data['l_grf']  # 左脚地面反作用力数据

        plt.figure(figsize=(10, 8))

        plt.plot(r_grf, label='Right Foot GRF', color='r')  # 绘制右脚GRF，标签为“Right Foot GRF”且颜色设为红色
        plt.plot(l_grf, label='Left Foot GRF', color='b')  # 绘制左脚GRF，标签为“Left Foot GRF”且颜色设为蓝色

        plt.title('Ground Reaction Forces (GRF) of Left and Right Foot')  # 添加标题
        plt.xlabel('Time (frames)')  # 添加X轴标签，表示时间或帧
        plt.ylabel('Force (N)')  # 添加Y轴标签，表示力的单位（牛顿）
        plt.legend()  # 显示图例，以区分不同的数据线

        plt.grid(True)  # 显示网格，方便观察数据
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        grf_data = {
            'Time Step': range(len(r_grf)),
            'Left Contact Force': l_grf,  # 修改为 Left Contact Force
            'Right Contact Force': r_grf  # 修改为 Right Contact Force
        }
        save_data_to_excel({'Contact Force': grf_data}, self.excel_path)

    def plot_root_vel_track(self):
        ref = self.ref['root_vel']
        v = self.data['root_vel']
        v_smooth = exponential_moving_average(v, 0.1)

        plt.figure()
        plot_with_tracking(v_smooth, ref, "Root Vel")
        self.save_plot('root_vel')

    # def plot_root_height(self, save_name=None):
    #     ref = self.ref['root_height']
    #     h = self.data['root_height']
    #
    #     plt.figure()
    #     plot_with_tracking(h, ref, "Root Height")
    #
    #     if save_name is not None:
    #         self.save_plot(save_name)
    #
    # def plot_foot_z(self, save_name=None):
    #     # 创建一个新的图形
    #     plt.figure()
    #
    #     l_z = np.array(self.data['lfoot_z'])
    #     r_z = np.array(self.data['rfoot_z'])
    #
    #     # 获取时间轴，假设时间轴是数据的索引
    #     time_steps = np.arange(len(l_z))
    #
    #     # 绘制左右足随时间的变化
    #     plt.plot(time_steps, l_z, label='Left Foot', color='blue')
    #     plt.plot(time_steps, r_z, label='Right Foot', color='red')
    #
    #     # 设置图表的标签和标题
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Position (X)')
    #     plt.title('Foot Z Position Over Time')
    #
    #     # 添加图例
    #     plt.legend()
    #
    #     # 显示网格
    #     plt.grid(True)
    #
    #     # 保存图表
    #     if save_name is not None:
    #         self.save_plot(save_name)
    #
    # def plot_grf(self, save_name=None):
    #     r_grf = self.data['r_grf']  # 右脚地面反作用力数据
    #     l_grf = self.data['l_grf']  # 左脚地面反作用力数据
    #     # for i, grf in enumerate(r_grf):
    #     #     if grf >= 1:
    #     #         grf = 1
    #     #     else:
    #     #         grf = 0
    #     #     r_grf[i] = grf
    #     # for i, grf in enumerate(l_grf):
    #     #     if grf > 1:
    #     #         grf = 1.2
    #     #     else:
    #     #         grf = 0
    #     #     l_grf[i] = grf
    #
    #     plt.figure(figsize=(10, 8))
    #
    #     plt.plot(r_grf, label='Right Foot GRF', color='r')  # 绘制右脚GRF，标签为“Right Foot GRF”且颜色设为红色
    #     plt.plot(l_grf, label='Left Foot GRF', color='b')  # 绘制左脚GRF，标签为“Left Foot GRF”且颜色设为蓝色
    #
    #     plt.title('Ground Reaction Forces (GRF) of Left and Right Foot')  # 添加标题
    #     plt.xlabel('Time (frames)')  # 添加X轴标签，表示时间或帧
    #     plt.ylabel('Force (N)')  # 添加Y轴标签，表示力的单位（牛顿）
    #     plt.legend()  # 显示图例，以区分不同的数据线
    #
    #     plt.grid(True)  # 显示网格，方便观察数据
    #     plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
    #
    #     if save_name is not None:
    #         self.save_plot(save_name)

    def save_plot(self, plot_name):
        plt.gcf().set_size_inches(16, 9)  # 获取当前figure并设置大小
        plt.savefig(self.save_addr + f'{plot_name}.png', dpi=100)  # 保存当前figure
        plt.close()  # 关闭当前figure释放内存

    def gen_joint_title(self, i):
        if self.name == 'hit':
            if i == 0:
                title = 'Hip_yaw'
            elif i == 1:
                title = 'Hip_roll'
            elif i == 2:
                title = 'Hip_pitch'
            elif i == 3:
                title = 'Knee'
            elif i == 4:
                title = 'Ankle_pitch'
            elif i == 5:
                title = 'Ankle_roll'
        elif self.name == 'G1':
            if i == 0:
                title = 'Hip_pitch'
            elif i == 1:
                title = 'Hip_roll'
            elif i == 2:
                title = 'Hip_yaw'
            elif i == 3:
                title = 'Knee'
            elif i == 4:
                title = 'Ankle_pitch'
            elif i == 5:
                title = 'Ankle_roll'
        elif self.name == 'jvrc':
            if i == 0:
                title = 'Hip_pitch'
                i = 2
            elif i == 1:
                title = 'Hip_roll'
            elif i == 2:
                title = 'Hip_yaw'
                i = 0
            elif i == 3:
                title = 'Knee'
            elif i == 4:
                title = 'Ankle_roll'
                i = 5
            elif i == 5:
                title = 'Ankle_pitch'
                i = 4
        elif self.name == 'aius':
            if i == 0:
                title = 'Hip_roll'
                i = 1
            elif i == 1:
                title = 'Hip_yaw'
                i = 0
            elif i == 2:
                title = 'Hip_pitch'
            elif i == 3:
                title = 'Knee'
            elif i == 4:
                title = 'Ankle_pitch'
        elif self.name == 'G1_whole':
            if i == 0:
                title = 'Hip_pitch'
            elif i == 1:
                title = 'Hip_roll'
            elif i == 2:
                title = 'Hip_yaw'
            elif i == 3:
                title = 'Knee'
            elif i == 4:
                title = 'Ankle_pitch'
            elif i == 5:
                title = 'Ankle_roll'
            elif i == 6:
                title = 'Shoulder_pitch'
            elif i == 7:
                title = 'Shoulder_roll'
            elif i == 8:
                title = 'Shoulder_yaw'
            elif i == 9:
                title = 'Elbow_pitch'
        return title, i + 1

    def select_robot(self):
        if self.name == 'hit':
            self.root_name = 'pelvis'
            self.lfoot_name = 'ZT_6'
            self.rfoot_name = 'YT_6'
            self.j_num = 6
            self.limits['joint_torque'] = [[-50, -120, -200, -250, -160, -100, -50, -120, -200, -250, -160, -100],
                                           [50, 120, 200, 250, 160, 100, 50, 120, 200, 250, 160, 100]]  # 关节扭矩范围
        elif self.name == 'G1':
            self.root_name = 'pelvis'
            self.lfoot_name = 'left_ankle_roll_link'
            self.rfoot_name = 'right_ankle_roll_link'
            self.j_num = 6
            self.limits['joint_torque'] = [[-360, -360, -120, -360, -120, -120, -360, -360, -120, -360, -120, -120],
                                           [360, 360, 120, 360, 120, 120, 360, 360, 120, 360, 120, 120]]  # 关节扭矩范围
        elif self.name == 'jvrc':
            self.root_name = 'PELVIS_S'
            self.lfoot_name = 'L_ANKLE_P_S'
            self.rfoot_name = 'R_ANKLE_P_S'
            self.j_num = 6
        elif self.name == 'aius':
            self.root_name = 'base_link'
            self.lfoot_name = 'ZT_5'
            self.rfoot_name = 'YT_5'
            self.j_num = 5
            self.limits['joint_torque'] = [[-14, -14, -51, -14, -4.8],
                                           [14, 14, 51, 14, 4.8]]
            self.ref['root_height'] = 0.71

        elif self.name == 'G1_whole':
            self.root_name = 'pelvis'
            self.lfoot_name = 'left_ankle_roll_link'
            self.rfoot_name = 'right_ankle_roll_link'
            self.j_num = 10
            self.limits['joint_torque'] = [[-360, -360, -120, -360, -120, -120, -360, -360, -120, -360, -120, -120],
                                           [360, 360, 120, 360, 120, 120, 360, 360, 120, 360, 120, 120]]  # 关节扭矩范围
            self.joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
                "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
                "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"]

    def plot_base_info_template(self, data, title, ylabel, colors, labels, save_name=None):
        """
        通用绘图模板函数。

        参数:
            data: 形状为 (3, N) 的 NumPy 数组，表示要绘制的数据。
            title: 图像的主标题。
            ylabel: y 轴的标签。
            colors: 颜色字典，键为数据名称，值为颜色。
            labels: 数据名称列表，顺序与 data 的行对应。
            save_name: 保存图片的文件名。如果为 None，则不保存图片。
        """
        # 创建图像窗口，大小为 (10, 8)
        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        # 绘制子图
        for i in range(3):
            plt.subplot(3, 1, i + 1)  # 3 行 1 列，第 i+1 个子图
            plt.plot(data[i], label=labels[i], color=colors[labels[i]])
            plt.title(labels[i].capitalize())
            plt.xlabel('Time Step')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid()

        # 调整子图间距
        plt.tight_layout()

        # 如果提供了 save_name，则保存图片
        if save_name is not None:
            self.save_plot(save_name)


def exponential_moving_average(velocity_data, alpha):
    """
    对速度数据应用指数移动平均滤波器。

    :param velocity_data: 二维 numpy 数组，每一行代表一个时间点的速度向量。
    :param alpha: 平滑因子，介于0和1之间。
    :return: 平滑后的速度数据。
    """
    # 初始化EMA数据数组，与输入数据形状相同
    smoothed_data = np.zeros_like(velocity_data)
    # 初始化第一个值
    smoothed_data[0] = velocity_data[0]

    # 遍历速度数据，并应用EMA滤波器
    for t in range(1, len(velocity_data)):
        smoothed_data[t] = alpha * velocity_data[t] + (1 - alpha) * smoothed_data[t - 1]

    return smoothed_data


def transpose_list_of_lists(nested_list):
    # 使用zip()函数和列表推导式来进行转置
    return [list(item) for item in zip(*nested_list)]


def plot_data_with_limits(upper_limit, lower_limit, data_list):
    # 绘制数据
    plt.plot(data_list)

    # 画出上限和下限的线
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Upper Limit')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Lower Limit')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Data Plot with Limits")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 显示网格
    plt.grid(True)


def plot_with_limits(data, lower_limit, upper_limit, title):
    """
    绘制带有上下限的图表。
    :param data: 数据列表。
    :param lower_limit: 下界。
    :param upper_limit: 上界。
    :param title: 图表的标题。
    """
    x = list(range(len(data)))
    plt.plot(x, data, label='Data')
    plt.fill_between(x, lower_limit, upper_limit, color='grey', alpha=0.3, label='Limits')
    plt.title(title)
    plt.legend()
    plt.grid(True)


def plot_with_tracking(data, track_value, title):
    """
    绘制带跟踪值的图表。
    :param data: 数据列表。
    :param track_value: 被跟踪的值。
    :param title: 图表的标题。
    """
    x = list(range(len(data)))
    plt.plot(x, data, label='Data')
    plt.axhline(y=track_value, color='r', linestyle='--', label=f'Track Value: {track_value}')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)


def plot_smooth_transition_areas(data, label_name):
    # 找到局部极大值即平台结束点
    peaks, _ = find_peaks(data)

    # 找到局部极小值即平台开始点（通过找到极大值点在-y上）
    troughs, _ = find_peaks(-data)

    # 确保首尾都被考虑进来
    if troughs[0] > peaks[0]:
        troughs = np.insert(troughs, 0, 0)
    if troughs[-1] < peaks[-1]:
        troughs = np.append(troughs, len(data) - 1)

    # 根据找到的极值索引进行绘图
    x_values = np.arange(len(data))
    # plt.figure(figsize=(14, 7))
    plt.plot(x_values, data, label=label_name)

    # 画出每个平台之间的高度差异
    for start, end in zip(troughs, peaks):
        height_diff = data[end] - data[start]
        # plt.plot([start, end], [y_values[start], y_values[end]], 'ro-')  # 'ro-'表示红色实心点和实线
        if height_diff > 0.05:
            mid_point = (start + end) / 2
            plt.text(mid_point, (data[start] + data[end]) / 2, f'{height_diff:.3f}',
                     ha='center', va='center', color='blue', fontsize=10,
                     bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.2'))

    plt.xlabel('Time Step')
    plt.ylabel('X Value')
    plt.title('Foothold observation')
    plt.legend()
    plt.grid(True)

# def save_data_to_excel(data_dict, save_path):
#     """
#     将数据保存到 Excel 文件中，每个键值对保存为一个工作表。
#     参数:
#         data_dict (dict): 包含数据的字典，键为工作表名称，值为数据框或数据列表。
#         save_path (str): Excel 文件的保存路径。
#     """
#     with pd.ExcelWriter(save_path, engine='openpyxl', mode='a' if os.path.exists(save_path) else 'w', if_sheet_exists='replace') as writer:
#         for sheet_name, data in data_dict.items():
#             # 如果数据是列表或字典，转换为数据框
#             if isinstance(data, (list, dict)):
#                 data = pd.DataFrame(data)
#             # 将数据保存到 Excel 的工作表
#             data.to_excel(writer, sheet_name=sheet_name, index=False)
#     print(f"{sheet_name}数据已成功保存到 {save_path}")
# def save_data_to_excel(data_dict, save_path):
#     """
#     将数据保存到 Excel 文件中，每个键值对保存为一个工作表。
#     参数:
#         data_dict (dict): 包含数据的字典，键为工作表名称，值为数据框或数据列表。
#         save_path (str): Excel 文件的保存路径。
#     """
#     try:
#         # 获取目标目录
#         save_dir = os.path.dirname(save_path)
#
#         # 检查目录是否存在，如果不存在则创建
#         # if not os.path.exists(save_dir):
#         #     os.makedirs(save_dir)
#         #     print(f"目录 {save_dir} 不存在，已创建。")
#
#         # 检查文件是否存在
#         if os.path.exists(save_path):
#             # 如果文件存在，尝试加载现有文件
#             try:
#                 book = load_workbook(save_path)
#                 mode = 'a'  # 追加模式
#             except Exception as e:
#                 print(f"文件 {save_path} 已损坏，将重新创建文件。错误信息: {e}")
#                 os.remove(save_path)  # 删除损坏的文件
#                 mode = 'w'  # 写入模式
#         else:
#             mode = 'w'  # 写入模式
#
#         # 使用 ExcelWriter 保存数据
#         if mode == 'a':
#             # 追加模式，使用 if_sheet_exists='replace'
#             with pd.ExcelWriter(save_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
#                 for sheet_name, data in data_dict.items():
#                     # 如果数据是列表或字典，转换为数据框
#                     if isinstance(data, (list, dict)):
#                         data = pd.DataFrame(data)
#                     # 将数据保存到 Excel 的工作表
#                     data.to_excel(writer, sheet_name=sheet_name, index=False)
#                 print(f"{sheet_name} 数据已成功保存到 {save_path}")
#         else:
#             # 写入模式，不使用 if_sheet_exists
#             with pd.ExcelWriter(save_path, engine='openpyxl', mode=mode) as writer:
#                 for sheet_name, data in data_dict.items():
#                     # 如果数据是列表或字典，转换为数据框
#                     if isinstance(data, (list, dict)):
#                         data = pd.DataFrame(data)
#                     # 将数据保存到 Excel 的工作表
#                     data.to_excel(writer, sheet_name=sheet_name, index=False)
#                 print(f"{sheet_name} 数据已成功保存到 {save_path}")
#
#     except Exception as e:
#         print(f"保存数据到 Excel 文件时发生错误: {e}")

def save_data_to_excel(data_dict, save_path):
    """增强版Excel保存函数"""
    try:
        # 处理目录创建
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # 文件存在性检查
        file_exists = os.path.exists(save_path)
        mode = 'a' if file_exists else 'w'

        # 使用上下文管理器处理Excel写入
        with pd.ExcelWriter(
                save_path,
                engine='openpyxl',
                mode=mode,
                if_sheet_exists='replace' if file_exists else None
        ) as writer:
            for sheet_name, data in data_dict.items():
                # 自动转换数据类型
                df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
                df.to_excel(writer, sheet_name=str(sheet_name)[:30], index=False)  # 截断超过30字符的表名
            print(f"数据成功保存至：{os.path.abspath(save_path)}")

    except PermissionError:
        print(f"错误：文件 {save_path} 被其他程序占用")
    except Exception as e:
        print(f"保存失败，错误类型：{type(e).__name__}，详细信息：{str(e)}")


def create_timestamped_dir(base_dir):
    """在指定基础目录下创建时间戳子目录"""
    # 生成时间戳格式：YYYY-MM-DD-HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 拼接完整路径
    full_path = os.path.join(base_dir, timestamp)

    # 创建目录（自动创建父目录）
    os.makedirs(full_path, exist_ok=True)
    return full_path


def g1_whole_rearrange_array(arr):
    """
    调整数组的顺序，按照 [0:6], [12:16], [6:12], [16:20] 进行重组。

    参数:
    arr: 输入的 NumPy 数组，长度为 20。

    返回:
    rearranged_arr: 重组后的 NumPy 数组。
    """
    # 检查输入数组的长度是否为 20
    if len(arr) != 20:
        raise ValueError("输入数组的长度必须为 20")

    # 按照 [0:6], [12:16], [6:12], [16:20] 的顺序重组
    rearranged_arr = np.concatenate([arr[0:6], arr[12:16], arr[6:12], arr[16:20]])

    return rearranged_arr

# # 使用方法
# robot_interface = MockRobotInterface()  # 这里假设你有一个实际的机器人接口
# observer = RobotStateObserver(robot_interface)
#
# # 在仿真循环中更新并在适当时候绘制图形
# for _ in range(100):  # 假设有100个仿真步骤
#     observer.update()
#
# # 绘制指定变量的图形
# observer.plot(['root_vel', 'joint_pos'])
#
# # 将指定图形保存到磁盘
# observer.save_plots(['root_vel', 'joint_pos'], 'robot_state')
