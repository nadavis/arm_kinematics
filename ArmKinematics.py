import yaml
import numpy as np
import matplotlib.pyplot as plt
import math
import time

class ArmKinematics():
    def __init__(self, config):
        a = config['kinematics']['link_size']
        theta0 = config['kinematics']['theta0']
        self.theta = [0, 0, 0, 0, 0]
        self.theta0 = theta0
        self.a = a
        self.theta_limit = 90
        self.num_of_theta = 3
        self.num_of_dof = 5
        self.z_limit = config['kinematics']['z_limit']
        # self.d_h_table = np.array([[np.deg2rad(theta0[0]), np.deg2rad(90), 0, a[0]],
        #                       [np.deg2rad(theta0[1]+90), np.deg2rad(0), a[1], 0],
        #                       [np.deg2rad(theta0[2]), np.deg2rad(0), a[2], 0]])
        self.d_h_table = np.array([[np.deg2rad(theta0[0]), np.deg2rad(90), 0, a[0]],
                                   [np.deg2rad(theta0[1]+90), np.deg2rad(0), a[1], 0],
                                   [np.deg2rad(theta0[2] + 90), np.deg2rad(90), a[3], 0],
                                   [np.deg2rad(theta0[3]), np.deg2rad(-90), a[4], a[2] + a[5]],
                                   [np.deg2rad(theta0[4] - 90), np.deg2rad(0), 0, 0],
                                   [np.deg2rad(0), np.deg2rad(0), a[6], 0]])


    def rot_tran_matrix(self, theta, alpha, r, d):
        return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha),
                                np.sin(theta) * np.sin(alpha),
                                r * np.cos(theta)],
                               [np.sin(theta), np.cos(theta) * np.cos(alpha),
                                -np.cos(theta) * np.sin(alpha),
                                r * np.sin(theta)],
                               [0, np.sin(alpha), np.cos(alpha), d],
                               [0, 0, 0, 1]])
    def inv_kinematics(self, pos):
        offset = self.a[6]
        a0 = self.a[0]
        a1 = self.a[1]
        a2 = self.a[2] + self.a[5]

        t0 = math.pi + np.arctan2(pos[1], pos[0])
        y = np.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
        x = pos[2] - a0 + offset
        sign_x = x
        sign_y = y
        x = np.abs(x)
        k0 = (x * x + y * y - a1 * a1 - a2 * a2) / (2 * a1 * a2)
        if(k0>1):
            if (k0 < 1.00001):
                k0 = 1
            else:
                print('NO RESULTS')
                return [], []

        t2 = np.arccos(k0)
        k1 = a2*np.sin(t2)
        k2 = a1 + a2 * np.cos(t2)
        t1 = np.arctan2(y, x) - np.arctan2(k1, k2)

        t2_ = np.arccos(k0)
        k1_ = a2 * np.sin(t2_)
        k2_ = a1 + a2 * np.cos(t2_)
        t1_ = np.arctan2(y, x) + np.arctan2(k1_, k2_)

        theta1 = np.round([np.rad2deg(t0), np.rad2deg(t1), np.rad2deg(t2)])
        theta2 = np.round([theta1[0], np.rad2deg(t1_), -np.rad2deg(t2_)])

        if(sign_x<0):
            theta1 = np.round([np.rad2deg(t0), 180-np.rad2deg(t1), -np.rad2deg(t2)])
            theta2 = np.round([theta1[0], 180-np.rad2deg(t1_), np.rad2deg(t2_)])

        theta1 = np.append(theta1, 0)
        theta1 = np.append(theta1, 180 - theta1[1] - theta1[2])
        theta2 = np.append(theta2, 0)
        theta2 = np.append(theta2, 180 - theta2[1] - theta2[2])
        return theta1, theta2

    def run_forward(self, theta):
        H = []
        T = np.eye(4)
        if(len(self.d_h_table)!= len(theta)):
            theta = np.append(theta, 0)

        for i in range(0, len(self.d_h_table)):
            res = self.rot_tran_matrix(self.d_h_table[i, 0]+np.deg2rad(theta[i]), self.d_h_table[i, 1], self.d_h_table[i, 2], self.d_h_table[i, 3])
            T = T @ res
            H.append(res)

        return np.round(H, 5), np.round(T, 5)

    def check_collision(self, T):
        if(T[2,3]< self.z_limit):
            print('Collision: ', T)
            return True
        return False

    def set_theta_by_motor_ind(self, theta, motor_ind):
        self.theta[motor_ind] = int(theta)

    def check_collision_by_theta(self, theta=[]):
        if(len(theta)==0):
            theta = self.theta
        H, T = self.run_forward(theta)
        return self.check_collision(T)

    def show_kinematics(self, H, T, fig=0, ax=0, c='g', linewidth=2, x0=0, y0=0, z0=0):
        ma = sum(self.a)
        x = []
        y = []
        z = []
        coords = []
        v = [0, 0, 0, 1]
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
        T_ = np.eye(4)
        for h in H:
            T_ = T_ @ h
            res = T_ @ v
            x.append(res[0])
            y.append(res[1])
            z.append(res[2])
            coords.append(res)

        res = T @ v
        x_ = res[0]
        y_ = res[1]
        z_ = res[2]
        if fig ==0:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot(x0, y0, z0, c='m', marker='o')
        ax.plot(x, y, z, c='r', marker='o')
        ax.plot(x, y, z, c=c, linewidth=linewidth)
        ax.plot(x_, y_, z_, c='b', marker='o')
        ax.set_xlim(-ma, ma)
        ax.set_ylim(-ma, ma)
        ax.set_zlim(0, ma)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # plt.draw()
        # plt.pause(1)
        # plt.show()
        return fig, ax

if __name__ == "__main__":
    pp = []
    for p in range(0,360, 1):
        pp.append(np.rad2deg(np.arctan(p)))

    with open("config.yml", 'r') as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)
    show = True
    k = ArmKinematics(config)
    err1 = []
    err2 = []
    errp1 = []
    errp2 = []
    for t in range(0, 100, 10):
        theta0 = np.random.randint(-90, 90, size=len(k.theta0))
        theta0[3] = 0
        theta0[4] = 180 - theta0[1] - theta0[2]
        # theta0 = [t, 0, 90]
        H0, T0 = k.run_forward(theta0)
        pos0 = T0[:3, 3]

        # fig, ax = k.show_kinematics(H0, T0, linewidth=4)
        # plt.draw()
        # plt.pause(1)

        # pos0_ = pos0.copy()
        # pos0_[2] = pos0[2] + k.a[5]
        theta1, theta2 = k.inv_kinematics(pos0)
        if(len(theta1)==0 or len(theta2)==0):
            print('No results')
            continue

        H1, T1 = k.run_forward(theta1)
        pos1 = T1[:3, 3]

        H2, T2 = k.run_forward(theta2)
        pos2 = T2[:3, 3]

        d1 = np.subtract(theta0, theta1)
        e1 = np.sqrt(np.mean(d1 * d1))
        err1.append(e1)

        d2 = np.subtract(theta0, theta2)
        e2 = np.sqrt(np.mean(d2 * d2))
        err2.append(e2)

        e = np.min([e1, e2])

        dp1 = np.subtract(pos0, pos1)
        ep1 = np.sqrt(np.mean(dp1 * dp1))
        errp1.append(ep1)

        dp2 = np.subtract(pos0, pos2)
        ep2 = np.sqrt(np.mean(dp2 * dp2))
        errp2.append(ep2)
        ep = np.min([ep1,ep2])
        # fig, ax = k.show_kinematics(H0, T0, linewidth=4)
        # plt.draw()
        # plt.pause(1)
        if(ep>2):
            print('pos', pos0, pos1, pos2)
            print('theta', theta0, theta1, theta2)
            print('err theta', e, e1, e2)
            print('err pos', ep, ep1, ep2)
            print('-------------------------------------')
            fig, ax = k.show_kinematics(H0, T0, linewidth=4)
            fig, ax = k.show_kinematics(H1, T1, fig, ax, 'y', linewidth=2)
            fig, ax = k.show_kinematics(H2, T2, fig, ax, 'y', linewidth=2)
            plt.draw()
            plt.pause(1)
            # plt.show()
        # time.sleep(10)
    print('theta', np.max(err1), np.max(err2))
    print('pos', np.max(errp1), np.max(errp2))

