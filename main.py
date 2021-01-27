"""

Path tracking simulation with LQR steering control and PID speed control.

author FYZ 2021/1/21

"""
import scipy.linalg as la
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math
import numpy as np

Kp = 1.0  # speed proportional gain

# LQR parameter
Q = np.eye(4)
R = np.eye(1)

# parameters
dt = 0.01  # time tick[s]
L = 2.8  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(33.84)  # maximum steering angle[rad]
max_rate = 5.64  # degree
show_animation = False


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, f=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.f = f

def update(state, a, delta):

    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer
    if delta - state.f >= np.deg2rad(0.3525):
        state.f = state.f + np.deg2rad(0.3525)
    elif delta - state.f <= -np.deg2rad(0.3525):
        state.f = state.f - np.deg2rad(0.3525)
    else:
        state.f = delta

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(state.f) * dt
    state.v = state.v + a * dt
    return state


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 1
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn
    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = la.eig(A - B @ K)

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e, ind):
    ind, e = calc_nearest_index(state, cx, cy, cyaw, ck, ind)
    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    A = np.zeros((4, 4))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    # print(A)

    B = np.zeros((4, 1))
    B[3, 0] = v / L

    K, _, _ = dlqr(A, B, Q, R)

    x = np.zeros((4, 1))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    ff = math.atan2(L * k, 1)
    fb = pi_2_pi((-K @ x)[0, 0])

    delta = ff + fb

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw, ck, ind):
    # dx = [state.x - icx for icx in cx]
    # dy = [state.y - icy for icy in cy]
    #
    # d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    # mind = min(d)
    print('ind:', ind)
    dx0 = state.x - cx[ind]
    dy0 = state.y - cy[ind]
    d0 = math.sqrt(dx0 ** 2 + dy0 ** 2)
    d = d0
    if ind < len(cx) - 1:
        dx1 = state.x - cx[ind + 1]
        dy1 = state.y - cy[ind + 1]
        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        if ck[ind]*ck[ind+2] >= 0 and d0 > 0.1 and cx[ind] - state.x > 0:
            ind += 1
            d = d1
        elif ck[ind]*ck[ind+2] < 0 and d0 < 0.05:  # and cx[ind] - state.x > 0:
            ind += 2
            dx2 = state.x - cx[ind + 2]
            dy2 = state.y - cy[ind + 2]
            d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
            d = d2

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        d *= -1

    return ind, d


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.2
    stop_speed = 0.05

    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    time = 0.0
    ind = 0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    fi = [0.0]
    f = [0.0]
    e, e_th = 0.0, 0.0

    while T >= time:
        dl, target_ind, e, e_th = lqr_steering_control(
            state, cx, cy, cyaw, ck, e, e_th, ind)
        ind = target_ind
        ai = PIDControl(speed_profile[target_ind], state.v)
        if np.rad2deg(dl) - fi[-1] > max_rate:
            dl = np.deg2rad(fi[-1] + max_rate)
        elif np.rad2deg(dl) - fi[-1] < -max_rate:
            dl = np.deg2rad(fi[-1] - max_rate)
        fi.append(np.rad2deg(dl))
        state = update(state, ai, dl)
        f.append(state.f)
        if x[-1] < cx[-1]:  # abs(state.v) <= stop_speed:
            break
        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if abs(dx) <= 0.1:  # math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "ob", label="course")
            plt.plot(x, y, "-r", label="trajectory")
            print(target_ind)
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.01)
            plt.close()
            plt.subplot(211)
            plt.plot(fi)
            plt.subplot(212)
            plt.plot(f)
            plt.show()
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind < len(cx) and show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "ob", label="course")
            plt.plot(x, y, "-r", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.01)
    return t, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    speed_profile[-1] = 0.0

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
    route = loadmat('RouteA4.3L5+A4.3L5+L0.2.mat')
    route = route['Route']
    cx = route[0]
    cy = route[1]
    cyaw = route[2]
    ck = route[3]
    index_end = np.argwhere(cx == 0)[1][0]
    cx = cx[0:index_end]
    cy = cy[0:index_end]
    cyaw = cyaw[0:index_end]
    goal = [cx[-1], cy[-1]]

    target_speed = -1.0 / 3.6  # simulation parameter km/h -> m/s

    sp = calc_speed_profile(cx, cy, cyaw, target_speed)

    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    if True:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        # plt.plot(ax, ay, "xb", label="input")
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
