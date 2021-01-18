from numpy import *
from math import *
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


Kp = 1  # speed proportional gain
dt = 0.01  # time tick[s]
L = 2.8  # wheel base[m]
Q = eye(4)
Q[0, 0] = 19.7
Q[1, 1] = 0.01
Q[2, 2] = 18.3
Q[3, 3] = 7.3
R = 1
max_steer = 33.85 * pi/180  # in rad
target_v = -2.0 / 3.6  # in m/s

# reference route
route = loadmat('RouteA4.3L5+A4.3L5+L0.2.mat')
route = route['Route']

cx = route[0]  # linspace(0, 200, 2000)
cy = route[1]  # zeros(len(cx))
# pd = zeros(len(cx))
# pdd = zeros(len(cx))
cyaw = route[2]  # zeros(len(cx))
ck = route[3]  # zeros(len(cx))
index_end = argwhere(cx == 0)[1][0]
cx = cx[0:index_end]
cy = cy[0:index_end]
cyaw = cyaw[0:index_end]
ck = ck[0:index_end]

pe = 0
pth_e = 0
i = 1
x = 0
y = 0
yaw = 0
v = 0
ind = 0
lat_error = []
yaw_error = []


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer
    state.x = state.x + state.v * cos(state.yaw) * dt
    state.y = state.y + state.v * sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * tan(delta) * dt
    state.v = state.v + a * dt

    return state


def PIDControl(target, current):
    a = Kp * (target - current)
    return a
#
# def pi_2_pi(angle):  # the unit of angle is in rad;
#     while (angle > pi):
#         angle = angle - 2.0 * pi
#
#     while (angle < -pi):
#         angle = angle + 2.0 * pi
#
#     return angle


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 500
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * la.pinv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.pinv(B.T * X * B + R) * (B.T * X * A)

    return K


def calc_nearest_index(state, cx, cy):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [abs(sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]

    error = min(d)

    ind = d.index(error)

    dy = cy[ind] - state.y
    if dy > 0:
        error = -error

    return ind, error


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e):
    ind, e = calc_nearest_index(state, cx, cy)

    k = ck[ind]
    v = state.v
    th_e = (state.yaw - cyaw[ind])

    A = mat(zeros((4, 4)))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    # print(A)

    B = mat(zeros((4, 1)))
    B[3, 0] = v / L

    K = dlqr(A, B, Q, R)
    print('K is', K)
    x =mat(zeros((4, 1)))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    ff = atan(L * k)
    fb = (-K * x)
    print(ff, fb)
    delta = 1*ff + 1 * fb
    print(delta)
    return delta, ind, e, th_e


# Initialize
model = tf.keras.models.load_model('dynamic_model_EP21.h5')
state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
trgt_spd = 2 * ones((20, 1))
shft_pos = 7 * ones((20, 1))
trgt_acc = zeros((20, 1))
steer_angle = zeros((20, 1))
brake_prs = zeros((20, 1))
veh_spd = zeros((20, 1))
veh_yaw = zeros((20, 1))
x_pos = []  # zeros(len(cx))
y_pos = []  # zeros(len(cx))

while ind < len(cx):
    delta, ind, e, th_e = lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e)
    pth_e = th_e
    pe = e
    lat_error.append(e)
    print('lateral error is ', e)
    v = state.v
    print("v is", v)
    # print('Index is ', ind)
    if abs(e) > 4:
       print('too far from reference!\n')
       break
    a = PIDControl(target_v, v)

    state = update(state, a, delta)
    x = state.x
    y = state.y
    x_pos.append(x)
    y_pos.append(y)
    if abs(y_pos[-1]) > abs(cy[-1]):
        break
plt.plot(cx, cy, "-b")

for i in range(len(x_pos)):
    plt.plot(x_pos[i], y_pos[i], ".r", markersize=1)

plt.grid(True)
plt.axis("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.show()

for i in range(len(lat_error)):
    plt.plot(lat_error, "-r")
plt.show()
