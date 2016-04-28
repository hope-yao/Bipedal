
def three_linked_chain(state, t):
    q1 = state[0]
    q1d = state[1]
    q2 = state[2]
    q2d = state[3]
    q3 = state[4]
    q3d = state[5]

    H = np.zeros((3,3))
    H[0,0] = ms*a1**2 + mt*(ls+a2)**2 + (mh+ms+mt)*l**2
    H[0,1] = -(mt*b2+ms*lt) * l * cos(q2-q1)
    H[0,2] = -ms*b1*l * cos(q3-q1)
    H[1,0] = H[0,1]
    H[1,1] = mt*b2**2 + ms*lt**2
    H[1,2] = ms*lt*b1 * cos(q3-q2)
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    H[2,2] = ms*b1**2

    B = np.zeros((3,3))
    h122 = -(mt*b2+ms*lt) * l * sin(q1-q2)
    h133 = -ms*b1*l * sin(q1-q3)
    h211 = -h122
    h233 = ms*lt*b1 * sin(q3-q2)
    h311 = -h133
    h322 = -h233
    B[0,0] = 0
    B[0,1] = h122 * q2d
    B[0,2] = h133 * q3d
    B[1,0] = h211 * q1d
    B[1,1] = 0
    B[1,2] = h233 * q3d
    B[2,0] = h311 * q1d
    B[2,1] = h322 * q2d
    B[2,2] = 0

    G = np.zeros(3)
    G[0] = -(ms*a1 + mt*(ls+a2) + (mh+ms+mt)*l) * g*sin(q1)
    G[1] = (mt*b2 + ms*lt) * g*sin(q2)
    G[2] = ms*b1*g * sin(q3)

    qd = [q1d,q2d,q3d]
    rhs = -np.dot(B,qd) - G
    sol = np.linalg.solve(H,rhs)
    xd = np.zeros_like(state)
    xd[0] = q1d
    xd[1] = sol[0]
    xd[2] = q2d
    xd[3] = sol[1]
    xd[4] = q3d
    xd[5] = sol[2]

    return xd



def two_linked_chain(state, t):
    q1 = state[0]
    q1d = state[1]
    q2 = state[2]
    q2d = state[3]

    H = np.zeros((2,2))
    H[0,0] = ms*a1**2 + mt*(ls+a2)**2 + (mh+ms+mt)*l**2
    H[0,1] = -(mt*b2 + ms*(lt+b1))*l *cos(q2-q1)
    H[1,0] = H[0,1]
    H[1,1] = mt*b2**2 + ms*(lt+b1)**2

    h = -(mt*b2 + ms*(lt+b1))*l * sin(q1-q2)
    B = np.zeros((2,2))
    B[0,0] = 0
    B[0,1] = h*q2d
    B[1,0] = 0
    B[1,1] = -h*q1d

    G = np.zeros(2)
    G[0] = -(ms*a1 + mt*(ls+a2) + (mh+mt+ms)*l) *g*sin(q1)
    G[1] = (mt*b2 + ms*(lt+b1)) * g*sin(q2)

    qd = [q1d,q2d]
    rhs = -np.dot(B,qd) - G
    sol = np.linalg.solve(H,rhs)

    xd = np.zeros_like(state)
    xd[0] = q1d
    xd[1] = sol[0]
    xd[2] = q2d
    xd[3] = sol[1]
    xd[4] = q2d
    xd[5] = sol[1]

    return xd


def knee_strike(state):
    q1 = state[0]
    q2 = state[2]
    q3 = state[4]

    alpha = cos(q1-q2)
    beta = cos(q1-q3)
    gamma = cos(q2-q3)

    Q_r = np.zeros((2,3))
    Q_r[0,0] = -(ms*lt+mt*b2)*l*cos(alpha) - ms*b1*l*cos(beta) + (mt+ms+mh)*l**2 + ms*a1**2 + mt*(ls+a2)**2
    Q_r[0,1] = -(ms*lt+mt*b2)*l*cos(alpha) + ms*b1*lt*cos(gamma) + mt*b2**2 + ms*lt**2
    Q_r[0,2] = -ms*b1*l*cos(beta) + ms*b1*lt*cos(gamma) + ms*b1**2
    Q_r[1,0] = -(ms*lt+mt*b2)*l*cos(alpha) - ms*b1*l*cos(beta)
    Q_r[1,1] = ms*b1*lt*cos(gamma) + ms*lt**2 + mt*b2**2
    Q_r[1,2] = ms*b1*lt*cos(gamma) + ms*b1**2

    Q_l = np.zeros((2,2))
    Q_l[1,1] = ms*(lt+b1)**2 + mt*b2**2
    Q_l[1,0] = -(ms*(b1+lt) + mt*b2) * l*cos(alpha)
    Q_l[0,1] = Q_l[1,0] + ms*(lt+b1)**2 + mt*b2**2
    Q_l[0,0] = Q_l[1,0] + mt*(ls+a2)**2 + (mh+mt+ms)*l**2 + ms*a1**2

    q_r = [state[1],state[3],state[5]]
    rhs = np.dot(Q_r,q_r)
    sol = np.linalg.solve(Q_l,rhs)

    xd = np.zeros_like(state)
    xd[0] = state[0]
    xd[1] = sol[0]
    xd[2] = state[2]
    xd[3] = sol[1]
    xd[4] = state[4]
    xd[5] = sol[1]  # q3d = q2d, thigh and shank binded

    return xd

def heel_strike(state):
    q1 = state[0]
    q2 = state[2]

    alpha = cos(q1-q2)

    Q_r = np.zeros((2,2))
    Q_r[1,1] = 0
    Q_r[1,0] = -ms*a1*(lt+b1) + mt*b2*(ls+a2)
    Q_r[0,1] = Q_r[1,0]
    Q_r[0,0] = Q_r[0,1] + (mh*l + 2*mt*(a2+ls) + ms*a1) * l*cos(alpha)

    Q_l = np.zeros((2,2))
    Q_l[1,1] = ms*(lt+b1)**2 + mt*b2**2
    Q_l[1,0] = -(ms*(b1+lt) + mt*b2) * l*cos(alpha)
    Q_l[0,1] = Q_l[1,0] + ms*(b1+lt)**2 + mt*b2**2
    Q_l[0,0] = Q_l[1,0] + (ms+mt+mh)*l**2 + ms*a1**2 + mt*(a2+ls)**2

    q_r = [state[1],state[3]]
    rhs = np.dot(Q_r,q_r)
    sol = np.linalg.solve(Q_l,rhs)

    xd = np.zeros_like(state)
    xd[0] = state[0]
    xd[1] = sol[0]
    xd[2] = state[2]
    xd[3] = sol[1]
    xd[4] = state[4]
    xd[5] = sol[1]  # q3d = q2d, thigh and shank binded

    return xd


def step_cycle(state,pos_sf,_time):
    print('>>>>>>>>>>>>>>>>>>>>>>>initial position<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(np.degrees([state[0],state[2],state[4]]))

    ttcosd = []
    cnt = 0
    num_time = 2
    dt = _time / num_time
    chain_num=3 # starts with three-chain state
    while cnt<300: # usually it takes less than three seconds for a step
        print('cnt: ',cnt)
        t = np.arange(0.0, _time, dt)
        if cnt==0:
            tmp = integrate.odeint(three_linked_chain, state, t)
            state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
            print('three linked chain, time: %s <<<' % (cnt*_time))
            print(np.degrees([state[-1,0],state[-1,2],state[-1,4]]))
            cnt += 1
            continue
        # knee strike
        if  (np.max(-(state[-num_time:,2]))>np.radians(_alpha/2-_gamma)*0.9)\
        and (chain_num==3) and (np.min(np.abs(state[-num_time:,2]-state[-num_time:,4]))<np.radians(1)):
            tmp = knee_strike(state[-1])
            state = np.insert(state, [len(state)], tmp, axis=0)
            chain_num = 2
            print('===============================================knee strike, time: %s' % (cnt*_time))
            print(np.degrees([state[-1,0],state[-1,2],state[-1,4]]))
            continue
        # heel strike
        if (chain_num==2) and cosd<(3): # in degree
            tmp = heel_strike(state[-1])
            state = np.insert(state, [len(state)], tmp, axis=0)
            chain_num = 3
            print('===============================================heel strike, time: %s' % (cnt*_time))
            print(np.degrees([state[-1,0],state[-1,2],state[-1,4]]))
            break
        # three linked chain
        if (chain_num==3):
            tmp = integrate.odeint(three_linked_chain, state[-1], t)
            state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
            print('three linked chain, time: %s <<<' % (cnt*_time))
            print(np.degrees([state[-1,0],state[-1,2],state[-1,4]]))
            chain_num = 3
            cnt += 1
        # two linked chain
        if (chain_num==2):
            tmp = integrate.odeint(two_linked_chain, state[-1], t)
            state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
            print('two linked chain, time: %s <<<' % (cnt*_time))
            print(np.degrees([state[-1,0],state[-1,2],state[-1,4]]))
            chain_num = 2
            cnt += 1

        q1 = tmp[:,0]
        q2 = tmp[:,2]
        q3 = tmp[:,4]
        x_h_tmp = -l*sin(q1)
        y_h_tmp = l*cos(q1)
        x_nsk_tmp = x_h_tmp - lt*sin(q2)
        y_nsk_tmp = y_h_tmp - lt*cos(q2)
        x_nsf_tmp = x_nsk_tmp - ls*sin(q3)
        y_nsf_tmp = y_nsk_tmp - ls*cos(q3)

        v1 = [(x_nsf_tmp[i],y_nsf_tmp[i]) for i in range(num_time)]
        v2 = [cos(np.radians(_gamma)),-sin(np.radians(_gamma))]
        ab = np.dot(v1, v2)
        print(ab)
        cosd = [np.degrees(math.acos( 0.999* ab[i] / np.linalg.norm((x_nsf_tmp[i],y_nsf_tmp[i])))) for i in range(num_time)] #don't mater to much for origin or nonstance feet
        # tmp = ab / np.linalg.norm([x_nsf_tmp,y_nsf_tmp])
        # cosd = [np.degrees(math.acos(tmp[0]))]
        ttcosd = ttcosd + (cosd)
        cosd = min(np.abs(cosd))
        print('cosd: ',cosd)
        print('chain_num: ',chain_num)

    q1 = state[:,0]
    q2 = state[:,2]
    q3 = state[:,4]
    x_h = pos_sf[0] - l*sin(q1)
    y_h = pos_sf[1] + l*cos(q1)
    x_nsk = x_h - lt*sin(q2)
    y_nsk = y_h - lt*cos(q2)
    x_nsf = x_nsk - ls*sin(q3)
    y_nsf = y_nsk - ls*cos(q3)

    return x_h,y_h,x_nsk,y_nsk,x_nsf,y_nsf,state

def init():
    for line in lines:
        line.set_data([],[])
    time_text.set_text('')
    return tuple(lines) + (time_text,)


def animate(i):

    x_bipedal = [0, x_h[i], x_nsk[i], x_nsf[i]]
    y_bipedal = [0, y_h[i], y_nsk[i], y_nsf[i]]
    # k=-1
    # x_bipedal = [0, x_h[k], x_nsk[k], x_nsf[k]]
    # y_bipedal = [0, y_h[k], y_nsk[k], y_nsf[k]]

    x_slop = [x_slop_up,x_slop_low,-2]
    y_slop = [y_slop_up,y_slop_low,y_slop_low]

    xlist = [x_bipedal, x_slop]
    ylist = [y_bipedal, y_slop]

    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    time_text.set_text(time_template % (i*dt))

    return tuple(lines) + (time_text,)



# Passive Dynamic Walking for bipedal robot
# % reset

from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import math

# parameters of leg structure
mt = 0.5  # mass of thigh
ms = 0.05  # mass of shank
mh = 0.5   #mass of hip
a1 = 0.375
b1 = 0.125
a2 = 0.175
b2 = 0.325
lt = a2 + b2  # length of thigh
ls = a1 + b1  # length of shank
l = lt + ls

g = 9.8  # acceleration due to gravity, in m/s^2
dt = 0.01 # time step of simulation

# slop of terran
_gamma = (9)
x_slop_low = 2 * cos(np.radians(_gamma))
x_slop_up = -x_slop_low
y_slop_low = -2 * sin(np.radians(_gamma))
y_slop_up = -y_slop_low


# initial states, defined by angle between two legs
_alpha = (20)
q1 = _alpha/2 - _gamma
q1d = 0.0
q2 = _alpha/2 + _gamma
q2d = 0.0
q3 = q2
q3d = 0.0
state = np.radians([q1, q1d, q2, q2d, q3, q3d])
pos_sf = [0,0]

x_h, y_h, x_nsk, y_nsk, x_nsf, y_nsf,state = step_cycle(state, pos_sf, dt)
# for i in range(1):
#     ini_state = [state[-1, 0],state[-1, 1],state[-1, 2],state[-1, 3],state[-1, 4],state[-1, 5]]
#     x_h_new, y_h_new, x_nsk_new, y_nsk_new, x_nsf_new, y_nsf_new, state = step_cycle(ini_state, dt)
#     x_h = np.insert(x_h,[len(x_h)],x_h_new,axis=0)
#     y_h = np.insert(y_h,[len(y_h)],y_h_new,axis=0)
#     x_nsk = np.insert(x_nsk,[len(x_nsk)],x_nsk_new,axis=0)
#     y_nsk = np.insert(y_nsk,[len(y_nsk)],y_nsk_new,axis=0)
#     x_nsf = np.insert(x_nsf,[len(x_nsf)],x_nsf_new,axis=0)
#     y_nsf = np.insert(y_nsf,[len(y_nsf)],y_nsf_new,axis=0)
#     pos_sf = [x_nsf[-1],y_nsf[-1]]

# f = open('out.txt', 'w')
# f.write(str(x_h))
# f.write(str(state))
# f.write('\n')


import matplotlib.pyplot as plt
# from numpy import random
import matplotlib.animation as animation
fig = plt.figure()
ax1 = plt.axes(xlim=(-2, 2), ylim=(-2,2))
ax1.grid(True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

lines = []
lobj = ax1.plot([],[],'o-',lw=2,color="black")[0]
lines.append(lobj)
lobj = ax1.plot([],[],lw=2,color="red")[0]
lines.append(lobj)

time_template = 'time = %.3fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

print('hello')

# call the animator.  blit=True means only re-draw the parts that have changed.
# ani = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=300, interval=10, blit=True)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x_h)),
                              interval=50, blit=True, init_func=init)
plt.show()
# ani.save('PDW.mp4', fps=15)


