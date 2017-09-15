import inspect
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


def three_linked_chain_rt(t, x, xdot):
    state = x
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
    h233 = -ms*lt*b1 * sin(q3-q2)
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
    try:
        sol = np.linalg.solve(H,rhs)
    except:
        print(H)
        sys.exit("wrong in solving three linked dynamics!")

    xdot[0] =  q1d
    xdot[1] =  sol[0]
    xdot[2] =  q2d
    xdot[3] =  sol[1]
    xdot[4] =  q3d
    xdot[5] =  sol[2]

def root_ks(t, y, out):
    """ root function to check the object reached height Y1 """
    out[0] = y[2] - y[4]
    return 0

# NOTICE, different name needs to be used
def two_linked_chain_rt(t, xx, xxdot):
    state = xx
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

    xdot = np.zeros_like(state)
    xxdot[0] = q1d
    xxdot[1] = sol[0]
    xxdot[2] = q2d
    xxdot[3] = sol[1]
    xxdot[4] = q2d
    xxdot[5] = sol[1]


def root_hs(t, y, out):
    """ root function to check the object reached height Y1 """

    q1 = y[0]
    q2 = y[2]
    q3 = y[4]
    x_h_tmp = -l * sin(q1)
    y_h_tmp = l * cos(q1)
    x_nsk_tmp = x_h_tmp + lt * sin(q2)
    y_nsk_tmp = y_h_tmp - lt * cos(q2)
    x_nsf_tmp = x_nsk_tmp + ls * sin(q3)
    y_nsf_tmp = y_nsk_tmp - ls * cos(q3)

    v1 = [x_nsf_tmp, y_nsf_tmp]
    v2 = [sin((_gamma)), cos((_gamma))]
    dist = np.dot(v1, v2)
    # print('dist:',dist)

    out[0] =  dist
    return 0

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

    '''equations from MIT thesis WRONG IN SIGN OF h233!'''
    B = np.zeros((3,3))
    h122 = -(mt*b2+ms*lt) * l * sin(q1-q2)
    h133 = -ms*b1*l * sin(q1-q3)
    h211 = -h122
    h233 = -ms*lt*b1 * sin(q3-q2)
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
    try:
        sol = np.linalg.solve(H,rhs)
    except:
        print(H)
        sys.exit("wrong in solving three linked dynamics!")

    return [ q1d, sol[0], q2d, sol[1], q3d, sol[2] ]


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

    '''using equations from MIT thesis, WRONG!'''
    # alpha = cos(q1-q2)
    # beta = cos(q1-q3)
    # gamma = cos(q2-q3)
    #
    # Q_r = np.zeros((2,3))
    # Q_r[0,0] = -(ms*lt+mt*b2)*l*cos(alpha) - ms*b1*l*cos(beta) + (mt+ms+mh)*l**2 + ms*a1**2 + mt*(ls+a2)**2
    # Q_r[0,1] = -(ms*lt+mt*b2)*l*cos(alpha) + ms*b1*lt*cos(gamma) + mt*b2**2 + ms*lt**2
    # Q_r[0,2] = -ms*b1*l*cos(beta) + ms*b1*lt*cos(gamma) + ms*b1**2
    # Q_r[1,0] = -(ms*lt+mt*b2)*l*cos(alpha) - ms*b1*l*cos(beta)
    # Q_r[1,1] = ms*b1*lt*cos(gamma) + ms*lt**2 + mt*b2**2
    # Q_r[1,2] = ms*b1*lt*cos(gamma) + ms*b1**2
    #
    # Q_l = np.zeros((2,2))
    # Q_l[1,1] = ms*(lt+b1)**2 + mt*b2**2
    # Q_l[1,0] = -(ms*(b1+lt) + mt*b2) * l*cos(alpha)
    # Q_l[0,1] = Q_l[1,0] + ms*(lt+b1)**2 + mt*b2**2
    # Q_l[0,0] = Q_l[1,0] + mt*(ls+a2)**2 + (mh+mt+ms)*l**2 + ms*a1**2

    q21 = -(ms*lt+mt*b2)*l*cos(q1-q3) - ms*b2*l*cos(q1-q2)
    q22 = ms*b1*lt*cos(q2-q3) + mt*b2**2 + ms*lt**2
    q23 = ms*b1*lt*cos(q2-q3) + ms*b1**2
    q11 = q21 + (mt+ms+mh)*l**2 + ms*a1**2 + mt*(ls+a2)**2
    q12 = -(ms*lt+mt*b2)*l*cos(q1-q2) + ms*b1*lt*cos(q2-q3) + mt*b2**2 + ms*lt**2
    q13 = -ms*b1*l*cos(q1-q3) + ms*b1*lt*cos(q2-q3) + ms*b1**2
    Q_r = [[q11, q12, q13], [q21, q22, q23]]

    q21 = -(ms*(b1+lt)+mt*b2)*l*cos(q1-q2)
    q22 = ms*(lt+b1)**2 + mt*b2**2
    q11 = q21 + mt*(ls+a2)**2 + (mh+ms+mt)*l**2 + ms*a1**2
    q12 = q21 + ms*(lt+b1)**2 + mt*b2**2
    Q_l = [[q11,q12],[q21,q22]]

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

    '''using equations from MIT thesis, WRONG!'''
    # alpha = cos(q1-q2)
    #
    # Q_r = np.zeros((2,2))
    # Q_r[0,1] = -ms*a1*(lt+b1) + mt*b2*(ls+a2)
    # Q_r[0,0] = Q_r[0,1] + (mh*l + 2*mt*(a2+ls) + ms*a1) * l*cos(alpha)
    # Q_r[1,1] = 0
    # Q_r[1,0] = Q_r[0,1]
    #
    # Q_l = np.zeros((2,2))
    # Q_l[1,0] = -(ms*(b1+lt) + mt*b2) * l*cos(alpha)
    # Q_l[0,0] = Q_l[1,0] + (ms+mt+mh)*l**2 + ms*a1**2 + mt*(a2+ls)**2
    # Q_l[0,1] = Q_l[1,0] + ms*(b1+lt)**2 + mt*b2**2
    # Q_l[1,1] = ms*(lt+b1)**2 + mt*b2**2

    q12 = -ms*a1*(lt+b1) - mt*b2*(ls+a2)
    q11 = (mh*l+2*mt*(a2+ls)+2*ms*a1)*l*cos(q1-q2) + q12
    Q_r = [[q11,q12],[q12,0]]

    q21 = -(ms*(b1+lt)+mt*b2)*l*cos(q1-q2)
    q22 = ms*(lt+b1)**2 + mt*b2**2
    q11 = mt*(ls+a2)**2 + (mh+ms+mt)*l**2 + ms*a1**2 +q21
    q12 = ms*(lt+b1)**2 + mt*b2**2 + q21
    Q_l = [[q11,q12],[q21,q22]]

    q_r = [state[1],state[5]]
    rhs = np.dot(Q_r,q_r)
    sol = np.linalg.solve(Q_l,rhs)

    A = np.linalg.inv(Q_l)
    B = np.dot(A,Q_r)
    xd = np.zeros_like(state)
    xd[0] = state[0]
    xd[2] = state[2]
    xd[4] = state[2]
    xd[1] = sol[1]
    xd[3] = sol[0]
    xd[5] = sol[0]

    return xd


def init():
    for line in lines:
        line.set_data([],[])
    time_text.set_text('')
    return tuple(lines) + (time_text,)


import sys
def animate(i):
    try:
        # x_bipedal = [x_sf[i], x_sk[i], x_h[i], x_nsk[i], x_nsf[i]]
        # y_bipedal = [y_sf[i], y_sk[i], y_h[i], y_nsk[i], y_nsf[i]]
        x_sl = [x_sf[i],x_sk[i],x_h[i]]
        y_sl = [y_sf[i],y_sk[i],y_h[i]]
        x_nst = [x_h[i], x_nsk[i]]
        y_nst = [y_h[i], y_nsk[i]]
        x_nss = [x_nsk[i], x_nsf[i]]
        y_nss = [y_nsk[i], y_nsf[i]]
    except:
        sys.exit("wrong in animating walking!")
    x_slop = [x_slop_up,x_slop_low,-2]
    y_slop = [y_slop_up,y_slop_low,y_slop_low]

    # xlist = [x_bipedal, x_slop]
    # ylist = [y_bipedal, y_slop]
    xlist = [x_sl, x_nst, x_nss, x_slop]
    ylist = [y_sl, y_nst, y_nss, y_slop]

    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    time_text.set_text('time = %.3fs' % (i*dt))

    try:

        xlist = [orbit_q2[0:i], orbit_q3[0:i]]
        ylist = [orbit_q2d[0:i], orbit_q3d[0:i]]

        for lnum, line in enumerate(lines_obt):
            line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.

        time_text_obt.set_text('time = %.3fs' % (i * dt))
    except:
        sys.exit("wrong in animating orbit!")

    return tuple(lines_obt) + tuple(lines) + (time_text_obt,) + (time_text,)

def show_animation():
    print('start animation...')

    global fig, ax1, ax2
    fig, (ax1,ax2) = plt.subplots(2,1)

    # plot hybrid trajectory in state space
    xmin = min([min(orbit_q1), min(orbit_q2), min(orbit_q3)])
    xmax = max([max(orbit_q1), max(orbit_q2), max(orbit_q3)])
    ymin= min([min(orbit_q1d), min(orbit_q2d), min(orbit_q3d)])
    ymax = max([max(orbit_q1d), max(orbit_q2d), max(orbit_q3d)])
    # ymin = -2.5
    # ymax = 2.5
    # xmin = -0.3
    # xmax = 0.4
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.grid(True)
    ax1.set_xlabel('angle')
    ax1.set_ylabel('angular velocity')

    global lines_obt, time_text_obt
    lines_obt = []
    lobj = ax1.plot([], [], lw=2, color="red")[0]
    lines_obt.append(lobj)
    lobj = ax1.plot([], [], lw=2, color="blue")[0]
    lines_obt.append(lobj)
    time_text_obt = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

    # slop
    global x_slop_low, x_slop_up, y_slop_low, y_slop_up
    x_slop_low = 2 * cos((_gamma))
    x_slop_up = -x_slop_low
    y_slop_low = -2 * sin((_gamma))
    y_slop_up = -y_slop_low


    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-0.1, 2])
    # ax2.axis('auto')
    ax2.grid(True)
    ax1.set_xlabel('q')
    ax1.set_ylabel('dq')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    global lines, time_text
    lines = []
    lobj = ax2.plot([], [], '.-', lw=2, color="black")[0]
    lines.append(lobj)
    lobj = ax2.plot([], [], '.-', lw=2, color="red")[0]
    lines.append(lobj)
    lobj = ax2.plot([], [], '.-', lw=2, color="blue")[0]
    lines.append(lobj)
    lobj = ax2.plot([], [], '.-', lw=2, color="black")[0]
    lines.append(lobj)

    time_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x_h)),
                                  interval=1, blit=True, init_func=init)

    plt.show()
    print('end animation...')
    # Set up formatting for the movie files
    print('saving animation...')
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('PDW.mp4', writer=writer)
    print('animation saved')

def robo(show_ani):
    # Passive Dynamic Walking for bipedal robot
    # % reset

    # parameters of leg structure
    paras = np.loadtxt('../data/parameters.txt', delimiter='  ')
    try:
        last = paras[-1,:]
    except:
        last = paras

    global c_a1,c_b1,c_a2,c_b2
    q1 = last[0]
    q2 = last[1]
    q3 = last[2]
    c_mh = 0.5
    c_mt = 0.5
    c_ms = 0.05
    c_a1 = last[3]
    c_b1 = last[4]
    c_a2 = last[5]
    c_b2 = last[6]

    global mh,mt,ms,a1,b1,a2,b2,lt,ls,l
    M = 1# total weight
    L = 1 # total lenth
    mh = M * c_mh   #mass of hip
    mt = M * c_mt  # mass of thigh
    ms = M * c_ms # mass of shank
    a1 = L * c_a1
    b1 = L * c_b1
    a2 = L * c_a2
    b2 = L * c_b2
    lt = a2 + b2  # length of thigh
    ls = a1 + b1  # length of shank
    l = lt + ls

    global g,dt
    g = 9.8  # acceleration due to gravity, in m/s^2
    dt = 0.001 # time step of simulation
    step_idx = 1
    step_tt = 1 # how many steps going to walk
    step_out = 0  # show state information at every time step of the first step cycle
    if show_ani:
        step_tt = 5
        step_out = 1  # show state information at every time step of the first step cycle

    # slop of terran
    global _gamma
    _gamma = 0.0504


    global x_sf,y_sf,x_h,y_h,x_nsk,y_nsk,x_nsf,y_nsf,x_sk,y_sk
    global orbit_q1,orbit_q2,orbit_q3,orbit_q1d,orbit_q2d,orbit_q3d
    x_sf, y_sf, x_h, y_h, x_nsk, y_nsk, x_nsf, y_nsf, x_sk, y_sk = (np.zeros(0) for _ in range(10))
    orbit_q1, orbit_q2, orbit_q3, orbit_q1d, orbit_q2d, orbit_q3d = (np.zeros(0) for _ in range(6))
    tt_time_step =[]

    from scikits.odes import ode
    ini_state = [0.1877, -1.1014, -0.2884, -0.0399, -0.2884, -0.0399]
    pos_sf = [0,0]
    # state = [q1, -1.1014, q2, -0.0399, q3, -0.0399]
    while True:
        try:
            solver = ode('cvode', three_linked_chain_rt, nr_rootfns=1, rootfn=root_ks, old_api=False)
            result = solver.solve([0,1], ini_state ) # detect event, generally it takes less than one second to reach knee strike
            t = np.arange(0.0, result.roots.t, dt)
            t = np.append(t,result.roots.t) # add event time stamp, right continuous
            tmp = integrate.odeint(three_linked_chain, ini_state , t)  # integrate three chain dynamics
            state = np.insert(tmp[1:len(tmp) + 1], [0], ini_state , axis=0)
            if step_out:
                print('init condition: ',ini_state )
                print('time to knee strike: ', result.roots.t)
                print('three link dynamics end state: ',result.roots.y)

            tmp = knee_strike(state[-1])
            state = np.insert(state, [len(state)], tmp, axis=0)
            if step_out:
                print('knee strike end state: ',state[-1])

            solver = ode('cvode', two_linked_chain_rt, nr_rootfns=1, rootfn=root_hs, old_api=False)
            result = solver.solve([0,0.5], state[-1]) # detect event, generally it takes less than half second to reach heel strike
            t = np.arange(0.0, result.roots.t, dt)
            t = np.append(t,result.roots.t) # add event time stamp, right continuous
            tmp = integrate.odeint(two_linked_chain, state[-1], t) # integrate two chain dynamics
            state = np.insert(tmp[1:len(tmp) + 1], [0], state, axis=0)
            if step_out:
                print('time to heel strike: ', result.roots.t)
                print('two link dynamics end state: ', result.roots.y)

            tmp = heel_strike(state[-1])
            state = np.insert(state, [len(state)], tmp, axis=0)
            if step_out:
                print('heel strike end state: ',state[-1])

            if step_idx ==1:
                time_idx = len(state)
            else:
                time_idx = len(state) + tt_time_step[-1]
            tt_time_step += [time_idx]

            if step_idx%2 ==1:
                orbit_q1 = np.append(orbit_q1,state[:, 0])
                orbit_q2 = np.append(orbit_q2,state[:, 2])
                orbit_q3 = np.append(orbit_q3,state[:, 4])
                orbit_q1d = np.append(orbit_q1d,state[:, 1])
                orbit_q2d = np.append(orbit_q2d,state[:, 3])
                orbit_q3d = np.append(orbit_q3d,state[:, 5])
            else:
                orbit_q1 = np.append(orbit_q1,state[:, 2])
                orbit_q2 = np.append(orbit_q2,state[:, 0])
                orbit_q3 = np.append(orbit_q3,state[:, 0])
                orbit_q1d = np.append(orbit_q1d,state[:, 3])
                orbit_q2d = np.append(orbit_q2d,state[:, 1])
                orbit_q3d = np.append(orbit_q3d,state[:, 1])

            if step_idx==1:
                t_start = 0
                t_end = tt_time_step[-1]
            else:
                t_start = tt_time_step[-2]
                t_end = tt_time_step[-1]

            q1 = state[:, 0]
            q2 = state[:, 2]
            q3 = state[:, 4]

            x_sf_tmp = np.ones_like(q1) * pos_sf[0]
            x_sf = np.append(x_sf,x_sf_tmp)
            y_sf_tmp = np.ones_like(q1) * pos_sf[1]
            y_sf = np.append(y_sf,y_sf_tmp)

            x_h_tmp = x_sf_tmp - l * sin(q1)
            x_h = np.append(x_h,x_h_tmp)
            y_h_tmp = y_sf_tmp + l * cos(q1)
            y_h = np.append(y_h,y_h_tmp)

            x_nsk_tmp =  x_h_tmp + lt * sin(q2)
            x_nsk = np.append(x_nsk,x_nsk_tmp)
            y_nsk_tmp = y_h_tmp - lt * cos(q2)
            y_nsk = np.append(y_nsk,y_nsk_tmp)

            x_nsf_tmp = x_nsk_tmp + ls * sin(q3)
            x_nsf = np.append(x_nsf,x_nsf_tmp)
            y_nsf_tmp = y_nsk_tmp - ls * cos(q3)
            y_nsf = np.append(y_nsf,y_nsf_tmp)

            ini_state = [state[-1, 4],state[-1, 5],state[-1, 0],state[-1, 1],state[-1, 0],state[-1, 1]]
        except:
            output = [[10000, -10000]] # something unexpected. set it as very unstable and slow.
            np.savetxt('../data/out.txt', output, delimiter='  ')
        if step_idx == 1:
            diff = [x - y for x, y in zip(ini_state, state[0])]  # difference in starting state of two step cycle
            stability = np.linalg.norm(diff)
            v1 = [cos(_gamma), -sin(_gamma)]
            v2 = [x_nsf[-1], y_nsf[-1]]
            disp = np.dot(v1, v2)
            if disp < 0:
                speed = 1000 * disp
            else:
                speed = disp
            output = [[stability, speed]]
            np.savetxt('out.txt', output, delimiter='  ')

        pos_sf = [x_nsf[-1], y_nsf[-1]]
        if step_idx==step_tt:
            break
        step_idx += 1


    x_sk = x_h * (c_a1 + c_b1) + x_sf * (c_a2 + c_b2)
    y_sk = y_h * (c_a1 + c_b1) + y_sf * (c_a2 + c_b2)
    s =  (c_a1 + c_b1) +  (c_a2 + c_b2)
    x_sk /= s
    y_sk /= s

    if show_ani:
        show_animation()


from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == '__main__':
    robo(1)

