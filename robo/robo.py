import inspect
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

'''
def three_linked_chain(t, x, xdot):
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

def rhseqn(t, x, xdot):
    """ we create rhs equations for the problem"""
    xdot[0] = x[1]
    xdot[1] = - 4 * x[0]

def root_fn(t, y, out):
    """ root function to check the object reached height Y1 """
    Y1 = 0.0
    out[0] = Y1 + y[0]
    return 0
def print_results(experiment_no, result, require_no_roots=False):
    ts, ys = result.values.t, result.values.y
    # Print computed values at tspan
    print('\n Experiment number: ', experiment_no)
    print('--------------------------------------')
    print('    t             Y               v')
    print('--------------------------------------')

    for (t, y, v) in zip(ts, ys[:, 0], ys[:, 1]):
        print('{:6.1f} {:15.4f} {:15.4f}'.format(t, y, v))


    t_roots, y_roots = result.roots.t, result.roots.y
    if not require_no_roots:
        # Print interruption points
        print('\n t_roots     y_roots        v_roots')
        print('--------------------------------------')
        if (t_roots is None) and (y_roots is None):
            print('{!s:6} {!s:15} {!s:15}'.format(t_roots, y_roots, y_roots))
        elif (t_roots is not None) and (y_roots is not None):
            if np.isscalar(t_roots):
                print('{:6.1f} {:15.4f} {:15.2f}'.format(t_roots, y_roots[0], y_roots[1]))
            else:
                for (t, y, v) in zip(t_roots, y_roots[:, 0], y_roots[:, 1]):
                    print('{:6.6f} {:15.4f} {:15.4f}'.format(t, y, v))
        else:
            print('Error: one of (t_roots, y_roots) is None while the other not.')
    else:
        print('Computation failed.')

'''

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
    Q_r[0,1] = -ms*a1*(lt+b1) + mt*b2*(ls+a2)
    Q_r[0,0] = Q_r[0,1] + (mh*l + 2*mt*(a2+ls) + ms*a1) * l*cos(alpha)
    Q_r[1,1] = 0
    Q_r[1,0] = Q_r[0,1]

    Q_l = np.zeros((2,2))
    Q_l[1,0] = -(ms*(b1+lt) + mt*b2) * l*cos(alpha)
    Q_l[0,0] = Q_l[1,0] + (ms+mt+mh)*l**2 + ms*a1**2 + mt*(a2+ls)**2
    Q_l[0,1] = Q_l[1,0] + ms*(b1+lt)**2 + mt*b2**2
    Q_l[1,1] = ms*(lt+b1)**2 + mt*b2**2

    q_r = [state[1],state[3]]
    rhs = np.dot(Q_r,q_r)
    sol = np.linalg.solve(Q_l,rhs)

    A = np.linalg.inv(Q_l)
    B = np.dot(A,Q_r)
    xd = np.zeros_like(state)
    xd[0] = state[0]
    xd[2] = state[2]
    xd[4] = state[2]
    xd[1] = sol[0]
    xd[3] = sol[1]
    xd[5] = sol[1]

    return xd



def step_cycle(state,pos_sf,_time,step_out=0):
    # print('start state: ', state)

    ttcosd = []
    cnt = 0
    num_time = 2
    dt = _time / num_time
    chain_num=3 # starts with three-chain state
    success = 1


    while True:
        try:
            # print('cnt: ',cnt)
            t = np.arange(0.0, _time, dt)
            if cnt==0:
                try:
                    tmp = integrate.odeint(three_linked_chain, state, t)
                    state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
                except:
                    print(state)
                    print(lineno())
                    sys.exit("wrong in this step!")
                if step_out:
                    print('three linked chain, time: %s <<<' % (cnt * _time))
                    print(state[-1])
                cnt += 1
                continue
            # knee strike
            if  ((state[-num_time:,2])>(state[-num_time:,0])).any() and cnt>0.1/_time\
            and (chain_num==3) and (np.min(state[-num_time:,2]-state[-num_time:,4])<np.radians(0.001)):
                tmp = knee_strike(state[-1])
                state = np.insert(state, [len(state)], tmp, axis=0)
                chain_num = 2
                if step_out:
                    print('===============================================knee strike, time: %s' % (cnt*_time))
                    print(state[-1])
                continue
            # heel strike
            if (chain_num==2) and (dist<0.005): # in degree
                tmp = heel_strike(state[-1])
                state = np.insert(state, [len(state)], tmp, axis=0)
                chain_num = 3
                if step_out:
                    print('===============================================heel strike, time: %s' % (cnt*_time))
                    print(state[-1])
                    # print('end state: ',state[-1,:])

                # update initial condition
                q1 = (state[-1, 2] + state[-1, 4]) / 2
                q1d = (state[-1, 3] + state[-1, 5]) / 2
                q2 = state[-1, 0]
                q2d = state[-1, 1]
                q3 = state[-1, 0]
                q3d = state[-1, 1]
                tmp = [q1, state[0, 1], q2, state[0, 3], q3,state[0, 5]]  ##############################ATTENTION HERE!
                tmp = [q1, q1d, q2, q2d, q3, q3d]  ##############################ATTENTION HERE!
                state = np.insert(state, [len(state)], tmp, axis=0)
                if step_out:
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>new initial')
                    print(state[-1])

                break
            # three linked chain
            if (chain_num==3):
                try:
                    tmp = integrate.odeint(three_linked_chain, state[-1], t)
                    state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
                    if step_out:
                        print('three linked chain, time: %s <<<' % (cnt*_time))
                        print(state[-1])
                    chain_num = 3
                    cnt += 1
                except:
                    print(state)
                    print(lineno())
                    break
                    # sys.exit("wrong in this step!")
            # two linked chain
            if (chain_num==2):
                try:
                    tmp = integrate.odeint(two_linked_chain, state[-1], t)
                    state = np.insert(tmp[1:len(tmp)+1],[0],state,axis=0)
                    if step_out:
                        print('two linked chain, time: %s <<<' % (cnt*_time))
                        print(state[-1])
                    chain_num = 2
                    cnt += 1
                except:
                    print(state)
                    print(lineno())
                    break
                    # sys.exit("wrong in this step!")
        except:
            print("this is not a good step!")
            success = 0

        q1 = tmp[0:-1,0]
        q2 = tmp[0:-1,2]
        q3 = tmp[0:-1,4]
        x_h_tmp = -l*sin(q1)
        y_h_tmp = l*cos(q1)
        x_nsk_tmp = x_h_tmp + lt*sin(q2)
        y_nsk_tmp = y_h_tmp - lt*cos(q2)
        x_nsf_tmp = x_nsk_tmp + ls*sin(q3)
        y_nsf_tmp = y_nsk_tmp - ls*cos(q3)

        dist = 1000
        for i in range(len(x_nsf_tmp)):
            v1 = [x_nsf_tmp[i], y_nsf_tmp[i]]
            v2 = [sin((_gamma)), cos((_gamma))]
            ab = np.dot(v1, v2)
            dist = min(dist,ab)
        # print('dist: ',dist)
        # print('chain_num: ',chain_num)

        if dist <-0.1 or cnt > 6 / _time:  # usually it takes less than three seconds for a step
            print("this is not a good step!")
            success = 0
            break

    q1 = state[0:-1,0]
    q2 = state[0:-1,2]
    q3 = state[0:-1,4]
    x_h = pos_sf[0] - l*sin(q1)
    y_h = pos_sf[1] + l*cos(q1)
    x_nsk = x_h + lt*sin(q2)
    y_nsk = y_h - lt*cos(q2)
    x_nsf = x_nsk + ls*sin(q3)
    y_nsf = y_nsk - ls*cos(q3)

    return x_h,y_h,x_nsk,y_nsk,x_nsf,y_nsf,state, success

def init():
    for line in lines:
        line.set_data([],[])
    time_text.set_text('')
    return tuple(lines) + (time_text,)


import sys
def animate(i):
    try:
        x_bipedal = [x_sf[i], x_sk[i], x_h[i], x_nsk[i], x_nsf[i]]
        y_bipedal = [y_sf[i], y_sk[i], y_h[i], y_nsk[i], y_nsf[i]]
        # k=0
        # x_bipedal = [x_sf[i], x_sk[i], x_h[k], x_nsk[k], x_nsf[k]]
        # y_bipedal = [y_sf[i], y_sk[i], y_h[k], y_nsk[k], y_nsf[k]]
    except:
        sys.exit("wrong in animation!")
    x_slop = [x_slop_up,x_slop_low,-2]
    y_slop = [y_slop_up,y_slop_low,y_slop_low]

    xlist = [x_bipedal, x_slop]
    ylist = [y_bipedal, y_slop]

    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    time_text.set_text('time = %.3fs' % (i*dt))

    try:
        orbit_q1 = orbit[0:i, 0]
        orbit_q1d = orbit[0:i, 1]
        orbit_q2 = orbit[0:i, 2]
        orbit_q2d = orbit[0:i, 3]
        orbit_q3 = orbit[0:i, 4]
        orbit_q3d = orbit[0:i, 5]

        xlist = [orbit_q1, orbit_q2, orbit_q3]
        ylist = [orbit_q1d, orbit_q2d, orbit_q3d]

        for lnum, line in enumerate(lines_obt):
            line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.

        time_text_obt.set_text('time = %.3fs' % (i * dt))
    except:
        sys.exit("wrong in animating orbit!")

    return tuple(lines_obt) + tuple(lines) + (time_text_obt,) + (time_text,)


def robo(show_ani):
    # Passive Dynamic Walking for bipedal robot
    # % reset

    # parameters of leg structure
    paras = np.loadtxt('parameters.txt', delimiter='  ')
    try:
        last = paras[-1,:]
    except:
        last = paras
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
    step_tt = 1
    step_out = 0  # show state information at every time step of the first step cycle
    if show_ani:
        step_tt = 6
        step_out = 1  # show state information at every time step of the first step cycle

    # slop of terran
    global _gamma
    _gamma = 0.0504
    # _gamma = 0.0904


    state = [0.1877, -1.1014, -0.2884, -0.0399, -0.2884, -0.0399]
    # state = [q1, -1.1014, q2, -0.0399, q3, -0.0399]
    print('init:',state)
    pos_sf = [0,0]


    global x_sf,y_sf,x_h,y_h,x_nsk,y_nsk,x_nsf,y_nsf
    global x_sk, y_sk

    t = np.arange(0.0, 1.5, 0.0001)
    # state = integrate.odeint(three_linked_chain, state, t)

######
    '''
    from scikits.odes import ode

    # solver = ode('cvode', three_linked_chain, old_api=False)
    # solution = solver.solve(t, state)

    # solver = ode('cvode', rhseqn, old_api=False)
    # initx = [1, 0.1]
    # solution = solver.solve([0., 1., 2.], initx)
    # for t, u in zip(solution.values.t, solution.values.y):
    #     print('{0:>4.0f} {1:15.6g} '.format(t, u[0]))

    solver = ode('cvode', rhseqn, nr_rootfns=1, rootfn=root_fn, old_api=False)
    t_end2 = 100.0  # Time of free fall for experiments 3,4
    tspan = np.arange(0, t_end2 + 1, 1.0, np.float)
    y0 = [1, 0.1]
    # print_results(1, solver.solve(tspan, y0))

    solver = ode('cvode', three_linked_chain, nr_rootfns=1, rootfn=root_ks, old_api=False)
    tspan = t
    y0 = state
    result = solver.solve(tspan, y0)
    print(result.roots.t)
    print(result.roots.y)
########

    # knee strike
    for i in range(len(t)):
        if ((state[i, 2]) > (state[i, 0])) and (state[i, 2] - state[i, 4]) < 0.001:
            tt = i
            break
    state = state[0:tt, :]

    q1 = state[0:-1, 0]
    q2 = state[0:-1, 2]
    q3 = state[0:-1, 4]
    x_h = pos_sf[0] - l * sin(q1)
    y_h = pos_sf[1] + l * cos(q1)
    x_nsk = x_h + lt * sin(q2)
    y_nsk = y_h - lt * cos(q2)
    x_nsf = x_nsk + ls * sin(q3)
    y_nsf = y_nsk - ls * cos(q3)
    show_ani = 1
    x_sf = np.zeros_like(x_h)
    y_sf = np.zeros_like(x_h)
    '''

    # start walking....
    x_h, y_h, x_nsk, y_nsk, x_nsf, y_nsf,state, success = step_cycle(state, pos_sf, dt, step_out)
    step_time = len(x_h)
    x_sf = np.zeros_like(x_h)
    y_sf = np.zeros_like(x_h)


    if success==0:
        output = [[10000, -10000]]
        np.savetxt('out.txt',output, delimiter='  ')
        # return

    ini_state = state[-1]
    # update location
    pos_sf = [x_nsf[-1], y_nsf[-1]]

    diff = ini_state - state[0] # difference in starting state of two step cycle
    stability = np.linalg.norm(diff) **2 * 1000
    v1 = [cos(_gamma),-sin(_gamma)]
    v2 = [x_nsf[-1],y_nsf[-1]]
    disp = np.dot(v1,v2)
    if disp<0:
        speed = 1000 * disp
    else:
        speed = disp
    output = [[stability, speed]]
    np.savetxt('out.txt',output, delimiter='  ')

    # more steps...
    while step_idx<step_tt:
        # start another step
        x_h_new, y_h_new, x_nsk_new, y_nsk_new, x_nsf_new, y_nsf_new, state_new, success = step_cycle(ini_state, pos_sf, dt)
        x_sf_new = np.ones_like(x_h_new) * pos_sf[0]
        y_sf_new = np.ones_like(x_h_new) * pos_sf[1]
        # add trajectory of new step
        x_sf = np.insert(x_sf,[len(x_sf)],x_sf_new,axis=0)
        y_sf = np.insert(y_sf,[len(y_sf)],y_sf_new,axis=0)
        x_h = np.insert(x_h,[len(x_h)],x_h_new,axis=0)
        y_h = np.insert(y_h,[len(y_h)],y_h_new,axis=0)
        x_nsk = np.insert(x_nsk,[len(x_nsk)],x_nsk_new,axis=0)
        y_nsk = np.insert(y_nsk,[len(y_nsk)],y_nsk_new,axis=0)
        x_nsf = np.insert(x_nsf,[len(x_nsf)],x_nsf_new,axis=0)
        y_nsf = np.insert(y_nsf,[len(y_nsf)],y_nsf_new,axis=0)
        y_nsf = np.insert(y_nsf,[len(y_nsf)],y_nsf_new,axis=0)
        state = np.insert(state,[len(state)],state_new,axis=0)

        ini_state = state[-1]
        # update location
        pos_sf = [x_nsf[-1],y_nsf[-1]]
        step_idx += 1

    if show_ani:
        print('start animation...')

        global fig, ax1, ax2
        fig, (ax1,ax2) = plt.subplots(2,1)
        # plot hybrid trajectory in state space
        global orbit, orbit_q1, orbit_q2, orbit_q3, orbit_q1d, orbit_q2d, orbit_q3d
        orbit_q1 = []
        orbit_q2 = []
        orbit_q3 = []
        orbit_q1d = []
        orbit_q2d = []
        orbit_q3d = []
        orbit = state

        ymin = min([min(orbit[:, 1]), min(orbit[:, 3]), min(orbit[:, 5])])
        ymax = max([max(orbit[:, 1]), max(orbit[:, 3]), max(orbit[:, 5])])
        xmin = min([min(orbit[:, 0]), min(orbit[:, 2]), min(orbit[:, 4])])
        xmax = max([max(orbit[:, 0]), max(orbit[:, 2]), max(orbit[:, 4])])
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
        lobj = ax1.plot([], [], lw=2, color="black")[0]
        lines_obt.append(lobj)
        lobj = ax1.plot([], [], lw=2, color="red")[0]
        lines_obt.append(lobj)
        lobj = ax1.plot([], [], lw=2, color="blue")[0]
        lines_obt.append(lobj)

        time_text_obt = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)


        x_sk = x_h * (c_a1 + c_b1) + x_sf * (c_a2 + c_b2)
        y_sk = y_h * (c_a1 + c_b1) + y_sf * (c_a2 + c_b2)
        s =  (c_a1 + c_b1) +  (c_a2 + c_b2)
        x_sk /= s
        y_sk /= s
        # slop
        global x_slop_low, x_slop_up, y_slop_low, y_slop_up
        x_slop_low = 2 * cos((_gamma))
        x_slop_up = -x_slop_low
        y_slop_low = -2 * sin((_gamma))
        y_slop_up = -y_slop_low


        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-0.5, 2])
        ax2.grid(True)
        ax1.set_xlabel('q')
        ax1.set_ylabel('dq')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')

        global lines, time_text
        lines = []
        lobj = ax2.plot([], [], 'o-', lw=2, color="black")[0]
        lines.append(lobj)
        lobj = ax2.plot([], [], lw=2, color="red")[0]
        lines.append(lobj)

        time_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x_h)),
                                      interval=1, blit=True, init_func=init)

        plt.show()
        print('end animation...')
        # Set up formatting for the movie files
        # print('saving animation...')
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save('PDW.mp4', writer=writer)
        # print('animation saved')

from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

robo(1)

