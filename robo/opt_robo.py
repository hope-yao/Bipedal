# multi object optimization for bipedal passive dynamic walking
# parameters: robot leg structure
# objective1: stability
# objective2: speed
# objective function is based on bipedal simulator robo.py

from scipy.optimize import minimize

import numpy as np
# from numpy import random
# import os
# from numpy import sin, cos
# import scipy.integrate as integrate
# import math
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import robo


c_alpha =  1 # defined by angle between two legs, times 20 in degree
# constrain: c_mh>0.3 c_mt>0.1 c_mh+c_mt<1
c_mh = 0.47
c_mt = 0.47
c_ms = (1-c_mh-c_mt)

c_a1 = 0.25
c_b1 = 0.25
c_a2 = 0.25
c_b2 = 1-c_a1-c_b1--c_a2

leg_struc = np.array([[c_alpha, c_mh, c_mt, c_a1, c_b1, c_a2]])
def obj_fun(x,show_ani):
    print(">>>>>>>>>>>>>>>>>>>>>>>leg structure<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(x)
    para_file = 'parameters.txt'
    np.savetxt(para_file, x, delimiter='  ')
    robo.robo(show_ani)
    out_file = 'out.txt'
    obj_val = np.loadtxt(out_file, delimiter='  ')
    # minimize negative displacement, times 10 to make it around same magnitude
    paerto_obj = obj_val[0] * pareto_para[0] - obj_val[1]*pareto_para[1]*10
    return paerto_obj
# obj_fun(leg_struc,1)
# obj_fun([1.22,0.3,0.8,0.01,0.66579248,0.31420752],1)


pareto_para = [[],[]]
num_pareto_points = 1
tt = [(i / (num_pareto_points+1)) for i in range(num_pareto_points+1)]
tt.pop(0)
for w in tt:
    pareto_para[0] = w / num_pareto_points
    pareto_para[1] = 1 - pareto_para[0]

    # notice, constrains are in the form of x+b>0
    cons = (
        {'type': 'ineq',
         'fun': lambda x: np.array([0.99 - (x[1] + x[2])]),
         'jac': lambda x: np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([0.99 - (x[3] + x[4] + x[5])]),
         'jac': lambda x: np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])}
    )
    para_bound = [(0.5,1.5),(0.4,0.6),(0.3,0.6),(0.1,0.4),(0.1,0.4),(0.1,0.4)]
    # res = minimize(obj_fun, leg_struc, args=0, method='L-BFGS-B', jac=None, bounds=para_bound, options={'maxfun':200, 'eps':1e-5, 'disp': False})
    res = minimize(obj_fun, leg_struc, args = 0, constraints=cons, method='SLSQP', jac=None, bounds=para_bound, options={'maxiter':30, 'eps':1e-4, 'disp': True})
    # import barecmaes2 as cma
    # x = cma.fmin(obj_fun, leg_struc, 0.5, args=0)

print('optimization result:')
# print(res)

# fmin_l_bfgs_b(obj_fun, leg_struc, approx_grad=True, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
# os.system('python robo.py')
robo.robo(1)
