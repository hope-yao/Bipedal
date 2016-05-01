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
c_ms = 0.06

c_a1 = 0.375
c_b1 = 0.125
c_a2 = 0.175
c_b2 = 0.325
leg_struc = np.array([c_alpha, c_mh, c_mt, c_ms, c_a1, c_b1, c_a2, c_b2])

# para_file = 'parameters.txt'
# np.savetxt(para_file, [leg_struc], delimiter='  ')
# robo.robo(1)

def obj_fun(x,show_ani):
    print(">>>>>>>>>>>>>>>>>>>>>>>leg structure<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(x)
    para_file = 'parameters.txt'
    np.savetxt(para_file, [x], delimiter='  ')
    robo.robo(show_ani)
    out_file = 'out.txt'
    obj_val = np.loadtxt(out_file, delimiter='  ')
    # minimize negative displacement
    paerto_obj = obj_val[0] * pareto_para[0] - obj_val[1]*pareto_para[1]
    print('obj value:', obj_val)
    # obj with initial para:
    # 5.608666131881968830e+00  1.999756583080514405e+00
    # 7.684611178255980057e-01  5.009724167888560675e-01
    return paerto_obj


pareto_para = [[],[]]
num_pareto_points = 2
tt = [(i / (num_pareto_points+1)) for i in range(num_pareto_points+1)]
tt.pop(0)
tt.pop(0)
for w in tt:
    pareto_para[0] = w / num_pareto_points
    pareto_para[1] = 1 - pareto_para[0]

    cons = (
        # total weight in range[0.8,1.0]
        {'type': 'ineq',
         'fun': lambda x: np.array([1 - (x[1] + x[2] + x[3])]),
         'jac': lambda x: np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-0.8 + (x[1] + x[2] + x[3])]),
         'jac': lambda x: np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        # total hight in range(0.9,1)
        {'type': 'ineq',
         'fun': lambda x: np.array([1 + (x[4] + x[5] + x[6] + x[7])]),
         'jac': lambda x: np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-0.9 + (x[4] + x[5] + x[6] + x[7])]),
         'jac': lambda x: np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])}
    )
    para_bound = [(0.5,1.5),(0.4,0.6),(0.4,0.6),(0.02,0.2),(0.2,0.4),(0.1,0.2),(0.1,0.2),(0.2,0.4)]
    res = minimize(obj_fun, leg_struc, args=0, method='L-BFGS-B', jac=None, bounds=para_bound, tol=1e-3, options={ 'maxiter':3000, 'disp': False})

    # import barecmaes2 as cma
    # x = cma.fmin(obj_fun, leg_struc, 0.5, args=0)

    print('optimization result:')
    print(res)
    result_obj = np.loadtxt('out.txt', delimiter='  ')
    result_para = np.loadtxt('out.txt', delimiter='  ')
    result = result_para + result_obj
    np.savetxt('result.txt', [result], delimiter='  ')
    robo.robo(1)


# fmin_l_bfgs_b(obj_fun, leg_struc, approx_grad=True, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
# os.system('python robo.py')
