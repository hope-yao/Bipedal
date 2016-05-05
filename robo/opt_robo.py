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
# constrain:
# q1 = 0.1877
# q2 = -0.2884
# q3 = -0.2884
# c_a1 = 0.375
# c_b1 = 0.125
# c_a2 = 0.175
# c_b2 = 0.325
q1 = 0.15
q2 = -0.25
q3 = -0.25
c_a1 = 0.25
c_b1 = 0.25
c_a2 = 0.25
c_b2 = 0.25
leg_struc = np.array([q1, q2, q3, c_a1, c_b1, c_a2, c_b2])
# 0.1894 -0.2942 -0.2942 0.375 0.125 0.175 0.352
global cnt
cnt = 0

para_file = 'parameters.txt'
np.savetxt(para_file, [leg_struc], delimiter='  ')
robo.robo(1)

def obj_fun(x,arg):
    arg[0] = arg[0] + 1
    show_ani = arg[1]
    # print(">>>>>>>>>>>>>>>>>>>>>>>leg structure<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # print(x)
    para_file = 'parameters.txt'
    np.savetxt(para_file, [x], delimiter='  ')
    robo.robo(show_ani)
    out_file = 'out.txt'
    obj_val = np.loadtxt(out_file, delimiter='  ')
    # minimize negative displacement
    paerto_obj = obj_val[0] * pareto_para[0] - obj_val[1]*pareto_para[1]

    iter_info = [arg[0]] + obj_val.tolist()
    print('number of simulation:',iter_info[0])
    print('obj value:', iter_info[1:3])
    f = open('iteration.txt', 'a')
    f.write(str(iter_info)[1:-1])
    f.write('\n')
    f.close()

    # obj with initial para:
    # save for T-SNE
    result_obj = np.loadtxt('out.txt', delimiter='  ')
    result_para = np.loadtxt('parameters.txt', delimiter='  ')
    result = result_para.tolist() + result_obj.tolist()
    f = open('result.txt', 'a')
    f.write(str(result)[1:-1])
    f.write('\n')
    f.close()
    return paerto_obj



pareto_para = [[],[]]
num_pareto_points = 10
tt = [(i / (num_pareto_points+1)) for i in range(num_pareto_points+1)]
tt.pop(0)
tt = [1]
print(tt)
for w in tt:
    pareto_para[0] = w
    pareto_para[1] = 1 - w

    cons = (
        # total weight in range[0.8,1.0]
        # {'type': 'ineq',
        #  'fun': lambda x: np.array([1 - (x[1] + x[2] + x[3])]),
        #  'jac': lambda x: np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        # {'type': 'ineq',
        #  'fun': lambda x: np.array([-0.8 + (x[1] + x[2] + x[3])]),
        #  'jac': lambda x: np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        # total hight in range(0.9,1)
        {'type': 'ineq',
         'fun': lambda x: np.array([1 + (x[4] + x[5] + x[6] + x[7])]),
         'jac': lambda x: np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([-0.9 + (x[4] + x[5] + x[6] + x[7])]),
         'jac': lambda x: np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])}
    )
    para_bound = [(0.15,0.2),(-0.3,-0.2),(-0.3,-0.2),(0.2,0.4),(0.1,0.26),(0.1,0.26),(0.2,0.4)]
    res = minimize(obj_fun, leg_struc, args=[cnt,0], method='L-BFGS-B', jac=None, bounds=para_bound, tol=1e-4, options={ 'maxiter':5000, 'disp': False})

    # f = open('bestresult.txt', 'a')
    # f.write(str(result)[1:-1])
    # f.write('\n')
    # f.close()

    # import barecmaes2 as cma
    # x = cma.fmin(obj_fun, leg_struc, 0.5, args=0)

    print('optimization result:')
    print(res)
    robo.robo(1)


# fmin_l_bfgs_b(obj_fun, leg_struc, approx_grad=True, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
# os.system('python robo.py')
