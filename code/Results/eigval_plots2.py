from operator import inv
import matplotlib.pyplot as plt
import numpy as np
import re
from numpy.core.numeric import NaN


def import_eigenvalues(L, t, d, type, folder_abs_path):

    file_name = ''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0

    full_path = folder_abs_path + file_name
    # print(full_path)
    with open(full_path, 'r') as f:
        # print("Opened")
        flag = False
        data = []
        for line in f:
            if line.startswith('#Correlation matrix eigenvalues'):
                flag = True
                continue
            if flag == False:
                continue
            if line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors'):
                return np.array(data, dtype=float)
            try:
                data.append(float(line))
            except:
                continue
    # print('Could not open '+full_path)
    return np.array([NaN])


def import_eigenvectors(L, t, d, type):
    folder_abs_path = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp3/'
    file_name = ''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0

    full_path = folder_abs_path + file_name

    with open(full_path, 'r') as f:
        flag = False
        data = []
        for line in f:
            if line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors'):
                flag = True
                continue
            if flag == False:
                continue
            if line.startswith('#Execution time'):
                return np.array(data, dtype=float)
            ss = re.split(r'[\s]', line.strip())
            
            ss = list(filter(None, ss))
            ss = ss[1:]
            ss = list(map(float, ss))
            
            try:
                data.append(ss)
            except:
                continue
    
    return np.array([[]])


def import_eigval_one_op(L, t, d, type):
    folder_abs_path = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp4_one_op2/'
    file_name = ''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0

    full_path = folder_abs_path + file_name
    # print(full_path)
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
            line = lines[-2]
            val = float(line)
            return val
    except:
        return NaN


L = [8,9, 10,11,12]
# L = [9,11,13]
delta = [-0.81, 0.31, 0.62]

folder_abs_path_m3 = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp3/'
folder_abs_path_m4 = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp4/'


t = -0.5
type = 's'
# style = [':o',':s']
size = [7,6]

labels_m3 = []
labels_m4 = []
# eigval_m3 = import_eigenvalues(14,t,1.0,type,folder_abs_path_m3)
# print(eigval_m3[-1])
for ind in range(len(delta)):
    labels_m3.append(r'$m=3,\;\Delta = ' + str(delta[ind]) + r'$')
    labels_m4.append(r'$m=4,\;\Delta = ' + str(delta[ind]) + r'$')
    
for ind1, d in enumerate(delta):
    eigs_m3 = []
    eigs_m4 = []
    LL = []
    for ind2, l in enumerate(L):
        # print(l)
        try:
            # eigval = import_eigval_one_op(l, t, d, type)
            eigval_m3 = import_eigenvalues(l,t,d,type,folder_abs_path_m3)
            eigs_m3.append(eigval_m3[-1])
        except:
            eigs_m3.append(NaN)
        try:
            eigval_m4 = import_eigenvalues(l,t,d,type,folder_abs_path_m4)
            eigs_m4.append(eigval_m4[-1])
        except:
            eigs_m4.append(NaN)            
        LL.append(l)
        # print(l, eigs)
    inv_L = [1/float(i) for i in LL]
    x = np.array(inv_L, dtype=float)
    
    # print(p[0], p[1])
    # print(inv_L)
    
    pp_m3 = plt.plot(inv_L, eigs_m3, 'o:',markersize = size[0], label=labels_m3[ind1])
    pp_m4 = plt.plot(inv_L, eigs_m4, 's:',markersize = size[0], label=labels_m4[ind1])
    # if d == 1.0 or d == -0.5 or d == 0.71:
    # if d == delta[0] or d == delta[1] or d == delta[2]:
        # p_m3 = np.poly1d(np.polyfit(x, np.array(eigs_m3), 1)) #NaN is causing problems
        # p_m4 = np.poly1d(np.polyfit(x[:-1], np.array(eigs_m4[:-1]), 1))
        # xx = np.linspace(0, np.max(inv_L))
        # plt.plot(xx, p_m3(xx),color = pp_m3[-1].get_color(), linewidth = 0.5)
        # plt.plot(xx, p_m4(xx),color = pp_m4[-1].get_color(), linewidth = 0.5)
ax = plt.gca()
ax.set_xlabel(r'$L^{-1}$')
ax.set_ylabel(r'max eigenvalue')
ax.set_xticks([1/float(i) for i in L])
xticklabels = [r'$\frac{1}{12}$',r'$\frac{1}{11}$',
               r'$\frac{1}{10}$',r'$\frac{1}{9}$', r'$\frac{1}{8}$']
# xticklabels = [r'$\frac{1}{13}$',r'$\frac{1}{11}$',r'$\frac{1}{9}$']
ax.set_xticklabels(reversed(xticklabels))
if type == 'f':
    plt.title(r'$Fermion$')
else:
    plt.title(r'$Spin$')
plt.xlim([0, 0.2])
plt.ylim([0, 0.5])
plt.legend()
plt.show()
