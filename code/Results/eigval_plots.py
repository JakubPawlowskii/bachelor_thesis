import matplotlib.pyplot as plt
import numpy as np
import re
from numpy.core.fromnumeric import ptp

from numpy.core.numeric import NaN


def import_eigenvalues(L, t, d, type):

    folder_abs_path = '/home/jakubp/Code/work/ETH/qliom_all/qliom/Results/data/supp3_trans_sym/' 
    file_name = ''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0

    full_path = folder_abs_path + file_name
    print(full_path)
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
            if line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors') or line.startswith('#2 eigenvectors'):
                return np.array(data, dtype=float)
            try:
                data.append(float(line))
            except:
                continue
    print('Could not open '+full_path)
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


# L = [8,9,10,11,12,13,14]
L = [8,9,10,11,12,13,14]
delta = [0.0]

t = -0.5
types = ['s']
style = [':o',':s']
size = [7,6]
for ind3, type in enumerate(types):
    labels = []
    if type == 'f':
        for ind in range(len(delta)):
            labels.append(r'Fermion $\Delta = ' + str(delta[ind]) + r'$')
    if type == 's':
        for ind in range(len(delta)):
            labels.append( r'Spin $\Delta = ' + str(delta[ind]) + r'$')

    for ind1, d in enumerate(delta):
        eigs = []
        LL = []
        for ind2, l in enumerate(L):
            # print(l)
            try:
                # eigval = import_eigval_one_op(l, t, d, type)
                eigval = import_eigenvalues(l,t,d,type)
                # print(eigval[-1])
                eigs.append(eigval[-1])
            except:
                # print('Error')
                continue
            
            LL.append(l)
            print(l, eigs)
        inv_L = [1/float(i) for i in LL]
        x = np.array(inv_L, dtype=float)
        
        # print(p[0], p[1])
        # print(inv_L)
        pp = plt.plot(inv_L, eigs, style[ind3],markersize = size[ind3], label=labels[ind1])
        # if d == 1.0 or d == -0.5:
        #     p = np.poly1d(np.polyfit(x, np.array(eigs), 1))
        #     xx = np.linspace(0, np.max(inv_L))
        #     plt.plot(xx, p(xx),color = pp[-1].get_color(), linewidth = 0.5)
        # if d == -0.5:
        #     p = np.poly1d(np.polyfit(x, np.array(eigs), 1))
        #     xx = np.linspace(0, np.max(inv_L))
        #     plt.plot(xx, p(xx),color = pp[-1].get_color(), linewidth = 0.5)
        # if d == 1.0:
        #     inv_L = [1/float(i) for i in L]
        #     x = np.array(inv_L, dtype=float)
        #     print(eigs)
        #     p = np.poly1d(np.polyfit(x, np.array(eigs), 1))
        #     xx = np.linspace(0,np.max(inv_L))
        #     plt.plot(xx, p(xx), color=pp[-1].get_color(), linewidth=0.5)
ax = plt.gca()
ax.set_xlabel(r'$L^{-1}$')
ax.set_ylabel(r'max eigenvalue')
ax.set_xticks([1/float(i) for i in L])
xticklabels = [r'$\frac{1}{14}$', r'$\frac{1}{13}$',r'$\frac{1}{12}$',r'$\frac{1}{11}$',
               r'$\frac{1}{10}$',r'$\frac{1}{9}$', r'$\frac{1}{8}$']
# xticklabels = [r'$\frac{1}{13}$',r'$\frac{1}{11}$',r'$\frac{1}{9}$']
ax.set_xticklabels(reversed(xticklabels))
plt.title(r'$m = 3$')
plt.xlim([0, 0.2])
plt.ylim([0, 1.2])
plt.legend()
plt.show()

# test = import_eigenvalues(16,-0.5,1.0,'f')
# print(test)