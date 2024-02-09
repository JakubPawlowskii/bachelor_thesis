import matplotlib.pyplot as plt
import numpy as np
import re
from numpy.core.fromnumeric import ptp
import sys
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
    return np.array([])

def check(L, delta, type):
    
    folder_abs_path1 = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp3_trans_sym/'
    folder_abs_path2 = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp3/'

    try:
        eig1 = import_eigenvalues(L,-0.5,delta,type,folder_abs_path1)
    except:
        print("Error opening ", folder_abs_path1)
        return False
    try:
        eig2 = import_eigenvalues(L,-0.5,delta,type,folder_abs_path2)
    except:
        print("Error opening ", folder_abs_path2)
        return False

    print("Translational symmetry : ",eig1[-1])
    print("No trans symm : ",eig2[-1])
    if abs(eig1[-1]-eig2[-1])>1e-5:
        return False
    else:
        return True

sizes = [9,11,13]
deltas = [-1.0]
types = ['s']
for type in types:
    for L in sizes:
        for delta in deltas:
            print("L = ",L,"; delta = ",delta,"; type = ",type)
            flag = check(L,delta,type)
            if flag:
                print("True")
            else:
                print("False")
            print('================================')
            
