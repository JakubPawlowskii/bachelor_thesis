import matplotlib.pyplot as plt
import numpy as np
import re

def import_eigenvalues(L,t,d,type):
    
    folder_abs_path = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp4/'
    file_name =''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0
    
    full_path = folder_abs_path + file_name
    print(full_path)
    with open(full_path,'r') as f:
        flag = False
        data = []
        for line in f:
            if  line.startswith('#Correlation matrix eigenvalues'):
                flag = True
                continue
            if flag == False:
                continue
            if line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors'):
                return np.array(data,dtype=float)
            try:
                data.append(float(line)) 
            except:
                continue            
    return np.array([])


def import_eigenvectors(L,t,d,type):
    folder_abs_path = '/home/jakubp/Code/work/ETH/qliom_all/results_qliom/data/supp4/'
    file_name =''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0
    
    full_path = folder_abs_path + file_name

    with open(full_path,'r') as f:
        flag = False
        data = []
        for line in f:
            if  line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors'):
                flag = True
                continue
            if flag == False:
                continue
            if line.startswith('#Execution time'):
                return np.array(data,dtype=float)
            ss = re.split(r'[\s]',line.strip())
            ss = list(filter(None,ss))
            ss = ss[1:]            
            ss = list(map(float,ss)) 
            # print(ss)
            try:
                data.append(ss) 
            except:
                continue
    return np.array([[]])

# test = import_eigenvectors(6,-0.5,-0.5,'f')
# print(test[:,i]) -- i-th eigenvector


L = [6,8,10]
delta = [-0.50]
# delta = [-0.5]
t = -0.5
type = 's' 

labels = [r'$\Delta = ' + str(delta[0]) + r'('+str(-1.0*(delta[0])) + r')$']

for ind1, d in enumerate(delta):
    eigs = []
    LL = []
    for ind2, l in enumerate(L):
        try:
            eigval = import_eigenvalues(l,t,d,type) # sorted in ascending order
            eigs.append(eigval[-2])
        except:
            continue
        LL.append(l)
    print(eigs)
    inv_L = [1/float(i) for i in LL]    
    plt.plot(inv_L,eigs,':o',label = labels[ind1] )
ax = plt.gca();
ax.set_xlabel(r'$L^{-1}$')
ax.set_ylabel(r'max eigenvalue')
if type == 'f':
    ax.set_title("Fermions")
elif type == 's':
        ax.set_title("Spins")
plt.xlim([0,0.2])
plt.ylim([0,1.4])
plt.legend()
plt.show()