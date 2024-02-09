import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def import_eigval(L, t, d,full_path):
  
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
            if line.startswith('#Eigenvector') or line.startswith('#5 eigenvectors'):
                return np.array(data, dtype=float)
            try:
                data.append(float(line))
            except:
                continue

def argsort_fast(x, thresh):
    idx, = np.where(x < thresh)
    return idx[np.argsort(x[idx])]


startTime = time.time()

t = -0.5
delta = -0.5
alphas = [0.06,0.09,0.12]
taus = np.linspace(1/10000,1/20,15) # actually 1/tau
taus = np.flip(taus)
L = [8,9,10,11,12,13,14]

folderPath_mat_el = '/media/jakubp/LFS/'
# folderPath_mat_el = 'data/'
commonName_mat_el = 'spin_energy_current_t_-0.5_d_' + '{:.1f}'.format(delta)

folderPath_int = '/media/jakubp/LFS/'
# folderPath_int = 'data/supp3_trans_sym/'
commonName_int = 'energy_current'

eigvals_int = np.zeros(len(L))
for i,l in enumerate(L):
    path_int = folderPath_int + commonName_int + '_L_' + str(l) + '_m_3'+'_t_'+'{:.1f}'.format(t) + '_delta_'+'{:.1f}'.format(delta)+ '_alpha_0.00'+'.dat'
    eigvals = np.array(import_eigval(l,t,delta,path_int))
    eigvals_int[i] = eigvals[-1]

inv_L = [1/i for i in L]
fit_int = np.polyfit(inv_L, eigvals_int, 1)
extrap_eigval_int = fit_int[-1]
print(extrap_eigval_int)
        
x = np.zeros((len(taus),len(alphas)))
extrap_eigval_noint = np.zeros((len(taus),len(alphas)))

for j,alpha in enumerate(alphas):
    x[:,j] = taus/(alpha**2)
    # x[:,j] = taus/(alpha)
    eigval_L_tau = np.zeros((len(L),len(taus)))
    for k,l in enumerate(L):        
        path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(l) + '_alpha_' + '{:.2f}'.format(alpha) + '.csv'
        data_df = pd.read_csv(path_noint)
        data = data_df.to_numpy()
        diffE = data[:,0]
        mat_el = data[:,1]
        abs_diffE = np.abs(diffE)
            
        for i, tau in enumerate(taus):
            out = 'alpha = ' + str(alpha) + ', tau = ' + str(1/tau) + ', L = ' + str(l)
            print(out)
            idx = argsort_fast(abs_diffE,tau)
            mat_el_sq = np.square(mat_el[idx])
            eigval_L_tau[k,i] = np.sum(mat_el_sq,dtype="float64")/2**l
            
    for i in range(len(taus)):
        fit_noint = np.polyfit(inv_L,eigval_L_tau[:,i],1)
        extrap_eigval_noint[i,j] = fit_noint[-1]
            

print(extrap_eigval_noint)
print(x)

symbols = ['x:','o:','d:']
for i in range(len(alphas)):    
    label = r'$\alpha = ' + r'{:.2f}'.format(alphas[i]) + r'$'
    data = extrap_eigval_noint[:,i]/extrap_eigval_int
    data[data>1.0] = 1.0
    plt.plot(x[:,i],data,symbols[i],label=label)

plt.xlabel(r'$\frac{1}{\tau \alpha^2}$',fontsize=16)
# plt.xlabel(r'$\frac{1}{\tau \alpha}$',fontsize=16)
plt.ylabel(r'$R_1$',fontsize=16)
plt.legend(loc='best',fontsize=12)
plt.ylim([0.0,1.05])
print('Execution time in seconds: ' + str(time.time()-startTime))
plt.tight_layout()
# filename = 'plots/R1_spin/R_1' + '_d_'+'{:.1f}'.format(delta)
filename = 'plots/supp3/energy_current/alpha_sq_R_1' + '_d_'+'{:.1f}'.format(delta)
# plt.savefig(filename+'.png')
# plt.savefig(filename+'.pdf')
plt.show()

