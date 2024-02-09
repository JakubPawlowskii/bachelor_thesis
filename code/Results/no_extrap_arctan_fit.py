import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import curve_fit as fit

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

def fun(x,gamma): 
    # x = 1/(tau*alpha^2)
    return 2/np.pi * np.arctan(x/gamma)


startTime = time.time()

t = -0.5
delta = 1.0
alphas = [0.06,0.09,0.12]
taus = np.linspace(1/10000,1/20,20) # actually 1/tau
taus = np.flip(taus)
LL = [8,9,10,11,12,13,14]

folderPath_mat_el = '/media/jakubp/LFS/'
# folderPath_mat_el = 'data/'
commonName_mat_el = 'spin_energy_current_t_-0.5_d_' + '{:.1f}'.format(delta)

folderPath_int = '/media/jakubp/LFS/'
# folderPath_int = 'data/supp3_trans_sym/'
commonName_int = 'energy_current'


for L in LL:
    path_int = folderPath_int + commonName_int + '_L_' + str(L) + '_m_3'+'_t_'+'{:.1f}'.format(t) + '_delta_'+'{:.1f}'.format(delta)+ '_alpha_0.00'+'.dat'
    eigvals = np.array(import_eigval(L,t,delta,path_int))
    eigval_int = eigvals[-1]

    # print(eigval_int)
            
    x = np.zeros((len(taus),len(alphas)))
    eigval_noint = np.zeros((len(taus),len(alphas)))

    for j,alpha in enumerate(alphas):
        x[:,j] = taus/(alpha**2)
        eigval_tau = np.zeros((len(taus)))
        path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(L) + '_alpha_' + '{:.2f}'.format(alpha) + '.csv'
        data_df = pd.read_csv(path_noint)
        data = data_df.to_numpy()
        diffE = data[:,0]
        mat_el = data[:,1]
        abs_diffE = np.abs(diffE)
                
        for i, tau in enumerate(taus):
            out = 'alpha = ' + str(alpha) + ', tau = ' + str(1/tau) + ', L = ' + str(L)
            print(out)
            idx = argsort_fast(abs_diffE,tau)
            mat_el_sq = np.square(mat_el[idx])
            eigval_noint[i,j] = np.sum(mat_el_sq,dtype="float64")/2**L
                
                
    # print(eigval_noint)

    symbols = ['x:','o:','d:']
    for i in range(len(alphas)):    
        data = eigval_noint[:,i]/eigval_int
        data[data>1.0] = 1.0
        popt = fit(fun,x[:,i],data)[0]
        label = r'$\alpha = ' + r'{:.2f}'.format(alphas[i]) + r'$, ' + r'$\gamma = ' +r'{:.3f}'.format(popt[0])+ '$'
        p = plt.plot(x[:,i],data,symbols[i],label=label)
    
        xx = np.linspace(x[0,i],x[-1,i],100)
        y = fun(xx,*popt)
        plt.plot(xx,y,'-',color=p[-1].get_color(),linewidth=1.0)

        # label = r'$\alpha = ' + r'{:.2f}'.format(alphas[i]) + r'$'
        # plt.plot(x[:,i],eigval_noint[:,i]/eigval_int,symbols[i],label=label)

    plt.xlabel(r'$\frac{1}{\tau \alpha^2}$',fontsize=16)
    plt.ylabel(r'$R_1$',fontsize=16)
    plt.legend(loc='best',fontsize=12)
    plt.ylim([0.0,1.05])
    plt.tight_layout()
    filename = 'plots/supp3/energy_current/fit_no_ext_R_1' + '_d_'+'{:.1f}'.format(delta) + '_L_'+str(L)
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.pdf')
    plt.clf()
print('Execution time in seconds: ' + str(time.time()-startTime))

# plt.show()

