import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import matplotlib.gridspec as grid
from cycler import cycler
from beautiful_latex import latex_plot

def import_eigval(full_path) -> np.ndarray:
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
                return np.array(data, dtype=np.float64)
            try:
                data.append(float(line))
            except:
                continue
    return np.ones(1)*-1


def get_lambdas(t,delta,L,mode,alpha,taus,idx):

    inv_L = [1/i for i in L]
    eigval_int = np.zeros(len(L))
    for i,l in enumerate(L):
        int_path = 'L_' + str(l) + '_m_3_t_' + '{:.1f}'.format(t) + '_delta_' + '{:.1f}'.format(delta) + '_alpha_0.00.dat'
        if mode == 'ec':
            int_path = 'data/preprocessed/' + 'energy_current_' + int_path
        elif mode == 's':
            int_path = 'data/preprocessed/' + 'spin_' + int_path
        eigval_int[i] = import_eigval(os.path.abspath(int_path))[-1]
        
    if len(L) > 1:
        fit_int = np.polyfit(inv_L,eigval_int,1)
        eigval_int = fit_int[-1]
        if eigval_int > 1.0: eigval_int = 1.0
        if eigval_int < 0.0: eigval_int = 0.0
        print("Extrapolated integrable lambda = ",eigval_int)
    else:
        eigval_int = eigval_int[0]
        print('integrable lambda for L = '+str(L[0])+': ',eigval_int)
        
    data_fs = np.empty((len(L),len(alpha)),dtype=np.ndarray)
    data_ft = np.empty((len(L),len(alpha)),dtype=np.ndarray)
    for i,l in enumerate(L):
        for j,a in enumerate(alpha):
            fixed_spacing_name = 'data/preprocessed/' + mode + '_fixed_mat_el_spacing_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(a) + '.csv'
            fixed_tau_name = 'data/preprocessed/' + mode + '_fixed_tau_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(a) + '.csv'
            data_fs[i,j] = pd.read_csv(os.path.abspath(fixed_spacing_name), header=None).to_numpy()
            data_ft[i,j] = pd.read_csv(os.path.abspath(fixed_tau_name), header=None).to_numpy()
            
    eigval_noint = np.zeros((len(alpha),len(taus)))
    if(len(L) > 1):
        for k in range(len(alpha)):
            eigval_L_tau = np.zeros((len(L),len(taus)))
            for i,l in enumerate(L):
                eigval_L_tau[i,:] = data_ft[i,k][idx,1]
            for j in range(len(taus)):
                fit_noint = np.polyfit(inv_L,eigval_L_tau[:,j],1)
                eigval_noint[k,j] = fit_noint[-1]
                if eigval_noint[k,j] > 1.0 : eigval_noint[k,j] = 1.0
                if eigval_noint[k,j] < 0.0 : eigval_noint[k,j] = 0.0
    else:
        for k in range(len(alpha)):
            eigval_noint[k,:] = data_ft[0,k][idx,1]


    return eigval_int, eigval_noint


startTime = time.time()

t = -0.5
# deltas = [-0.5,1.0]
# alphas = [0.06,0.09,0.12]
# taus = np.array([20,33,50,100,200,1e8])
deltas = [1.0]
alphas = [0.2]
mode = 'ec'
taus_all = np.logspace(-1,4,300)
taus_all = np.append(taus_all,[1e+5,1e+6,1e+7,1e+8])


for delta in deltas:
    for alpha in alphas:
        out = 'delta = '+str(delta)+', alpha = ' + str(alpha)
        print(out)
        folderPath_mat_el = '/media/jakubp/LFS/'
        # folderPath_mat_el = 'data/'
        commonName_mat_el = 'spin_energy_current_t_-0.5_d_' + '{:.1f}'.format(delta)

        folderPath_int = '/media/jakubp/LFS/'
        # folderPath_int = 'data/supp3_trans_sym/'
        commonName_int = 'energy_current'
        
        # L = [9,11,13]
        L = [8,9,10,11,12,13,14]
        inv_L = [1/float(i) for i in L]
        eigvals_int = np.zeros(len(L))
        for i,l in enumerate(L):
            path_int = folderPath_int + commonName_int + '_L_' + str(l) + '_m_3'+'_t_'+'{:.1f}'.format(t) + '_delta_'+'{:.1f}'.format(delta)+ '_alpha_0.00'+'.dat'
            eigvals = np.array(import_eigval(l,t,delta,path_int))
            # print(eigvals)
            eigvals_int[i] = eigvals[-1]
        print(eigvals_int)

        L = [8,9,10,11,12,13,14]
        inv_L = [1/float(i) for i in L]
        eigvals_noint = np.zeros((len(L),len(taus)))
        for k,l in enumerate(L):
            path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(l) + '_alpha_' + '{:.2f}'.format(alpha) + '.csv'
            data_df = pd.read_csv(path_noint)
            data = data_df.to_numpy()
            diffE = data[:,0]
            mat_el = data[:,1]
            abs_diffE = np.abs(diffE)    
            for i, tau in enumerate(taus):
                out = 'tau = ' + str(tau) + ', L = ' + str(l)
                print(out)    
                idx = argsort_fast(abs_diffE,1/tau)
                mat_el_sq = np.square(mat_el[idx])
                eigvals_noint[k,i] = np.sum(mat_el_sq,dtype="float64")/2**l
                        


        symbols = ['x:','o:','s:','v:','p:','^:','d:','<:','>:']
        for i in range(len(taus)):    
            label = ''
            if taus[i] < 1e6:
                label = r'$\tau = ' + r'{:.2f}'.format(taus[i]) + r'$'
            else:
                label = r'$\tau = \infty $'

            pp = plt.plot(inv_L,eigvals_noint[:,i],symbols[i],label=label)
            p = np.poly1d(np.polyfit(inv_L, eigvals_noint[:,i], 1)) #NaN is causing problems
            xx = np.linspace(0, np.max(inv_L))
            plt.plot(xx, p(xx),color = pp[-1].get_color(), linewidth = 0.5)

        # L = [9,11,13]
        L = [8,9,10,11,12,13,14]
        inv_L = [1/float(i) for i in L]
        label = r'$\tau = \infty $ INT'
        pp = plt.plot(inv_L,eigvals_int,symbols[len(taus)],label=label)
        p = np.poly1d(np.polyfit(inv_L,eigvals_int,1))
        xx = np.linspace(0,np.max(inv_L))
        plt.plot(xx,p(xx),color=pp[-1].get_color(),linewidth=0.5)

        plt.xlabel(r'$\frac{1}{L}$',fontsize=14)
        plt.ylabel(r'$\lambda_1$',fontsize=14)
        plt.legend(loc='best',fontsize=10)
        tit = r'$\Delta = ' +'{:.1f}'.format(delta) + r',\; \alpha = ' + '{:.2f}'.format(alpha) + r'$'
        plt.title(tit,fontsize = 12)
        ax = plt.gca()
        
        # L = [9,11,13]
        L = [8,9,10,11,12,13,14]
        inv_L = [1/float(i) for i in L]
        
        ax.set_xticks([1/float(i) for i in L])
        xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in L]
        ax.set_xticklabels(xticklabels,fontsize=14)
        ax.set_xlim([0,0.15])
        ax.set_ylim([-0.05,1.05])
        plt.tight_layout()
        # filename = 'plots/supp3/energy_current/lambda_1' + '_d_'+'{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(alpha)
        # plt.savefig(filename+'.png')
        # plt.savefig(filename+'.pdf')
        # plt.clf()
print('Execution time in seconds: ' + str(time.time()-startTime))

# plt.show()

