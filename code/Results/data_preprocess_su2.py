import numpy as np
from numba import jit
import pandas as pd
import time
import csv

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
                return data
            try:
                data.append(float(line))
            except:
                continue

def argsort_fast(x, thresh):
    idx, = np.where(x < thresh)
    return idx[np.argsort(x[idx])]

@jit(nopython=True)
def fixed_mat_el(abs_diffE,mat_el,L):
    sum_mat_el_tmp = 0
    thres = 0.01
    en = []
    mes = []
    for i, dE in enumerate(abs_diffE):
       # print('dE = ',abs_diffE[i])
        if sum_mat_el_tmp >= thres-1e-7:
            en.append(dE)
            mes.append(sum_mat_el_tmp)
            thres = thres + 0.01
        sum_mat_el_tmp = sum_mat_el_tmp + mat_el[i]**2/2**L
         
    return en,mes



startTime = time.time()

folderPath_mat_el = '/mnt/HDD2/kuba/Source/Spin/su2_breaking/'
folderPath_int = '/mnt/HDD2/kuba/Source/Spin/su2_breaking/'
commonName_int = 'spin'

t = -0.5
alphas = [0.95,0.9,0.8,0.7,0.6,0.5,1.05,1.1,1.2,1.3,1.4,1.5]
deltas = [1.0]
taus = np.logspace(-1,4,300)
LL = [8,9,10,11,12,13,14]

for delta in deltas:
    for L in LL:
        path_int = folderPath_int + commonName_int + '1_L_' + str(L) + '_m_3'+'_t_'+'{:.1f}'.format(t) + '_delta_'+'{:.1f}'.format(delta)+ '_alpha_0.00'+'.dat'
        eigvals = np.array(import_eigval(L,t,delta,path_int),dtype="float64")
        eigval_int = eigvals[-1]

        for j,alpha in enumerate(alphas):
            if alpha == 0.95 or alpha == 1.05:
                commonName_mat_el = 'spin_t_-0.5_d_' + '{:.2f}'.format(alpha)
            commonName_mat_el = 'spin_t_-0.5_d_' + '{:.1f}'.format(alpha)
            sum_mat_el_taus = np.zeros(len(taus))
            path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(L) + '_alpha_' + '{:.2f}'.format(0.00) + '.csv'
            data_df = pd.read_csv(path_noint)
            data = data_df.to_numpy()
            diffE = data[:,0]
            mat_el = data[:,1]
            abs_diffE = np.abs(diffE)
            # print('max diff E', np.max(abs_diffE))
            idx = np.argsort(abs_diffE)
            mat_el = mat_el[idx]
            abs_diffE = abs_diffE[idx]
            
            for i, tau in enumerate(taus):
                out = 'delta = ' + str(alpha) + ', tau = ' + str(tau) + ', L = ' + str(L)
                print(out)                    
                idx = np.where(abs_diffE < 1/tau)
                mat_el_sq = np.square(mat_el[idx])
                sum_mat_el_taus[i] = np.sum(mat_el_sq,dtype="float64")/2**L
            
            with open('/mnt/HDD2/kuba/Source/preprocessed/su2_breaking/fixed_tau_L_' + str(L) + '_d_' 
                      + '{:.2f}'.format(alpha) + '_a_' + '{:.2f}'.format(0.0) + '.csv','w') as f:
                writer = csv.writer(f,delimiter=',')
                writer.writerows(zip(taus,sum_mat_el_taus))
            
            en,mes = fixed_mat_el(abs_diffE,mat_el,L)
            with open('/mnt/HDD2/kuba/Source/preprocessed/su2_breaking/fixed_mat_el_spacing_L_' + str(L) + '_d_'
                      + '{:.2f}'.format(alpha) + '_a_' + '{:.2f}'.format(0.0) + '.csv','w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(zip(en,mes))

print('Execution time in seconds: ' + str(time.time()-startTime))

# plt.show()

