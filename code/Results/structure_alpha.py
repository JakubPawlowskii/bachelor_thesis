import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

startTime = time.time()

deltas = [-0.5,1.0]
L = [8,9,10,11,12]
alphas = [0.06,0.09,0.12]

for delta in deltas:
    folderPath_mat_el = 'data/supp4_noint_mat_el/'
    # folderPath_mat_el = '../../../../../../../../media/jakubp/LaCie Drive/qliom_all/qliom/Results/data/supp3_noint_mat_el/'
    commonName_mat_el = 'spin_t_-0.5_d_' + '{:.1f}'.format(delta)        
    for l in L:
        for alpha in alphas:
            out = 'delta = '+str(delta)+', alpha = ' + str(alpha) + ', L = ' + str(l)
            print(out)
            path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(l) + '_alpha_' + '{:.2f}'.format(alpha) + '.csv'
            data_df = pd.read_csv(path_noint)
            data = data_df.to_numpy()
            diffE = np.abs(data[:,0])
            mat_el = data[:,1]
            idx = np.argsort(diffE)
            diffE = diffE[idx]
            mat_el = mat_el[idx]
            
            plt.plot(diffE/alpha,np.square(mat_el),'x', label=r'$\alpha = ' + '{:.2f}'.format(alpha) + r'$')
        
        plt.xlabel(r'$|E_n-E_m|/\alpha$')
        plt.ylabel(r'$|\langle n|O|m\rangle|^2 $')
            # plt.ylim(bottom=10**(-6))
        plt.xlim([0.0,2.0])
        plt.legend(loc='best',fontsize=12)
        tit = r'$\Delta = ' +'{:.1f}'.format(delta) + r', L = '+str(l)+r'$'
        plt.title(tit,fontsize = 12)
        filename = 'plots/supp4/mat_el_spin/abs_en/alpha_abs_en_mat_el' + '_d_'+'{:.1f}'.format(delta) + '_L_' + str(l)
        print(filename)
        plt.savefig(filename+'.png')
        plt.savefig(filename+'.pdf')
        plt.clf()
print('Execution time in seconds: ' + str(time.time()-startTime))
