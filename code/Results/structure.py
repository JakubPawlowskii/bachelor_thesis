from beautiful_latex import latex_plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.gridspec as grid

startTime = time.time()


latex_plot(scale=1.0,fontsize=20)

deltas = [1.0]
L = [10]
alphas = [0.0,0.1]
taus = [0.5]
ncols = 1
nrows = 2
fig = plt.figure(constrained_layout=True,figsize=(8,4))
spec = grid.GridSpec(ncols,nrows,figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
# ax3 = fig.add_subplot(spec[2])
# ax4 = fig.add_subplot(spec[3])

axes = [ax1,ax2]


colors = ['xkcd:red','xkcd:fuchsia', 'xkcd:sunflower']

for delta in deltas:
    for i,alpha in enumerate(alphas):
        ax = axes[i]
        folderPath_mat_el = 'data/supp3_noint_mat_el/'
        commonName_mat_el = 'spin_t_-0.5_d_' + '{:.1f}'.format(delta)
        for l in L:
            out = 'delta = '+str(delta)+', alpha = ' + str(alpha) + ', L = ' + str(l)
            print(out)
            path_noint = folderPath_mat_el + commonName_mat_el + '_L_' + str(l) + '_alpha_' + '{:.2f}'.format(alpha) + '.csv'
            data_df = pd.read_csv(path_noint)
            data = data_df.to_numpy()
            # diffE = np.abs(data[:,0])
            diffE = (data[:,0])
            mat_el = data[:,1]
            idx = np.argsort(diffE)
            diffE = diffE[idx]
            mat_el = mat_el[idx]
            
            
            ax.semilogy(diffE,np.square(mat_el),'.',markersize=0.4, color='xkcd:ocean blue')
            ax.set_xlabel(r'$E_n-E_m$',fontsize=20)
            if ax == ax1:
                ax.set_ylabel(r'$|\langle n|O|m\rangle|^2 $',fontsize=20)
            ax.set_ylim(bottom=10**(-15))
            ax.set_ylim(top=6*10**(3))
            ax.set_xlim([-6,6])
            for i ,tau in enumerate(taus):
                ax.vlines(1/tau,0,6, color = colors[i], label=r'$$\tau=' + str(tau) + r'$$',linewidth=2,zorder=6)
                ax.vlines(-1/tau,0,6, colors = colors[i], linewidth=2, zorder=6)
                ax.annotate('', xy=(1/tau, 0.001), xytext=(-1/tau+0.01, 0.001),
                        arrowprops=dict(width=2.0,facecolor='black', 
                        shrinkA=0.0,shrinkB=0.0,
                        mutation_scale=40),)
                ax.annotate('', xy=(-1/tau, 0.001), xytext=(1/tau-0.01, 0.001),
                        arrowprops=dict(width=2.0,facecolor='black', 
                        shrinkA=0.0,shrinkB=0.0,
                        mutation_scale=40),)
                ax.text(1/tau-0.2,40,r'$\frac{1}{\tau}$',fontsize=28)
                ax.text(-1/tau-0.6,40,r'-$\frac{1}{\tau}$',fontsize=28)
            # ax.set_xticklabels(fontsize=20)
            # ax.set_yticklabels(fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            txt = ''
            if ax == ax1:
                txt = '(a)'
            else:
                txt = '(b)'
            ax.text(0.9,0.92,txt,horizontalalignment='left',verticalalignment='center', transform=ax.transAxes,zorder=6, fontsize=20)
            
            if ax == ax2:
                ax.tick_params(labelleft=False)
            # tit = r'$\Delta = ' +'{:.1f}'.format(delta) + r',\; \alpha = ' + '{:.2f}'.format(alpha) + r',\; L = '+str(l)+r'$'
            # plt.title(tit,fontsize = 12)
            filename = 'plots/supp4/mat_el_spin/abs_en/abs_en_mat_el' + '_d_'+'{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(alpha) + '_L_' + str(l)
            # print(filename)
            # plt.savefig(filename+'.png')
            # plt.savefig(filename+'.pdf')
            # plt.clf()
            # plt.legend()
    plt.show()
print('Execution time in seconds: ' + str(time.time()-startTime))
