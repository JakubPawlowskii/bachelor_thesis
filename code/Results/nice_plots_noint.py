from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from beautiful_latex import latex_plot
from cycler import cycler
import matplotlib.gridspec as grid
import pandas as pd
import os
import math

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
        
    data_fs = np.empty(len(L),dtype=np.ndarray)
    data_ft = np.empty(len(L),dtype=np.ndarray)
    for i,l in enumerate(L):
        fixed_spacing_name = 'data/preprocessed/' + mode + '_fixed_mat_el_spacing_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(alpha) + '.csv'
        fixed_tau_name = 'data/preprocessed/' + mode + '_fixed_tau_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(alpha) + '.csv'
        data_fs[i] = pd.read_csv(os.path.abspath(fixed_spacing_name), header=None).to_numpy()
        data_ft[i] = pd.read_csv(os.path.abspath(fixed_tau_name), header=None).to_numpy()
            
    eigval_noint = np.zeros(len(taus))
    eigval_noint = data_ft[0][idx,1]


    return eigval_int, eigval_noint

def plot_lambdas(ax,t,delta,mode,alpha,taus,taus_goal,idx):
    L = [8,9,10,11,12,13,14]
    inv_L = [1/float(i) for i in L]
    lambda_noint = np.zeros((len(L),len(taus)))
    lambda_int = np.zeros(len(L))
    for i,l in enumerate(L):
        tmp1, tmp2 = get_lambdas(t,delta,[l],mode,alpha,taus,idx)
        lambda_int[i] = tmp1
        lambda_noint[i,:] = tmp2                         
    symbols = ['o:','s:','v:','p:','^:','d:','<:','>:','x:']
    # symbols = ['x' for i in range(len(taus)+1)]
    for i in range(len(taus)):    
        label = ''
        if taus[i] < 1e6:
            label = r'$\tau = ' + r'{:.2f}'.format(taus_goal[i]) + r'$'
        else:
            label = r'$\tau = \infty $'
        pp = ax.plot(inv_L,lambda_noint[:,i],symbols[i],label=label,markersize = 7, markeredgewidth=.75, markeredgecolor='k',linewidth=0.8)
        p = np.poly1d(np.polyfit(inv_L, lambda_noint[:,i], 1)) #NaN is causing problems
        xx = np.linspace(0, np.max(inv_L))
        ax.plot(xx, p(xx),color = pp[-1].get_color(), linewidth = 0.5)

    label = r'$\tau = \infty $ INT'
    pp = ax.plot(inv_L,lambda_int,symbols[len(taus)],label=label,markersize = 7, markeredgewidth=.75,linewidth=0.8)
    p = np.poly1d(np.polyfit(inv_L,lambda_int,1))
    xx = np.linspace(0,np.max(inv_L))
    ax.plot(xx,p(xx),color=pp[-1].get_color(),linewidth=0.5)

def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


t = -0.5
taus_all = np.logspace(-1,4,300)
taus_all = np.append(taus_all,[1e+5,1e+6,1e+7,1e+8])

latex_plot(fontsize=14)
default_cycler = cycler(color=['xkcd:ocean blue','xkcd:grass green','xkcd:salmon','xkcd:fuchsia', 'xkcd:sunflower','xkcd:neon purple','xkcd:raspberry'
                               ,'xkcd:neon green','xkcd:cherry red'])
plt.rc('axes', prop_cycle=default_cycler)

ncols = 2
nrows = 2
fig = plt.figure(constrained_layout=True,figsize=(12,10))
spec = grid.GridSpec(ncols,nrows,figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ax4 = fig.add_subplot(spec[3])

axes = [ax1,ax2,ax3,ax4]

# fig.suptitle("Energy current")

markersize = 5
markevery = 1
legend_size = 12
# text_x = 0.82
# text_y = 0.4
text_x = 0.8
text_y = 0.4
markers = ['x:','o:','s:','^:','v:','d:','>:']

mode = 'ec'

L = [8,9,10,11,12,13,14]
xticks_vals = [1/float(i) for i in L]
xticks_vals.append(0.03)
xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in L]
xticklabels.append(r'$\infty \longleftarrow L$')

#===================== Left upper panel ============================================
# taus_goal = [5,7,10,20,50,100,500,1e+8] #spin
taus_goal = [5,10,20,33,50,100,500,1e+8] #ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

plot_lambdas(ax1,t,1.0,mode,0.3,taus,taus_goal,idx)

ax1.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6)
ax1.text(text_x,text_y-0.07,r'$\alpha = 0.3$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6)
ax1.set_ylabel(r'$\lambda_1$',fontsize=20)

# #================ Right upper panel ======================================
# taus_goal = [1,1.5,3,5,10,30,100,1e+8] # spin
taus_goal = [1,1.5,3,5,7,10,30,1e+8] # ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

plot_lambdas(ax2,t,1.0,mode,0.9,taus,taus_goal,idx)

ax2.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6)
ax2.text(text_x,text_y-0.07,r'$\alpha = 0.9$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6)

#================ Left lower panel ======================================
# taus_goal = [1.5,2,3,5,10,20,80,1e+8] #spin
taus_goal = [5,10,20,33,50,100,500,1e+8] #ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

plot_lambdas(ax3,t,-0.5,mode,0.3,taus,taus_goal,idx)
ax3.set_ylabel(r'$\lambda_1$',fontsize=20)
ax3.text(text_x,text_y,r'$\Delta = -0.5$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6)
ax3.text(text_x,text_y-0.07,r'$\alpha = 0.3$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6)

# ax3.set_xlim([-0.5,7])

#================ Right lower panel ======================================
# taus_goal = [1,1.5,3,5,7,20,40,1e+8] # spin
taus_goal = [1,1.5,3,5,7,10,30,1e+8] # ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]
print(taus)
plot_lambdas(ax4,t,-0.5,mode,0.9,taus,taus_goal,idx)

ax4.text(text_x,text_y,r'$\Delta = -0.5$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)
ax4.text(text_x,text_y-0.07,r'$\alpha = 0.9$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)

#========================================================================

for ax in axes:
    ax.legend(loc='best',fontsize=legend_size)
    ax.set_xlim([0.0,0.19])
    if ax == ax1 or ax == ax2:
        ax.set_ylim([-0.02,1.05])
    else:
        ax.set_ylim([-0.02,1.05])
    ax.grid(False)
    ax.tick_params(top=False,bottom=True,left=True,right=False,labeltop=False,labelbottom=True,labelright=False,labelleft=True)
    ax.set_xticks(xticks_vals)
    ax.set_xticklabels(xticklabels,fontsize=14)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].tick1line.set_visible(False)
    if ax==ax3 or ax==ax4:
        ax.set_xlabel(r'System size $1/L$',fontsize=12)


path_to_fig = 'plots/paper_plots/lambda/'
fig_name = 'lambda_energy_current_noint'
# plt.savefig(path_to_fig+fig_name+'.pdf')
# plt.savefig(path_to_fig+fig_name+'.png')
plt.show()


