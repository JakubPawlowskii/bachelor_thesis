from cProfile import label
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
    L = [8,10,12,14]
    inv_L = [1/float(i) for i in L]
    lambda_noint = np.zeros((len(L),len(taus)))
    lambda_int = np.zeros(len(L))
    for i,l in enumerate(L):
        tmp1, tmp2 = get_lambdas(t,delta,[l],mode,alpha,taus,idx)
        lambda_int[i] = tmp1
        lambda_noint[i,:] = tmp2                         
    symbols = ['o','s','v','p','^','d','<','>','x']
    # symbols = ['x' for i in range(len(taus)+1)]
    for i in range(len(taus)):    
        label = ''   
        if taus[i] < 1e6:
            if alpha == 0.05:
                label = r'$\omega = ' + r'{:.4f}'.format(1/taus_goal[i]) + r'$'
            else:    
                label = r'$\omega = ' + r'{:.2f}'.format(1/taus_goal[i]) + r'$'
        else:
            label = r'$\omega = 0^{+}$'
        pp = ax.plot(inv_L,lambda_noint[:,i],symbols[i],label=label,markersize = 12, markeredgewidth=.75, markeredgecolor='k',linewidth=0.8)
        p = np.poly1d(np.polyfit(inv_L, lambda_noint[:,i], 1)) #NaN is causing problems
        xx = np.linspace(0, np.max(inv_L))
        ax.plot(xx, p(xx),color = pp[-1].get_color(), linewidth = 0.5)

    label = r'$\omega = 0^{+},\; \alpha = 0 $ '
    pp = ax.plot(inv_L,lambda_int,symbols[len(taus)],label=label,markersize = 12, markeredgewidth=.75,linewidth=0.8,markeredgecolor='k')
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

latex_plot(scale = 1, fontsize=20)
# default_cycler = cycler(color=['xkcd:aqua','xkcd:ocean blue','xkcd:grass green','xkcd:salmon','xkcd:sunflower','xkcd:fuchsia' ,'xkcd:raspberry'])

# default_cycler = cycler(color=['xkcd:ocean blue','xkcd:grass green','xkcd:fuchsia','xkcd:salmon', 'xkcd:sunflower','xkcd:neon purple','xkcd:raspberry'
#                                ,'xkcd:neon green','xkcd:cherry red'])
default_cycler = cycler(color=['xkcd:cherry red','xkcd:sunflower','xkcd:grass green','xkcd:raspberry','xkcd:ocean blue','xkcd:fuchsia','xkcd:salmon','xkcd:neon purple'])
plt.rc('axes', prop_cycle=default_cycler)

ncols = 1
nrows = 3
fig = plt.figure(constrained_layout=True,figsize=(18,6))
spec = grid.GridSpec(ncols,nrows,figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])

axes = [ax1,ax2,ax3]

# fig.suptitle("Energy current")

markersize = 6
markevery = 1
legend_size = 20
# text_x = 0.82
# text_y = 0.4
text_x = 0.75
text_y = 0.6
labelssize = 24
textsize = 20
markers = ['x:','o:','s:','^:','v:','d:','>:']
mode = 'ec'

L = [8,10,12,14]
# L = [8,10,12,14]
xticks_vals = [1/float(i) for i in L]
xticks_vals.append(0.03)
xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in L]
xticklabels.append(r'$\infty \longleftarrow L$')

xticks_vals.append(0.00)
xticklabels.append(r'$0$')
#===================== Left upper panel ============================================
# taus_goal = [5,7,10
# ,20,50,100,500,1e+8] #spin
taus_goal = [5,7,15,50,100,1e+8] #ec
# taus_goal = [100,200,300,1000,10000,1e+8] #ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

alpha = 0.05
plot_lambdas(ax1,t,1.0,mode,alpha,taus,taus_goal,idx)

ax1.text(0.88,0.92,r'(a)',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6, fontsize=textsize+4)
ax1.text(0.3,0.12,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6, fontsize=textsize)
ax1.text(0.309,0.12-0.07,r'$\alpha = '+'{:.2}'.format(alpha)+r'$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6)
ax1.set_ylabel(r'$\lambda$',fontsize=labelssize)

# #================ Right upper panel ======================================
# taus_goal = [1,5,7,10,30,1e+8] # spin
taus_goal = [5,7,15,50,100,1e+8] # ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

alpha = 0.3
plot_lambdas(ax2,t,1.0,mode,alpha,taus,taus_goal,idx)

ax2.text(0.88,0.92,r'(b)',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6, fontsize=textsize+4)
ax2.text(0.3,0.12,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6, fontsize=textsize)
ax2.text(0.309,0.12-0.07,r'$\alpha = '+'{:.2}'.format(alpha)+r'$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6)

#================ Left lower panel ======================================
# taus_goal = [1.5,2,3,5,10,20,80,1e+8] #spin
taus_goal = [5,7,15,50,100,1e+8] #ec
idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
taus = taus_all[idx]

alpha = 1.2
plot_lambdas(ax3,t,1.0,mode,alpha,taus,taus_goal,idx)
# ax3.set_ylabel(r'$\lambda_1$',fontsize=20)
ax3.text(0.88,0.92,r'(c)',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6, fontsize=textsize+4)
ax3.text(0.45, 0.8,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6)
ax3.text(0.459,0.8-0.07,r'$\alpha = '+'{:.2}'.format(alpha)+r'$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6)

# ax3.set_xlim([-0.5,7])

# #================ Right lower panel ======================================
# # taus_goal = [1,1.5,3,5,7,20,40,1e+8] # spin
# taus_goal = [1,1.5,3,5,7,10,30,1e+8] # ec
# idx_near = [find_nearest_idx(taus_all,i) for i in taus_goal]
# idx = [True if i in idx_near else False for i in np.arange(0,len(taus_all))]
# taus = taus_all[idx]
# print(taus)
# plot_lambdas(ax4,t,-0.5,mode,0.9,taus,taus_goal,idx)

# ax4.text(text_x,text_y,r'$\Delta = -0.5$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)
# ax4.text(text_x,text_y-0.07,r'$\alpha = 0.9$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)

#========================================================================

for ax in axes:
    if ax == ax3:
        ax.legend(loc='center left',fontsize=legend_size,frameon=False,bbox_to_anchor=(-0.05,0.6))
    else:
        ax.legend(loc='lower right',fontsize=legend_size,frameon=False,bbox_to_anchor=(1.04,0.0))        
    ax.set_xlim([0.0,0.19])
    ax.set_ylim([-0.02,1.05])

    # if ax == ax1:
    #     ax.set_ylim([-0.02,1.05])
    # elif ax == ax2:
    #     ax.set_ylim([-0.02,0.6])
    ax.grid(False)
    ax.tick_params(top=False,bottom=True,left=True,right=False,labeltop=False,labelbottom=True,labelright=False,labelleft=True)
    ax.set_xticks(xticks_vals)
    ax.set_xticklabels(xticklabels,fontsize=labelssize)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-2].tick1line.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=labelssize)
    if ax==ax1 or ax==ax2 or ax==ax3:
        ax.set_xlabel(r'System size $1/L$',fontsize=labelssize)

path_to_fig = '../../../../../../Documents/Studies/bachelor_thesis/Figures/'
fig_name = 'current_decay'
# plt.savefig(path_to_fig+fig_name+'.pdf')
plt.show()


