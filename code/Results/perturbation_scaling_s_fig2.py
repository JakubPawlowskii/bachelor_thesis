import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit as fit
from scipy.special import erf
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


def get_R1(t,delta,L,mode,alpha,taus,idx)->np.ndarray:
    R1 = np.zeros((len(alpha),len(taus)))
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

    R1 = np.zeros((len(alpha),len(taus)))
    for i in range(len(alpha)):
        for j in range(len(taus)):
            R1[i,j] = eigval_noint[i,j]/eigval_int
            if R1[i,j] > 1.0: R1[i,j] = 1.0
    
    return R1
    
def plot_R1(ax,R1,x,alpha,mode,markers):
    for i,a in enumerate(alpha):
        p = []
        if mode == 'ec':
            label = r'$\alpha = ' + r'{:.2f}'.format(a) + r'$'#, ' + r'$'\gamma = ' +r'{:.3f}'.format(popt[0])+ '$'
            ax.plot(x[i,:],R1[i,:],markers[i],markersize=markersize,label=label,markevery=markevery)
        else:
            label = r'$\alpha = ' + r'{:.2f}'.format(a) + r'$'
            ax.plot(x[i,:],R1[i,:],markers[i],markersize=markersize,label=label,markevery=markevery,markeredgecolor='k',markeredgewidth=0.2)
        ax.set_ylim([0.0,1.05])


def fun(x,gamma): 
    # x = 1/(tau*alpha^scaling)
    # return 2/np.pi * np.arctan(x/gamma)
    # return np.tanh(x/gamma)
    return erf(x/gamma)
    # return 4/np.pi * np.arctan(np.tanh(x/(2*gamma)))
    # return (x/gamma)/np.sqrt(1+(x/gamma)**2)

def fit_and_plot(ax,fun,x,R1,col,ls,lw,scaling):
    popt = fit(fun,x_flat,R1_flat)[0]
    xx = np.linspace(x_flat[0],x_flat[-1],500)
    # popt[0] = 0.4
    y = fun(xx,*popt)
    if scaling != 1:
        # label = r'$2/\pi \; \arctan{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha^{'+str(scaling)+ r'} )\right]}$'
        label = r'$\; \textrm{erf}{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha^{'+str(scaling)+ r'} )\right]}$'
        # label = r'$\; \tanh{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha^{'+str(scaling)+ r'} )\right]}$'
        # label = r'$2/\pi \;\textrm{gd}{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha^{'+str(scaling)+ r'} )\right]}$'
        # label = r'$(x/'+r'{:.3f}'.format(popt[0])+')/\sqrt{1+(x/'+r'{:.3f}'.format(popt[0])+')^2}$'
    else:
        # label = r'$2/\pi \; \arctan{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha)\right]}$'
        label = r'$\; \textrm{erf}{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha)\right])}$'
        # label = r'$\; \tanh{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha)\right]}$'
        # label = r'$2/\pi \;\textrm{gd}{\left[\omega/(' + r'{:.3f}'.format(popt[0])+ r'\alpha)\right]}$'
        # label = r'$(x/'+r'{:.3f}'.format(popt[0])+')/\sqrt{1+(x/'+r'{:.3f}'.format(popt[0])+')^2}$'

    ax.plot(xx,y,ls,color=col,linewidth=lw,label=label) 

            
#===================== Parameters ===========================================
t = -0.5
taus_all = np.logspace(-1,4,300)
taus_all = np.append(taus_all,[1e+5,1e+6,1e+7,1e+8])
# idx = (taus_all < 1000) & (taus_all > 2)  # 's' , d = -0.5
# idx = (taus_all < 1000) & (taus_all > 0.4) #'s', d = 1.0
#============================================================================

# colors = plt.cm.viridis(100)

latex_plot(scale=1, fontsize=16)
default_cycler = cycler(color=['xkcd:ocean blue','xkcd:grass green','xkcd:fuchsia', 'xkcd:sunflower','xkcd:neon purple','xkcd:raspberry','xkcd:salmon'])
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

# fig.suptitle("Noncommuting LIOM")

scaling_factors = [1.1,1.1,1.1,1.1]
# scaling_factors = [0.5,0.5,0.5,0.5]
# scaling_factors = [1,1,1,1]

markersize = 6
markevery = 1
legend_size = 20
labelssize = 24
textsize = 16
text_x = 0.77
text_y = 0.55
markers = ['o:','s:','^:','v:','d:','>:','x:']

# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
alpha = [0.05,0.1,0.2]
idx = (taus_all < 10000000) & (taus_all > 0.4) 
#===================== Left upper panel ============================================
scaling_factor = scaling_factors[0]
# alpha = [0.6]
taus = taus_all[idx]
mode = 's'

R1 = get_R1(t,1.0,[11],mode,alpha,taus,idx)
x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,2)
sl3 = slice(midpoint1+midpoint2+1,-1,5)

x = np.concatenate((x[:,sl1],x[:,sl2],x[:,sl3]),axis=1)
R1 = np.concatenate((R1[:,sl1],R1[:,sl2],R1[:,sl3]),axis=1)

x_flat = x.flatten()
R1_flat = R1.flatten()
ind = np.argsort(x_flat)
x_flat = x_flat[ind]
R1_flat = R1_flat[ind]
print(len(x_flat))
# fit_and_plot(ax1,fun,x_flat,R1_flat,'k','-',1.5)
plot_R1(ax1,R1,x,alpha,mode,markers)

ax1.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6, fontsize=textsize)
ax1.text(text_x,text_y-0.07,r'$L = 11$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6, fontsize=textsize)
ax1.set_ylabel(r'$R$',fontsize=labelssize)

# #================ Right upper panel ======================================
scaling_factor = scaling_factors[1]
# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all[idx]
mode = 's'
R1 = get_R1(t,1.0,[12],mode,alpha,taus,idx)
x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,2)
sl3 = slice(midpoint1+midpoint2+1,-1,5)

x = np.concatenate((x[:,sl1],x[:,sl2],x[:,sl3]),axis=1)
R1 = np.concatenate((R1[:,sl1],R1[:,sl2],R1[:,sl3]),axis=1)

x_flat = x.flatten()
R1_flat = R1.flatten()
ind = np.argsort(x_flat)
x_flat = x_flat[ind]
R1_flat = R1_flat[ind]

# fit_and_plot(ax2,fun,x_flat,R1_flat,'k','-',1.5)
plot_R1(ax2,R1,x,alpha,mode,markers)
# ax2.legend(loc='best',fontsize=legend_size)
ax2.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6, fontsize=textsize)
ax2.text(text_x,text_y-0.07,r'$L = 12$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6, fontsize=textsize)
# ax2.set_xlim([-0.5,7])

#================ Left lower panel ======================================

# alpha = [0.2,0.3,0.4,0.5]

scaling_factor = scaling_factors[2]
# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all[idx]
mode = 's'
R1 = get_R1(t,1.0,[13],mode,alpha,taus,idx)
x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,2)
sl3 = slice(midpoint1+midpoint2+1,-1,5)

x = np.concatenate((x[:,sl1],x[:,sl2],x[:,sl3]),axis=1)
R1 = np.concatenate((R1[:,sl1],R1[:,sl2],R1[:,sl3]),axis=1)

x_flat = x.flatten()
R1_flat = R1.flatten()
ind = np.argsort(x_flat)
x_flat = x_flat[ind]
R1_flat = R1_flat[ind]

# fit_and_plot(ax3,fun,x_flat,R1_flat,'k','-',1.5)
plot_R1(ax3,R1,x,alpha,mode,markers)

# ax3.legend(loc='best',fontsize=legend_size)

ax3.set_ylabel(r'$R$',fontsize=labelssize)
# if scaling_factor == 1:
#     ax3.set_xlabel(r'$\frac{1}{\tau \alpha}$',fontsize=20)
# elif scaling_factor == 0:
#     ax3.set_xlabel(r'$\frac{1}{\tau}$',fontsize=20)
# else:
#     ax3.set_xlabel(r'$\frac{1}{\tau \alpha^{'+str(scaling_factor)+'}}$',fontsize=20)

ax3.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6, fontsize=textsize)
ax3.text(text_x,text_y-0.07,r'$L = 13$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6, fontsize=textsize)


# ax3.set_xlim([-0.5,7])

#================ Right lower panel ======================================
scaling_factor = scaling_factors[3]
# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all[idx]
mode = 's'
R1 = get_R1(t,1.0,[14],mode,alpha,taus,idx)
x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,2)
sl3 = slice(midpoint1+midpoint2+1,-1,5)

x = np.concatenate((x[:,sl1],x[:,sl2],x[:,sl3]),axis=1)
R1 = np.concatenate((R1[:,sl1],R1[:,sl2],R1[:,sl3]),axis=1)

x_flat = x.flatten()
R1_flat = R1.flatten()
ind = np.argsort(x_flat)
x_flat = x_flat[ind]
R1_flat = R1_flat[ind]

plot_R1(ax4,R1,x,alpha,mode,markers)
fit_and_plot(ax4,fun,x_flat,R1_flat,'k','-',1.5, scaling=scaling_factor)


# ax4.legend(loc='best',fontsize=legend_size)

# if scaling_factor == 1:
#     ax4.set_xlabel(r'$\frac{1}{\tau \alpha}$',fontsize=20)
# elif scaling_factor == 0:
#     ax4.set_xlabel(r'$\frac{1}{\tau}$',fontsize=20)
# else:
#     ax4.set_xlabel(r'$\frac{1}{\tau \alpha^{'+str(scaling_factor)+'}}$',fontsize=20)

ax4.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6, fontsize=textsize)
ax4.text(text_x,text_y-0.07,r'$L = 14$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6, fontsize=textsize)
# ax4.text(text_x-0.06,text_y,r'extrap. from',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)
# ax4.text(text_x-0.06,text_y-0.05,r'$L=11,12,13,14$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6)
# ax4.set_xlim([-0.5,7])
letters = ['(a)', '(b)', '(c)', '(d)']

for i,ax in enumerate(axes):
    ax.legend(loc='lower right',fontsize=legend_size,frameon=False)
    if ax == ax1 or ax == ax2:
        ax.set_xlim([-0.1,2.5])
    else:
        ax.set_xlim([-0.1,2.5])
    ax.grid(False)
    ax.text(0.03,0.92,letters[i],transform=ax.transAxes)
    ax.tick_params(top=True,bottom=True,left=True,right=True,labeltop=False,labelbottom=True,labelright=False,labelleft=True)
    if scaling_factors[i] == 1:
        ax.set_xlabel(r'$\omega / \alpha$',fontsize=labelssize)
    elif scaling_factors[i] == 0:
        ax.set_xlabel(r'$\omega$',fontsize=labelssize)
    else:
        ax.set_xlabel(r'$\omega / \alpha^{'+str(scaling_factors[i])+'}$',fontsize=labelssize)
    ax.tick_params(axis='both', which='major', labelsize=16)


path_to_fig = 'plots/paper_plots/'
fig_name = 'fig2_small_alpha_1.1_tanh'
# plt.savefig(path_to_fig+fig_name+'.pdf')
# plt.savefig(path_to_fig+fig_name+'.png')
plt.show()


