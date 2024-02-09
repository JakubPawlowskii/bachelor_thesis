import csv
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


def get_R1_su2(t,delta,L,mode,alpha,taus,idx)->np.ndarray:
    R1 = np.zeros((len(alpha),len(taus)))
    inv_L = [1/i for i in L]
    eigval_int = np.zeros(len(L))
    for i,l in enumerate(L):
        int_path = 'L_' + str(l) + '_m_3_t_' + '{:.1f}'.format(t) + '_delta_' + '{:.1f}'.format(delta) + '_alpha_0.00.dat'
        if mode == 'ec':
            int_path = 'data/preprocessed/su2_breaking/' + 'energy_current_' + int_path
        elif mode == 's':
            int_path = 'data/preprocessed/su2_breaking/' + 'spin1_' + int_path
        if l == 16 and delta == 0.5:
            eigval_int[i] = 0.45677
        elif l == 16 and delta == 1.0:
            eigval_int[i] = 1.0
        else:
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
            # if abs(a) == 0.05:
            #     fixed_spacing_name = 'data/preprocessed/fixed_mat_el_spacing_L_' + str(l) + '_d_' + '{:.2f}'.format(delta+a) + '_a_' + '{:.2f}'.format(0.00) + '.csv'
            #     fixed_tau_name = 'data/preprocessed/fixed_tau_L_' + str(l) + '_d_' + '{:.2f}'.format(delta+a) + '_a_' + '{:.2f}'.format(0.00) + '.csv'
            # fixed_spacing_name = 'data/preprocessed/su2_breaking/delta_'+str(delta)+'/fixed_mat_el_spacing_L_' + str(l) + '_d_' + '{:.2f}'.format(delta+a) + '_a_' + '{:.2f}'.format(0.00) + '.csv'
            fixed_tau_name = 'data/preprocessed/su2_breaking/delta_'+str(delta)+'/s_fixed_tau_L_' + str(l) + '_d_' + '{:.2f}'.format(delta+a) + '_a_' + '{:.2f}'.format(0.00) + '.csv'
            # data_fs[i,j] = pd.read_csv(os.path.abspath(fixed_spacing_name), header=None).to_numpy()
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
        if l == 16 and delta == 0.5:
            eigval_int[i] = 0.45677
        elif l == 16 and delta == 1.0:
            eigval_int[i] = 1.0
        else:
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
            # fixed_spacing_name = 'data/preprocessed/' + mode + '_fixed_mat_el_spacing_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(a) + '.csv'
            fixed_tau_name = 'data/preprocessed/' + mode + '_fixed_tau_L_' + str(l) + '_d_' + '{:.1f}'.format(delta) + '_a_' + '{:.2f}'.format(a) + '.csv'
            # data_fs[i,j] = pd.read_csv(os.path.abspath(fixed_spacing_name), header=None).to_numpy()
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
        # if a > 0:
            # marker='s:'
        # else:
            # marker='s:'
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
    xx = np.linspace(x_flat[0],x_flat[-1],5000)
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
taus_all_su2 = taus_all
# idx = (taus_all < 1000) & (taus_all > 2)  # 's' , d = -0.5
# idx = (taus_all < 1000) & (taus_all > 0.4) #'s', d = 1.0
#============================================================================

# colors = plt.cm.viridis(100)

latex_plot(scale=1,fontsize=16)
default_cycler = cycler(color=['xkcd:ocean blue','xkcd:grass green','xkcd:fuchsia', 'xkcd:sunflower','xkcd:neon purple','xkcd:bright red','xkcd:salmon','xkcd:apple green'])
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

scaling_factors = [0,2,1,1]

markersize = 6
markevery = 1
legend_size = 20
labelssize = 24
text_x = 0.6
text_y = 0.6
textsize = 16
markers = ['o:','s:','^:','v:','d:','>:','1:','x:']
# markers = ['-','-','-','-','-','-','-','-']

# alpha = [0.3,-0.3]
idx = (taus_all < 100000000) & (taus_all > 0.4) 
idx_su2 = (taus_all_su2 < 100000000) & (taus_all_su2 > 0.4) 
# idx = (taus_all < 100000000000000) & (taus_all > 0) 
# idx_su2 = (taus_all_su2 < 100000000000000) & (taus_all_su2 > 0) 


#===================== Left upper panel ============================================

alpha = [0.05,0.1,0.2]
scaling_factor = scaling_factors[0]
# alpha = [0.6]
taus = taus_all[idx]
mode = 's'

R1 = get_R1(t,1.0,[16],mode,alpha,taus,idx)
R1_1 = np.copy(R1)

x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

x1 = np.copy(x)

midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,5)
sl3 = slice(midpoint1+midpoint2+1,-1,10)

x = np.concatenate((x[:,sl1],x[:,sl2],x[:,sl3]),axis=1)
R1 = np.concatenate((R1[:,sl1],R1[:,sl2],R1[:,sl3]),axis=1)

# x_flat = x.flatten()
# R1_flat = R1.flatten()
# ind = np.argsort(x_flat)
# x_flat = x_flat[ind]
# R1_flat = R1_flat[ind]
# print(len(x_flat))
# fit_and_plot(ax1,fun,x_flat,R1_flat,'k','-',1.5)
plot_R1(ax1,R1,x,alpha,mode,markers)

ax1.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6,fontsize=textsize)
ax1.text(text_x,text_y-0.06,r'$L = 14$',horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6,fontsize=textsize)
ax1.text(text_x,text_y-0.12,r"$H' = \alpha \sum_{i=1}^L S_{i}^z S_{i+2}^z $",horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes,zorder=6,fontsize=textsize)
ax1.set_ylabel(r'$R$',fontsize=labelssize)

if scaling_factor == 1:
    ax1.set_xlabel(r'$\omega/\alpha$',fontsize=labelssize)
elif scaling_factor == 0:
    ax1.set_xlabel(r'$\omega$',fontsize=labelssize)
else:
    ax1.set_xlabel(r'$\omega / \alpha^{'+str(scaling_factor)+'}$',fontsize=labelssize)


with open('data/to_send/panel_a_delta_1.0.dat','w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(x1[0,:],x1[1,:],x1[2,:],
             R1_1[0,:],R1_1[1,:],R1_1[2,:]))



#================ Right upper panel ======================================
scaling_factor = scaling_factors[1]
# alpha = [0.2,0.3,0.4,0.5]
# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all[idx]
mode = 's'
R1 = get_R1(t,1.0,[16],mode,alpha,taus,idx)
R1_2 = np.copy(R1)

x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

x2 = np.copy(x)
with open('data/to_send/panel_b_delta_1.0.dat','w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(x2[0,:],x2[1,:],x2[2,:],
             R1_2[0,:],R1_2[1,:],R1_2[2,:]))


midpoint1 = 30
midpoint2 = midpoint1 + 20
sl1 = slice(0,midpoint1,2)
sl2 = slice(midpoint1+1,midpoint1+midpoint2,5)
sl3 = slice(midpoint1+midpoint2+1,-1,10)

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
ax2.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6,fontsize=textsize)
ax2.text(text_x,text_y-0.06,r'$L = 14$',horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6,fontsize=textsize)
ax2.text(text_x,text_y-0.12,r"$H' = \alpha \sum_{i=1}^L S_{i}^z S_{i+2}^z $",horizontalalignment='left',verticalalignment='center', transform=ax2.transAxes,zorder=6,fontsize=textsize)
# ax2.set_ylabel(r'$R_1$',fontsize=labelssize)

if scaling_factor == 1:
    ax2.set_xlabel(r'$\omega/\alpha$',fontsize=labelssize)
elif scaling_factor == 0:
    ax2.set_xlabel(r'$\omega$',fontsize=labelssize)
else:
    ax2.set_xlabel(r'$\omega / \alpha^{'+str(scaling_factor)+'}$',fontsize=labelssize)
#================================ Left lower panel =================================================

# alpha = [0.2,0.3,0.4,0.5]

scaling_factor = scaling_factors[2]
# alpha = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all[idx]
mode = 's'
R1 = get_R1(t,1.0,[16],mode,alpha,taus,idx)
R1_3 = np.copy(R1)

x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))
x3 = np.copy(x)

with open('data/to_send/panel_c_delta_1.0.dat','w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(x3[0,:],x3[1,:],x3[2,:],
             R1_3[0,:],R1_3[1,:],R1_3[2,:]))


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

plot_R1(ax3,R1,x,alpha,mode,markers)
fit_and_plot(ax3,fun,x_flat[x_flat<2],R1_flat[x_flat<2],'k','--',2.5, scaling=scaling_factor)

# ax3.legend(loc='best',fontsize=legend_size)

ax3.set_ylabel(r'$R$',fontsize=labelssize)

ax3.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6, fontsize=textsize)
ax3.text(text_x,text_y-0.06,r'$L = 14$',horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6, fontsize=textsize)
ax3.text(text_x,text_y-0.12,r"$H' = \alpha \sum_{i=1}^L S_{i}^z S_{i+2}^z $",horizontalalignment='left',verticalalignment='center', transform=ax3.transAxes,zorder=6,fontsize=textsize)

if scaling_factor == 1:
    ax3.set_xlabel(r'$\omega/\alpha$',fontsize=labelssize)
elif scaling_factor == 0:
    ax3.set_xlabel(r'$\omega$',fontsize=labelssize)
else:
    ax3.set_xlabel(r'$\omega / \alpha^{'+str(scaling_factor)+'}$',fontsize=labelssize)

#================ Right lower panel ======================================
scaling_factor = scaling_factors[3]

# alpha = [0.2,0.3,0.4,0.5]
# idx = (taus_all < 10000) & (taus_all > 1.5) 
taus = taus_all_su2[idx_su2]
mode = 's'
R1 = get_R1_su2(t,1.0,[16],mode,alpha,taus,idx_su2)
R1_4 = np.copy(R1)

x = np.zeros((len(alpha),len(taus)))
for i,a in enumerate(alpha):
    x[i,:] = 1/(taus*(np.abs(a)**scaling_factor))

x4 = np.copy(x)
with open('data/to_send/panel_d_delta_1.0.dat','w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(x4[0,:],x4[1,:],x4[2,:],
             R1_4[0,:],R1_4[1,:],R1_4[2,:]))

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
fit_and_plot(ax4,fun,x_flat[x_flat<2],R1_flat[x_flat<2],'k','--',2.5, scaling=scaling_factor)


ax4.text(text_x,text_y,r'$\Delta = 1.0$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6, fontsize=textsize)
ax4.text(text_x,text_y-0.06,r'$L = 14$',horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6, fontsize=textsize)
ax4.text(text_x,text_y-0.12,r"$H' = \alpha \sum_{i=1}^L S_{i}^z S_{i+1}^z $",horizontalalignment='left',verticalalignment='center', transform=ax4.transAxes,zorder=6,fontsize=textsize)

if scaling_factor == 1:
    ax4.set_xlabel(r'$\omega/\alpha$',fontsize=labelssize)
elif scaling_factor == 0:
    ax4.set_xlabel(r'$\omega$',fontsize=labelssize)
else:
    ax4.set_xlabel(r'$\omega / \alpha^{'+str(scaling_factor)+'}$',fontsize=labelssize)

letters = ['(a)', '(b)', '(c)', '(d)']
for i,ax in enumerate(axes):
    ax.legend(loc='lower right',fontsize=legend_size, frameon=False)
    if ax == ax1:
        ax.set_xlim([-0.01,0.6])
    elif ax == ax2:
        ax.set_xlim([-0.1,12])
    else:
        ax.set_xlim([-0.1,2])
    ax.grid(False)
    ax.text(0.03,0.92,letters[i],transform=ax.transAxes)
    ax.tick_params(top=True,bottom=True,left=True,right=True,labeltop=False,labelbottom=True,labelright=False,labelleft=True)
    # if ax == ax4 or ax == ax3:
    #     if scaling_factors[i] == 1:
    #         ax.set_xlabel(r'$\frac{1}{\tau |\alpha|}$',fontsize=labelssize)
    #     elif scaling_factors[i] == 0:
    #         ax.set_xlabel(r'$\frac{1}{\tau}$',fontsize=labelssize)
    #     else:
    #         ax.set_xlabel(r'$\frac{1}{\tau |\alpha|^{'+str(scaling_factors[i])+'}}$',fontsize=labelssize)
    ax.tick_params(axis='both', which='major', labelsize=16)

# handles, labels = ax.get_legend_handles_labels()
# labels[0] = labels[0] + " Full"
# labels[1] = labels[1] + " Full"
# labels[2] = labels[2] + " SU(2)"
# labels[3] = labels[3] + " SU(2)"
# ax.legend(handles, labels)

path_to_fig = 'data/to_send/'
fig_name = 'delta_1.0_L_16'
plt.savefig(path_to_fig+fig_name+'.pdf')
# plt.savefig(path_to_fig+fig_name+'.png')
plt.show()
