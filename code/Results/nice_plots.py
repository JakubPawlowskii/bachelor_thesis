from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from beautiful_latex import latex_plot

def import_eigenvalues(L, t, d, type, folder_abs_path):

    file_name = ''

    if type == 'f':
        file_name = 'fermion_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    elif type == 's':
        file_name = 'spin_L_'+str(L)+'_t_'+str(t)+'_delta_'+str(d)+'.dat'
    else:
        print('No such type')
        return 0

    full_path = folder_abs_path + file_name
    # print(full_path)
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
            if line.startswith('#Eigenvector') or line.startswith('#10 eigenvectors') or line.startswith('#2 eigenvectors'):
                return np.array(data, dtype=float)
            try:
                data.append(float(line))
            except:
                continue
    print('Could not open '+full_path)
    return np.NaN


latex_plot()
delta1 = [-0.5,1.0]
delta2 = [0.5,-1.0]

t = -0.5

folder_abs_path_m3 = '/home/jakubp/Code/Research/ETH/qliom_all/qliom/Results/data/supp3_trans_sym/'
folder_abs_path_m4 = '/home/jakubp/Code/Research/ETH/qliom_all/qliom/Results/data/supp4_trans_sym/'

eigval_num = 1

fig = plt.figure(constrained_layout=True, figsize=(12,10))
gs = fig.add_gridspec(2, 2)

# cmap = cm.get_cmap('viridis',4)
# colors1 = cmap(range(4))
# colors = (colors1[1],colors1[3],colors1[0],colors1[2])
colors = ('xkcd:ocean blue','xkcd:grass green','xkcd:salmon','xkcd:sunflower')

symbols = [':v',':o',':s',':^']

#=================================================== Fermion subplot ======================================================
ax_fermion = plt.axes(fig.add_subplot(gs[0, :]))
ax_fermion.text(0.05, 0.9, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax_fermion.transAxes, fontsize=14)
ax_fermion.set_xlabel(r'System size $1/L$',fontsize=12)
ax_fermion.set_ylabel(r'$\lambda_{1}$',fontsize=14)

col_ind = 3

Lf = [8,9, 10,11,12,13,14,16]

labels_m3 = []
labels_m4 = []
for d in delta1:
    labels_m3.append(r'$m=3,\;\Delta = ' + str(d) + r'$')
    labels_m4.append(r'$m=4,\;\Delta = ' + str(d) + r'$')

    # labels_m3.append(r'$m=3,\;\Delta = ' + str(d) + r'('+str(-d)+r')$')
    # labels_m4.append(r'$m=4,\;\Delta = ' + str(d) + r'('+str(-d)+r')$')

for ind1, d in enumerate(delta1):
    eigs_m3 = []
    eigs_m4 = []
    LL_3 = []
    LL_4 = []
    for ind2, l in enumerate(Lf):
        try:
            data_m3 = np.array(import_eigenvalues(l,t,d,'f',folder_abs_path_m3))
            eigs_m3.append(data_m3[-eigval_num])
            LL_3.append(l)
        except:
            print('Could not find ', folder_abs_path_m3)
            continue
        try:
            data_m4 = np.array(import_eigenvalues(l,t,d,'f',folder_abs_path_m4))
            eigs_m4.append(data_m4[-eigval_num])
            LL_4.append(l)
        except:
            # print('Could not find ', folder_abs_path_m3)
            continue
   
    inv_L_3 = [1/float(i) for i in LL_3]
    inv_L_4 = [1/float(i) for i in LL_4]
    x3 = np.array(inv_L_3, dtype=float)
    x4 = np.array(inv_L_4, dtype=float)
         
    pp_m3 = plt.plot(inv_L_3, eigs_m3, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k', label=labels_m3[ind1],color=colors[col_ind], linewidth=0.8)
    col_ind = col_ind - 1
    pp_m4 = plt.plot(inv_L_4, eigs_m4, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k',label=labels_m4[ind1], color=colors[col_ind],linewidth=0.8)
    col_ind = col_ind - 1
    p_m3 = np.poly1d(np.polyfit(x3, np.array(eigs_m3), 1)) #NaN is causing problems
    p_m4 = np.poly1d(np.polyfit(x4, np.array(eigs_m4), 1))
    
    xx3 = np.linspace(0, np.max(inv_L_3))
    xx4 = np.linspace(0, np.max(inv_L_4))
    plt.plot(xx3, p_m3(xx3),color = pp_m3[-1].get_color(), linewidth = 0.5)
    plt.plot(xx4, p_m4(xx4),color = pp_m4[-1].get_color(), linewidth = 0.5)



xticks_vals = [1/float(i) for i in Lf]
xticks_vals.append(0.03)
ax_fermion.set_xticks(xticks_vals)
xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf]
xticklabels.append(r'$\infty \longleftarrow L$')
# ax_spin2.set_xticks([1/float(i) for i in Lf if not i%2])
# xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf if not i%2]
ax_fermion.set_xticklabels(xticklabels,fontsize=14)

xticks = ax_fermion.xaxis.get_major_ticks()
xticks[-1].tick1line.set_visible(False)

# ax_fermion.set_xticks([1/float(i) for i in Lf])
# xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf]
# ax_fermion.set_xticklabels(xticklabels,fontsize=14)
ax_fermion.set_xlim([0, 0.15])
ax_fermion.set_ylim([0, 0.5])

plt.legend(frameon=False, loc='lower right')

#=================================================== Spin subplot left ======================================================
ax_spin1 = plt.axes(fig.add_subplot(gs[1, 0]))
ax_spin1.text(0.1, 0.85, '(b)', horizontalalignment='center', verticalalignment='center', transform=ax_spin1.transAxes, fontsize=14)
ax_spin1.set_xlabel(r'System size $1/L$',fontsize=12)
ax_spin1.set_ylabel(r'$\lambda_{1}$',fontsize=14)

col_ind = 3

Lf = [8,9, 10,11,12,13,14]

labels_m3 = []
labels_m4 = []
for d in delta1:
    labels_m3.append(r'$m=3,\;\Delta = ' + str(d) + r'$')
    labels_m4.append(r'$m=4,\;\Delta = ' + str(d) + r'$')

for ind1, d in enumerate(delta1):
    eigs_m3 = []
    eigs_m4 = []
    LL_3 = []
    LL_4 = []
    for ind2, l in enumerate(Lf):
        try:
            data_m3 = np.array(import_eigenvalues(l,t,d,'s',folder_abs_path_m3))
            eigs_m3.append(data_m3[-eigval_num])
            LL_3.append(l)
        except:
            continue
        try:
            data_m4 = np.array(import_eigenvalues(l,t,d,'s',folder_abs_path_m4))
            eigs_m4.append(data_m4[-eigval_num])
            LL_4.append(l)
        except:
            continue
   
    inv_L_3 = [1/float(i) for i in LL_3]
    inv_L_4 = [1/float(i) for i in LL_4]
    x3 = np.array(inv_L_3, dtype=float)
    x4 = np.array(inv_L_4, dtype=float)
        
    pp_m3 = plt.plot(inv_L_3, eigs_m3, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k',label=labels_m3[ind1],color=colors[col_ind],linewidth=0.8)
    col_ind = col_ind - 1
    pp_m4 = plt.plot(inv_L_4, eigs_m4, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k', label=labels_m4[ind1], color=colors[col_ind],linewidth=0.8)
    col_ind = col_ind - 1
    p_m3 = np.poly1d(np.polyfit(x3, np.array(eigs_m3), 1)) #NaN is causing problems
    print(d,eigs_m4)
    p_m4 = np.poly1d(np.polyfit(x4, np.array(eigs_m4), 1))

    xx3 = np.linspace(0, np.max(inv_L_3))
    xx4 = np.linspace(0, np.max(inv_L_4))
    plt.plot(xx3, p_m3(xx3),color = pp_m3[-1].get_color(), linewidth = 0.5)
    plt.plot(xx4, p_m4(xx4),color = pp_m4[-1].get_color(), linewidth = 0.5)


xticks_vals = [1/float(i) for i in Lf]
xticks_vals.append(0.03)
ax_spin1.set_xticks(xticks_vals)
xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf]
xticklabels.append(r'$\infty \longleftarrow L$')
# ax_spin2.set_xticks([1/float(i) for i in Lf if not i%2])
# xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf if not i%2]
ax_spin1.set_xticklabels(xticklabels,fontsize=14)

xticks = ax_spin1.xaxis.get_major_ticks()
xticks[-1].tick1line.set_visible(False)

ax_spin1.set_xlim([0, 0.15])
ax_spin1.set_ylim([0.0, 1.05])
plt.legend(frameon=False, loc='lower left')



#=================================================== Spin subplot right ======================================================
ax_spin2 = plt.axes(fig.add_subplot(gs[1, 1]))
ax_spin2.text(0.1, 0.85, '(c)', horizontalalignment='center', verticalalignment='center', transform=ax_spin2.transAxes, fontsize=14)
ax_spin2.set_xlabel(r'System size $1/L$',fontsize=12)
ax_spin2.set_ylabel(r'$\lambda_{1}$',fontsize=14)


col_ind = 3

Lf = [8,9, 10,11,12,13,14]

labels_m3 = []
labels_m4 = []
for d in delta2:
    labels_m3.append(r'$m=3,\;\Delta = ' + str(d) + r'$')
    labels_m4.append(r'$m=4,\;\Delta = ' + str(d) + r'$')

for ind1, d in enumerate(delta2):
    eigs_m3 = []
    eigs_m4 = []
    LL_3 = []
    LL_4 = []
    for ind2, l in enumerate(Lf):
        try:
            data_m3 = np.array(import_eigenvalues(l,t,d,'s',folder_abs_path_m3))
            eigs_m3.append(data_m3[-eigval_num])
            LL_3.append(l)
        except:
            print('Could not find ', folder_abs_path_m3)
            continue
        try:
            data_m4 = np.array(import_eigenvalues(l,t,d,'s',folder_abs_path_m4))
            eigs_m4.append(data_m4[-eigval_num])
            LL_4.append(l)
        except:
            print('Could not find ', folder_abs_path_m4)
            continue
   
    inv_L_3 = [1/float(i) for i in LL_3]
    inv_L_4 = [1/float(i) for i in LL_4]
    x3 = np.array(inv_L_3, dtype=float)
    x4 = np.array(inv_L_4, dtype=float)
        
    pp_m3 = plt.plot(inv_L_3, eigs_m3, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k', label=labels_m3[ind1],color=colors[col_ind],linewidth=0.8)
    col_ind = col_ind - 1
    pp_m4 = plt.plot(inv_L_4, eigs_m4, symbols[col_ind],markersize = 7, markeredgewidth=.75, markeredgecolor='k', label=labels_m4[ind1], color=colors[col_ind],linewidth=0.8)
    col_ind = col_ind - 1
    
    p_m3_even = np.poly1d(np.polyfit(x3[0:-1:2], eigs_m3[0:-1:2], 1)) #NaN is causing problems
    p_m4_even = np.poly1d(np.polyfit(x4[0:-1:2], eigs_m4[0:-1:2], 1))
    xx3_even = np.linspace(0, np.max(inv_L_3[0:-1:2]))
    xx4_even = np.linspace(0, np.max(inv_L_4[0:-1:2]))
    plt.plot(xx3_even, p_m3_even(xx3_even),color = pp_m3[-1].get_color(), linewidth = 0.5)
    plt.plot(xx4_even, p_m4_even(xx4_even),color = pp_m4[-1].get_color(), linewidth = 0.5)

    p_m3_odd = np.poly1d(np.polyfit(x3[1:-1:2], eigs_m3[1:-1:2], 1)) #NaN is causing problems
    p_m4_odd = np.poly1d(np.polyfit(x4[1:-1:2], eigs_m4[1:-1:2], 1))
    xx3_odd = np.linspace(0, np.max(inv_L_3[1:-1:2]))
    xx4_odd = np.linspace(0, np.max(inv_L_4[1:-1:2]))
    plt.plot(xx3_odd, p_m3_odd(xx3_odd),color = pp_m3[-1].get_color(), linewidth = 0.5)
    plt.plot(xx4_odd, p_m4_odd(xx4_odd),color = pp_m4[-1].get_color(), linewidth = 0.5)

xticks_vals = [1/float(i) for i in Lf]
xticks_vals.append(0.03)
ax_spin2.set_xticks(xticks_vals)
xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf]
xticklabels.append(r'$\infty \longleftarrow L$')
# ax_spin2.set_xticks([1/float(i) for i in Lf if not i%2])
# xticklabels = [r'$\frac{1}{'+str(i)+r'}$' for i in Lf if not i%2]
ax_spin2.set_xticklabels(xticklabels,fontsize=14)

xticks = ax_spin2.xaxis.get_major_ticks()
xticks[-1].tick1line.set_visible(False)

ax_spin2.set_xlim([0, 0.15])
ax_spin2.set_ylim([0.0, 1.05])
plt.legend(frameon=False, loc='lower left')



# plt.savefig('plots/paper_plots/best_ops.pdf')
# plt.savefig('plots/paper_plots/best_ops.png')

# plt.savefig('notes/Figures/best_ops'+str(eigval_num)+r'.pdf')
# plt.savefig('notes/Figures/best_ops'+str(eigval_num)+r'.pdf')
plt.show()
