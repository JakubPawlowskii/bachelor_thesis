import os
import subprocess
from datetime import datetime as dt

filename = 'energy_current'

alphas = [0.0]
ts = [-0.5]
deltas = [1.0, -0.5]
sitess = [8,9,10]
ms = [3]
trans_symm = True
flag = False
commutes = True
flip = False
save_op = 1

compile_command = "/usr/bin/g++ -g " + filename + ".cpp  ../Includes/combinadics.cpp ../Includes/xxz_real.cpp"  + " -o  " + filename + ".out " + " -Ofast -larmadillo -std=c++17 -march=native -fopenmp"
os.system(compile_command)

for alpha in alphas:
    for t in ts:
        for delta in deltas:
            for sites in sitess:
                for m in ms:
                    str_delta = "{:.1f}".format(delta)
                    str_alpha = "{:.2f}".format(alpha)
                    str_t = "{:.1f}".format(t)
                    fileName3 = filename + "_L_" + str(sites) + "_m_" + str(m) + "_t_" + str_t + "_delta_" + str_delta + "_alpha_" + str_alpha

                    run_command = 'nohup ./' + filename + '.out' + ' -s ' + str(sites) + ' -m ' + str(m) + ' -t ' + str_t + ' -d ' + str_delta + ' -a ' + str_alpha 
                    if(commutes):
                        run_command = run_command + ' -C 1'
                    else:
                        run_command = run_command + ' -C 0'
                    
                    if(flip):
                        run_command = run_command + ' -F 1'
                    else:
                        run_command = run_command + ' -F 0'

                    if(trans_symm):
                        run_command = run_command + ' -T 1'
                    else:
                        run_command = run_command + ' -T 0'

                    if(flag):
                        run_command = run_command + ' -G 1'
                    else:
                        run_command = run_command + ' -G 0'

                    if(save_op != 0):
                        run_command = run_command + ' -S ' + str(save_op)

                    run_command = run_command + ' > ' + fileName3 + '.dat'
                    now = dt.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") 
                    print(run_command + ' --- ' + dt_string) 
                    os.system(run_command)


#            compile_command_s = "/usr/bin/g++ -g " + fileName3_fermion + ".cpp" + "-o" + fileName3_fermion + ".out" + "-Ofast -larmadillo"             
#            fileName3_spin = fileName2_spin + str(sites[i]) + "_t_" + str(t[k]) + "_delta_" + str(delta[j])
            


