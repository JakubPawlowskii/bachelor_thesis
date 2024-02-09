import pysed as sed
from shutil import copyfile
#import math as m
import os 
#import scipy.special

from math import factorial as fac

import time

def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0


fileName_spin = 'spin_edit.cpp'
fileName2_spin = 'spin_L_'
fileName_fermion = 'fermion_edit.cpp'
fileName2_fermion = 'fermion_L_'
number_of_variables = 3
variables = []

for i in range(number_of_variables):
    variables.append('var' + str(i+1))


sites = [8,9,10]   #  chain length
t = [-0.5]   #  hopping integral
#delta = [0.3090169943749475, 0.6234898018587335,0.766044443118978,0.5,1.0]   #  delta 
delta = [1.0,-0.5]
types = [fileName_spin]
types_L = [fileName2_spin]

for ind, name in enumerate(types):
    for i in range(len(sites)):
        for j in range(len(delta)):
            for k in range(len(t)):
                str_delta = "{:.2f}".format(delta[j])
                fileName3 = types_L[ind] + str(sites[i]) + "_t_" + str(t[k]) + "_delta_" + str_delta
            #  compile_command = "gfortran -g " + fileName3 + ".f" + " -o " + fileName3 +'.out'+ " -llapack"
                compile_command = "/usr/bin/g++ -g " + fileName3 + ".cpp " + " -o  " + fileName3 + ".out " + " -Ofast -larmadillo -std=c++17"
                run_command = ' '
                if 'spin' in name:
                    run_command = "nohup ./" + fileName3 + '.out' + ' > ' + 'spin_L_'+str(sites[i]) + "_t_" + str(t[k]) + "_delta_" + str_delta+ '.dat' 
                if 'fermion' in name:
                    run_command = "nohup ./" + fileName3 + '.out' + ' > ' + 'fermion_L_'+str(sites[i]) + "_t_"  + str(t[k]) + "_delta_" + str_delta+ '.dat'             
                copyfile(name,fileName3+'.cpp')
                sed.replace('var1',str(sites[i]),fileName3+'.cpp')
                sed.replace('var2',str(delta[j]),fileName3+'.cpp')
                sed.replace('var3',str(t[k]),fileName3+'.cpp')    
                t0 = time.clock()
                os.system(compile_command)
                os.system(run_command)
                os.system("rm *.out")
                os.system("rm "+fileName3 + '.cpp')
                print(fileName3+" took " + str(time.clock()-t0) + " seconds")


#            compile_command_s = "/usr/bin/g++ -g " + fileName3_fermion + ".cpp" + "-o" + fileName3_fermion + ".out" + "-Ofast -larmadillo"             
#            fileName3_spin = fileName2_spin + str(sites[i]) + "_t_" + str(t[k]) + "_delta_" + str(delta[j])
            





