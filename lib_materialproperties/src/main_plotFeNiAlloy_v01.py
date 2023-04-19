import numpy as np
import pylab as py
import lib_matprop as lib_mp
import sys


def read_txt5f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=3:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5)


def convT2E_WFlaw(temperature,thermal_conductivity):
	Lorentz = 2.44e-8
	electric_conductivity = thermal_conductivity/Lorentz/temperature
	return electric_conductivity

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dir_in = '/Users/tomotake_matsumura/Qsync/Projects/Codes_git/material_properties/data/'
filename = 'Al6061/Al6061_NIST_thermalcod_specificheat'

out_al = lib_mp.read_txt3f_NISTdata(dir_in+filename+'.txt')
#print(out_al)

T = np.linspace(3,301,500)
delT = T[1]-T[0]
thermal_conductivity_Al6061 = lib_mp.modeleq_10powerlog_8terms(T,out_al[0])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dir_in = '/Users/tomotake_matsumura/Qsync/Projects/Codes_git/material_properties/data/'
filename = 'OFHCCpper_NIST_thermalcond_RRR'

out_cu = lib_mp.read_txt_NISTdata_cupper(dir_in+filename+'.txt')
#print(out_cu)

thermal_conductivity_RRR50 = lib_mp.modeleq_10powerlog_8terms(T,out_cu[0])
thermal_conductivity_RRR100 = lib_mp.modeleq_10powerlog_8terms(T,out_cu[1])
thermal_conductivity_RRR150 = lib_mp.modeleq_10powerlog_8terms(T,out_cu[2])
thermal_conductivity_RRR300 = lib_mp.modeleq_10powerlog_8terms(T,out_cu[3])
thermal_conductivity_RRR500 = lib_mp.modeleq_10powerlog_8terms(T,out_cu[4])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

filename =  '/Users/tomotake_matsumura/Qsync/Projects/Codes_git/material_properties/data/NIST_ThermalCond_FeNiAlloy.txt'
data = read_txt5f(filename)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

py.figure()
py.subplot(131)
py.plot(data[0], data[1], '-', label='Fe-2.25 Ni')
py.plot(data[0], data[2], '-', label='Fe-3.5 Ni')
py.plot(data[0], data[3], '-', label='Fe-5.0 Ni')
py.plot(data[0], data[4], '-', label='Fe-9.0 Ni')
py.plot(T, thermal_conductivity_Al6061,label='Al6061')
py.plot(T, thermal_conductivity_RRR50,label='Cu RRR50')
py.plot(T, thermal_conductivity_RRR100,label='Cu RRR100')
py.plot(T, thermal_conductivity_RRR150,label='Cu RRR150')
#py.plot(T, thermal_conductivity_RRR300,label='Cu RRR300')
py.plot(T, thermal_conductivity_RRR500,label='Cu RRR500')
py.plot([4,4],[0.1,1e4],'--')
py.plot([300,300],[0.1,1e4],'--')
py.ylabel('Thermal conductivity [W/m/K]')
py.xlabel('T [K]')
py.loglog()
py.legend(loc='best')
py.ylim([4e-1,5e2])
py.grid()

py.subplot(132)
py.plot(data[0], convT2E_WFlaw(data[0],data[1]), '-', label='Fe-2.25 Ni')
py.plot(data[0], convT2E_WFlaw(data[0],data[2]), '-', label='Fe-3.5 Ni')
py.plot(data[0], convT2E_WFlaw(data[0],data[3]), '-', label='Fe-5.0 Ni')
py.plot(data[0], convT2E_WFlaw(data[0],data[4]), '-', label='Fe-9.0 Ni')
py.plot(T, convT2E_WFlaw(T, thermal_conductivity_Al6061),label='Al6061')
py.plot(T, convT2E_WFlaw(T,thermal_conductivity_RRR50),label='Cu RRR50')
py.plot(T, convT2E_WFlaw(T,thermal_conductivity_RRR100),label='Cu RRR100')
py.plot(T, convT2E_WFlaw(T,thermal_conductivity_RRR150),label='Cu RRR150')
#py.plot(T, convT2E_WFlaw(T,thermal_conductivity_RRR300),label='Cu RRR300')
py.plot(T, convT2E_WFlaw(T,thermal_conductivity_RRR500),label='Cu RRR500')
py.plot([4,4],[1e5,1e10],'--')
py.plot([300,300],[1e5,1e10],'--')
py.ylabel('Electric conductivity [1./Ohm.m]')
py.xlabel('T [K]')
py.loglog()
py.legend(loc='best')
py.ylim([5e5,5e9])
py.grid()


py.subplot(133)
py.plot(data[0], 1./convT2E_WFlaw(data[0],data[1]), '-', label='Fe-2.25 Ni')
py.plot(data[0], 1./convT2E_WFlaw(data[0],data[2]), '-', label='Fe-3.5 Ni')
py.plot(data[0], 1./convT2E_WFlaw(data[0],data[3]), '-', label='Fe-5.0 Ni')
py.plot(data[0], 1./convT2E_WFlaw(data[0],data[4]), '-', label='Fe-9.0 Ni')
py.plot(T, 1./convT2E_WFlaw(T, thermal_conductivity_Al6061),label='Al6061')
py.plot(T, 1./convT2E_WFlaw(T,thermal_conductivity_RRR50),label='Cu RRR50')
py.plot(T, 1./convT2E_WFlaw(T,thermal_conductivity_RRR100),label='Cu RRR100')
py.plot(T, 1./convT2E_WFlaw(T,thermal_conductivity_RRR150),label='Cu RRR150')
#py.plot(T, 1./convT2E_WFlaw(T,thermal_conductivity_RRR300),label='Cu RRR300')
py.plot(T, 1./convT2E_WFlaw(T,thermal_conductivity_RRR500),label='Cu RRR500')
py.plot([4,4],[2e-10,4e-6],'--')
py.plot([300,300],[2e-10,4e-6],'--')
py.ylabel('Electric resistivity [Ohm.m]')
py.xlabel('T [K]')
py.loglog()
py.legend(loc='best')
py.ylim([2e-10,4e-6])
py.grid()

py.show()

