import numpy as np
import pylab as py
import lib_matprop as lib_mp
import sys

dir_in = '/Users/tomotake_matsumura/Qsync/Projects/Codes_git/material_properties/data/'
#filename = 'G10_NIST/G10_NIST_thermalcod_specificheat'
filename = 'OFHCCpper_NIST_thermalcond_RRR'

out = lib_mp.read_txt_NISTdata_cupper(dir_in+filename+'.txt')

T = np.linspace(3,301,500)
delT = T[1]-T[0]

thermal_conductivity_RRR50 = lib_mp.modeleq_10powerlog_8terms(T,out[0])
thermal_conductivity_RRR100 = lib_mp.modeleq_10powerlog_8terms(T,out[1])
thermal_conductivity_RRR150 = lib_mp.modeleq_10powerlog_8terms(T,out[2])
thermal_conductivity_RRR300 = lib_mp.modeleq_10powerlog_8terms(T,out[3])
thermal_conductivity_RRR500 = lib_mp.modeleq_10powerlog_8terms(T,out[4])


WF_const = 2.45e-8 # W Ohm/K^2 k_e 


thermal_conductivity = lib_mp.modeleq_10powerlog_8terms(T,out[0])
#specific_heat = lib_mp.modeleq_10powerlog_8terms(T,out[1])
#fit_par = np.polyfit(T,specific_heat,1)
#print('fit_par')


py.subplot(121)
py.plot(T, thermal_conductivity_RRR50,label='$\\kappa$, Cu RRR50')
py.plot(T, thermal_conductivity_RRR100,label='$\\kappa$, Cu RRR100')
py.plot(T, thermal_conductivity_RRR150,label='$\\kappa$, Cu RRR150')
py.plot(T, thermal_conductivity_RRR300,label='$\\kappa$, Cu RRR300')
py.plot(T, thermal_conductivity_RRR500,label='$\\kappa$, Cu RRR500')
py.ylabel("Thermal conductivity [W/K/m]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.loglog()

py.subplot(122)
py.plot(T, 1./((WF_const*T)/thermal_conductivity_RRR50),label='elec., Cu RRR50')
py.plot(T, 1./((WF_const*T)/thermal_conductivity_RRR50),label='elec., Cu RRR100')
py.plot(T, 1./((WF_const*T)/thermal_conductivity_RRR50),label='elec., Cu RRR150')
py.plot(T, 1./((WF_const*T)/thermal_conductivity_RRR50),label='elec., Cu RRR300')
py.plot(T, 1./((WF_const*T)/thermal_conductivity_RRR50),label='elec., Cu RRR500')
py.ylabel("Electrical conductivity from Wiedemann-Franz law [1/$\\Omega$m]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.grid()
py.loglog()

py.show()

sys.exit()
