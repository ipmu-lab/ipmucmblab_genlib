import numpy as np
import pylab as py
import lib_matprop as lib_mp
import sys

dir_in = '/Users/tomotake_matsumura/Qsync/Projects/Codes_git/material_properties/data/'
#filename = 'G10_NIST/G10_NIST_thermalcod_specificheat'
filename = 'Al6061/Al6061_NIST_thermalcod_specificheat'

out = lib_mp.read_txt3f_NISTdata(dir_in+filename+'.txt')

WF_const = 2.45e-8 # W Ohm/K^2 k_e 


T = np.linspace(3,301,500)
delT = T[1]-T[0]

thermal_conductivity = lib_mp.modeleq_10powerlog_8terms(T,out[0])
specific_heat = lib_mp.modeleq_10powerlog_8terms(T,out[1])

fit_par = np.polyfit(T,specific_heat,1)
print('fit_par')

#print( '300K', lib_mp.modeleq_10powerlog_8terms(300., out[2]) )
#print( '200K', lib_mp.modeleq_10powerlog_8terms(200., out[2]) )
#print( '100K', lib_mp.modeleq_10powerlog_8terms(100., out[2]) )
#print( ' 20K', lib_mp.modeleq_10powerlog_8terms(20., out[2]) )
#print( '  4K', lib_mp.modeleq_10powerlog_8terms(4., out[2]) )


py.subplot(121)
py.plot(T, thermal_conductivity,label='Aluminum6061 from NIST data')
#py.loglog()
py.ylabel("Thermal conductivity [W/K/m]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.loglog()

py.subplot(122)
py.plot(T, (WF_const*T)/thermal_conductivity,label='Aluminum6061 from NIST data')
#py.plot(T, fit_par[0]*T+fit_par[1], label='fit: $C_p=3.16T-15.30$')
#py.loglog()
py.ylabel("Electrical resistivity from Wiedemann-Franz law [$\Omega$m]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.grid()
py.loglog()

py.show()

sys.exit()
