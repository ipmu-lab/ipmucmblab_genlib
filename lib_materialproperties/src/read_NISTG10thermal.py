import numpy as np
import pylab as py
import lib_matprop as lib_mp
import sys

dir_in = '/Users/tomotake_matsumura/work/codes/material_properties/data/'
filename = 'G10_NIST/G10_NIST_thermalcod_specificheat'
#filename = 'Al6061_NIST/Al6061_NIST_thermalcod_specificheat'

out = lib_mp.read_txt4f_NISTdata(dir_in+filename+'.txt')

T = np.linspace(3,301,500)
delT = T[1]-T[0]

thermal_conductivity = lib_mp.modeleq_10powerlog_8terms(T,out[0])
thermal_conductivity_wrap = lib_mp.modeleq_10powerlog_8terms(T,out[1])
specific_heat = lib_mp.modeleq_10powerlog_8terms(T,out[2])

fit_par = np.polyfit(T,specific_heat,1)
print fit_par

print '300K', lib_mp.modeleq_10powerlog_8terms(300., out[2])
print '200K', lib_mp.modeleq_10powerlog_8terms(200., out[2])
print '100K', lib_mp.modeleq_10powerlog_8terms(100., out[2])
print ' 20K', lib_mp.modeleq_10powerlog_8terms(20., out[2])
print '  4K', lib_mp.modeleq_10powerlog_8terms(4., out[2])

N_truss = 20.
Area = np.pi * ((25e-3)**2 - (23.e-3)**2) * 0.25
print Area
print Area*N_truss
print '\n 4 - 20K'
T_c = 4
T_h = 20
ind = np.where((T<T_h) & (T>T_c))
ApL =  Area/ 0.20 * N_truss
#print np.mean(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[2])), 'J/kg.K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[1])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT/(T_h-T_c), 'W/K/m'

print '\n 20 - 100K'
T_c = 20
T_h = 100
ind = np.where((T<T_h) & (T>T_c))
ApL = Area / 0.20 * N_truss
#print np.mean(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[2])), 'J/kg.K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[1])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT/(T_h-T_c), 'W/K/m'

print '\n 100 - 200K'
T_c = 100
T_h = 200
ind = np.where((T<T_h) & (T>T_c))
ApL = Area / 0.24 * N_truss
#print np.mean(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[2])), 'J/kg.K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[1])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT/(T_h-T_c), 'W/K/m'

print '\n 200 - 300K'
T_c = 200
T_h = 300
ind = np.where((T<T_h) & (T>T_c))
ApL = Area / 0.20 * N_truss
#print np.mean(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[2])), 'J/kg.K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[1])) * delT * ApL *1e3/(T_h-T_c), 'mW/K'
print np.sum(lib_mp.modeleq_10powerlog_8terms(T[ind[0]], out[0])) * delT/(T_h-T_c), 'W/K/m'


py.subplot(211)
py.plot(T, specific_heat,label='G10 from NIST data')
py.plot(T, fit_par[0]*T+fit_par[1], label='fit: $C_p=3.16T-15.30$')
#py.loglog()
py.ylabel("Specific heat [J/kg/K]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.loglog()

py.subplot(212)
py.plot(T, thermal_conductivity,label='G10 from NIST data')
py.plot(T, thermal_conductivity_wrap,label='G10 (wrap) from NIST data')
#py.plot(T, fit_par[0]*T+fit_par[1], label='fit: $C_p=3.16T-15.30$')
#py.loglog()
py.ylabel("Thermal conductivity [W/m/K]")
py.xlabel("Temperature [K]")
py.legend(loc='best')
py.xlim(1,400)
#py.ylim(0,50)
py.grid()
py.loglog()

py.show()

sys.exit()
