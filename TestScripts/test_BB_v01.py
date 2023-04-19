import numpy as np
import pylab as py
import lib_bb as lib_bb

c = 299792458.

freq_c = 150.e9
f_freq = 0.3
freq_i = freq_c *(1.-f_freq/2.)
freq_f = freq_c *(1.+f_freq/2.)
Tin = 2.725

print(freq_i*1e-9, freq_f*1e-9, Tin)
print( lib_bb.cal_BBpower_singlemode_freqi_freqf(Tin,freq_i,freq_f) )


freq0_arr = np.linspace(10e9,1000e9,1000)
freq1_arr = np.linspace(freq_i,freq_f,10000)

delta_freq = freq1_arr[1]-freq1_arr[0]
print( np.sum(lib_bb.planck_bb_singlepol(freq1_arr,Tin)*delta_freq ) )
print( np.sum(lib_bb.planck_bb_singlepol(freq1_arr,Tin)*delta_freq/freq1_arr**2*c**2 ) )

py.plot(freq0_arr*1e-9, lib_bb.planck_bb_singlepol(freq0_arr,Tin))
py.plot(freq1_arr*1e-9, lib_bb.planck_bb_singlepol(freq1_arr,Tin),'o')
py.grid()
py.loglog()
py.show()
