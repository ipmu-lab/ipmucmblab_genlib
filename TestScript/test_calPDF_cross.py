import numpy as np
import pylab as py
import lib_m as lib_m


# define a constant parameter
pi = np.pi

# define inputs
omega1 = 2.*pi*1.
omega2 = 2.*pi*2.
omega3 = 2.*pi*2.
omega4 = 2.*pi*3.

samplerate = 50.

# generate the time
time = np.arange(0,10,1./samplerate)
num = len(time)

# generate the fake data, data1 and data2
data1 = np.cos(omega1*time) + np.cos(omega2*time) + np.random.normal(0,1,num)
data2 = np.cos(omega3*time) + np.cos(omega4*time) + np.random.normal(0,1,num)

# compute the psd and cross-psd
#   note that cross spectrum can be negative and so we don't take a square
psd1 = lib_m.calPSD(data1, samplerate, 3)
psd2 = lib_m.calPSD(data2, samplerate, 3)
psd3 = lib_m.calPSD_cross(data1,data2, samplerate, 3)

# plot
py.subplot(311)
py.plot(psd1[0],psd1[1])
py.loglog()
py.ylabel('psd(data1)')

py.subplot(312)
py.plot(psd2[0],psd2[1])
py.loglog()
py.ylabel('psd(data2)')

py.subplot(313)
py.plot(psd3[0],psd3[1])
py.semilogx()
py.ylabel('cross-psd(data1,data2)')

py.show()