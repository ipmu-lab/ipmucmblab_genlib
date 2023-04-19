import numpy as np
import pylab as py
import lib_optics as lib_o
import os
import sys

'''
 main_normalmultireflectHWP_v02.py, written by T. Matsumura in 2020-1-7.
    - set the input parameters
    - do > python main_normalmultireflectHWP_v02.py
    - generate the plot

    in [oblique_basic_multilayer_r_t] it computes the reflectance and transmittance as a function of the frequency
    in [genplot_Poleff_trans], it computes the IVA and extract the polarizaion efficiency and the phase
    	!! Note that the fit is done to a model equestion
    		in function, IVA_model2_CF(t,p1,p2,p3,p4,p5), defined in lib_optics, which is 0.5*(p1+p2*np.cos(4.*t+4.*p3)+p4*np.cos(2.*t+2.*p5) )
'''

c = 3.e8
pi = np.pi
radeg = (180./pi)

# -------------------------------------------------
# Define the basic parameters to carry out the computation in this code
n_air = 1.
nx_sample = 3.07
ny_sample = 3.40
d_sample = 5.e-3

freq_i = 70.e9
freq_f = 110.e9
freq_int = 0.5e9
angle_i = 0.*radeg
incpol = 1

nx_arr = np.array([1.,nx_sample,1.])
ny_arr = np.array([1.,ny_sample,1.])
d_arr = np.array([d_sample])

# -------------------------------------------------
# Compute the reflectance and transmittance for two different indices of refraction in oblique_basic_multilayer_r_t
RT1 = lib_o.oblique_basic_multilayer_r_t( nx_arr, d_arr, freq_i, freq_f, freq_int, angle_i, incpol)
RT2 = lib_o.oblique_basic_multilayer_r_t( ny_arr, d_arr, freq_i, freq_f, freq_int, angle_i, incpol)

# -------------------------------------------------
# Compute the followings in the genplot_Poleff_trans
# 1) intensity as a function of the HWP angle
# 2) fit the intensity vs HWP angle
# 3) extract the polarization efficiency and phase as a function of frequency

P_in = 1.
alpha_in = 45./radeg
Sin = np.array([1., P_in*np.cos(2.*alpha_in), P_in*np.sin(2.*alpha_in), 0.])

hwp_offsetangle_arr = np.array([0.])*radeg
hwp_thickness_arr = np.array([d_sample])
par = np.hstack([hwp_offsetangle_arr, hwp_thickness_arr])
freq, Poleff, phase = lib_o.genplot_Poleff_trans(par, nx_arr, ny_arr, freq_i, freq_f, freq_int, Sin, angle_i, incpol, \
	freq_c1=False, freq_c2=False, freq_c3=False, band_frachalf=False, plot=False, option_dump='')


# -------------------------------------------------
# Make a three-panel plot from the outputs
py.subplot(311)
py.plot(np.abs(RT1[0])*1e-9, np.abs(RT1[2])**2)
py.plot(np.abs(RT2[0])*1e-9, np.abs(RT2[2])**2)
py.ylim([0,1.1])

py.subplot(312)
py.plot(freq*1e-9, Poleff)
py.ylim([0.,1.1])

py.subplot(313)
py.plot(freq*1e-9, phase*radeg)

py.show()

