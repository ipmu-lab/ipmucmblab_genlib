import numpy as np
import multi_ref_birefring_funcs as lib_hwp

pi = np.pi
c = 299792458.
perm0 = (4.*pi*1.e-7)
radeg = (180./pi)

n_o = 3.07
n_e = 3.4
d_HWP = 5.e-3
phi = np.array([0.,0.])/radeg
n_AR = 1.
d_AR = 0.

num_freq = 10
freq_i = 70.e9
freq_f = 110.e9

num_rho = 360
hwp_anglemax = 360./radeg

offset_plate_trans1 = np.array([0.])/radeg

SVin = np.array([1.,1.,0.,0.])
lib_hwp.Mueller_Irho_reffreq(n_o, n_e, d_HWP, phi, n_AR, d_AR, \
	num_freq, freq_i, freq_f, num_rho, hwp_anglemax, offset_plate_trans1, SVin)

print out

#lib_hwp.multi_ref_birefringent_freqcoverage(nsa_o, nsa_e, d_HWP, phi, freq_i, freq_f, num_freq, n_AR, d_AR)

#out = lib_hwp.multibire_Iout_HWPangle(output_r_t, noI, neI, relative_angle)


