import numpy as np
import scipy.integrate as integrate

pi = np.pi
h = 6.67e-34
c = 299792458.
kb = 1.38e-23

def planck_2pibb(freq, T):
	return pi*2*h*freq**3/c**2/(np.exp(h*freq/kb/T)-1.)

def planck_bb_singlepol(freq, T):
	return h*freq**3/c**2/(np.exp(h*freq/kb/T)-1.)

def cal_BBpower_2piSA_freqi_freqf(Tin,Area,freq_i,freq_f):
	power = integrate.quad(lambda x: pi*2*h*x**3/c**2/(np.exp(((h/kb)*x)/Tin)-1.), freq_i, freq_f)
	power = power[0]*Area
	return power

def BBpower_stefanboltzmann(Tin,Area):
		return Area*Tin**4*(5.67e-8)


def cal_BBpower_singlemode_freqi_freqf(Tin,freq_i,freq_f):
	power = integrate.quad(lambda x: h*x/(np.exp(((h/kb)*x)/Tin)-1.), freq_i, freq_f)
	power = power[0]
	return power