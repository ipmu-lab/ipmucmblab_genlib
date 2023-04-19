import numpy as np
import pylab as py
import lib_m as lib_m


def test_AppFilt():

    num = 1000
    noise = np.random.normal(0.,1.,num)
    d_time = 1.e-3
    samplerate  = 1./d_time
    time = np.arange(num)*d_time

#    filtertype = 'sp_lowpass'
#    par = [50.]
    filtertype = 'cosine_lowpass'
    par = [20.,20]

    freq, psd, psd_filt, noise_filt, filt = lib_m.AppFilt(noise, samplerate, 6, filtertype, par)

    py.subplot(211)
    py.plot(time, noise)
    py.plot(time, noise_filt)
    py.subplot(212)
    py.plot(freq, psd)
    py.plot(freq, psd_filt)
    py.plot(freq, filt)
    py.loglog()
    py.show()

def test_cosine_lowpass():
    num = 100
    freq = np.arange(num)
    freq_0 = 20.
    freq_w = 10.
    tf = lib_m.cosine_lowpass(freq,freq_0,freq_w)
    py.plot(freq,tf)
    py.show()

def test_cosine_highpass():
    num = 100
    freq = np.arange(num)
    freq_0 = 20.
    freq_w = 10.
    tf = lib_m.cosine_highpass(freq,freq_0,freq_w)
    py.plot(freq,tf)
    py.show()

def test_cosine_bandpass():
    num = 100
    freq = np.arange(num)
    freq_l = 40.
    freq_wl = 10.
    freq_h = 10.
    freq_wh = 10.
    tf = lib_m.cosine_bandpass(freq,freq_l,freq_wl,freq_h,freq_wh)
    py.plot(freq,tf,'.')
    py.show()

#test_AppFilt()
#test_cosine_lowpass()
#test_cosine_highpass()
test_cosine_bandpass()
