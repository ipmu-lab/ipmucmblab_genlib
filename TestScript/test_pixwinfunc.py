import numpy as np
import pylab as py
import healpy as h

nside = 128
pol = True
lmax = 1024
pix_l = h.pixwin(nside, pol=False, lmax=None)

py.plot(pix_l)
py.semilogx()
py.xlim([1,1000])
py.grid()
py.show()