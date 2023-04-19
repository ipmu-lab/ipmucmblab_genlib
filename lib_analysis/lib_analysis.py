import numpy as np
import pylab as py
import lib_optics as lib_o
from scipy.optimize import fmin

pi = np.pi
radeg = (180./pi)

def FTguess(x,data):
    del_x = x[1]-x[0]
    num = len(data)
    ftout = np.fft.fft(data)
    ind = np.where(np.array(np.abs(ftout)**2) == max(np.array(np.abs(ftout)**2)[1:]))
    ind = ind[0]
    if num%2 == 0: 
        freq = np.arange(0,num/2+1)/float(num)*2./del_x*pi
    if num%2 == 1: 
        freq = np.arange(0,num/2+1/2)/float(num)*2./del_x*pi
    freqout = freq[ind[0]]
    tmp1 = np.array((ftout[ind[0]]).real)
    tmp2 = np.array((ftout[ind[0]]).imag)
    phase = np.arctan(tmp2,tmp1)
    return freqout, phase
                   
def modeleq(p, xin):
    return p[0] + p[1]*np.cos((p[2]*xin-p[3]))

def modeleq_init(p, xin, period):
    return p[0] + p[1]*np.cos((period*xin-p[2]))

def modeleq_slope_curvefit(xin, p0,p1,p2,p3,p4):
    return p0 + p1*np.cos((p2*xin-p3)) + p4*xin
    
def modeleq_slope(p, xin):
    return p[0] + p[1]*np.cos((p[2]*xin-p[3])) + p[4]*xin
    
def optimize_modeleq_slope(p, xin, yin, sigma):
    model = modeleq_slope(p, xin)
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    chisq = ( ((model-yin)/(sigma))**2 ).sum()
    return chisq

def modeleq_slope_givenfreq(p, xin, k):
    return p[0] + p[1]*np.cos((k*xin-p[2])) + p[3]*xin
        
def optimize4leastsq_modeleq_slope_givenfreq(p, xin, yin, sigma, freq0):
#    freq0 = multi_fact*freq1[0]
    k = 4.*pi/300.*freq0
    model = modeleq_slope_givenfreq(p, xin, k)
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    return ((model-yin)/(sigma))

def optimize_modeleq_slope_givenfreq(p, xin, yin, sigma, freq0):
    k = 4.*pi/300.*freq0
    model = modeleq_slope_givenfreq(p, xin, k)
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    chisq = ( ((model-yin)/(sigma))**2 ).sum()
    return chisq

def chisq( p, xin, yin, sigma):
    return ( ((modeleq(p,xin)-yin)/(sigma))**2 ).sum()

def chisq_slope( p, xin, yin, sigma):
    return ( ((modeleq_slope(p,xin)-yin)/(sigma))**2 ).sum()

def chisq_slope_givenfreq( p, xin, yin, sigma, k):
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    return ( ((modeleq_slope_givenfreq(p,xin,k)-yin)/(sigma))**2 ).sum()

def fit_mod(p,xin,yin,sigma):
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    chisq_out = chisq(p,xin,yin,sigma)
    return chisq_out

def fit_mod_init(p,xin,yin,sigma,period):
    model = modeleq_init(p, xin, period)
    ind = np.where(sigma==0)
    if len(ind[0])!=0: sigma[ind[0]] = 0.0004
    chisq_out = ( ((model-yin)/(sigma))**2 ).sum()
    return chisq_out

def read_txt4f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4)

def read_txt5f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    filelines = fileinput.input(filename)
#    i=0
    for line in filelines:
        ar = line.split()
#        if i>0:
        ar = line.split()
#        print float(ar[0])
        arr1.append(float(ar[0]))
        arr2.append(float(ar[1]))
        arr3.append(float(ar[2]))
        arr4.append(float(ar[3]))
        arr5.append(float(ar[4]))
#        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5)

def plot_hist(x,nbin,par=-1,fit=False,init_auto=False,xtitle=-1,no_plot=False,normed=False):
    """
    plot_hist.py: plot histogram and fit with a 2D gaussian
     inputs
         x: input
         nbin: number of bin
     options
         par: initial guess of parmaeters (amp,mu,sigma)
         fit: True/False
         init_auto: True/False (auto initial guess)
         xtitle: xtitle
     output:
         fit parameters
    """
    # the histogram of the data
    non, bins, patches = py.hist(x, nbin, histtype='step', normed=normed)#, normed=1, facecolor='green', alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])

    func_gauss = lambda p, xin: p[0]*np.exp(-(xin-p[1])**2/(2.*p[2]**2))
    chi_nosigma = lambda p, xin, d: ((func_gauss(p,xin)-d)**2).sum()

    if fit: 
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '+++  Fit the histogram with Gaussian +++'
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print 'initial guess:', par0
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        par, fopt,iterout,funcalls,warnflag=fmin(chi_nosigma,par0,args=(bincenters,non),maxiter=10000,maxfun=10000,xtol=0.01,full_output=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),'r', linewidth=1)
#        if no_plot == False: py.plot(bincenters,func_gauss(par,bincenters),'r', linewidth=1)
        #y = mlab.normpdf(bincenters, par[1], par[2])
        #l = py.plot(bincenters, y, 'r--', linewidth=1)
        print 'fitted parameters:', par
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
    py.grid(True)

#    py.show()

    return np.array(par)
