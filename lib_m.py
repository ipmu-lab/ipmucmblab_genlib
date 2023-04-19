import numpy as np
import pylab as py
import healpy as h
import ephem as ephem
#import slalib as sla
from pyslalib import slalib as sla
from pylab import errorbar
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.optimize import fmin
import numpy.fft as fft
import nfft as nf
import math as m
import sys
import os

pi = np.pi
r2d = (180./pi)
d2r = (pi/180.)
radeg = r2d

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# SYSTEM related
def len_all(a):
    a = np.array(a)
    b = a/a
    c = np.sum(b)
    if c == 1: nb = 1
    if c > 1: nb = len(a)
    return nb

def help_var(var,var_name):
    if isinstance(var,ndarray):
            print( '   [help_var] ', var_name, '->', type(var), var.shape, var.dtype.name )
    if isinstance(var,tuple):
            print( '   [help_var] ', var_name, '->', type(var), len(var), ',', len(var[0]) )
    if isinstance(var,str):
            print( '   [help_var] ', var_name, '->', type(var), len(var) )

def pidinfo(title, runtime_init):
    print( '' )
    print( "________________________________________" )
    print( title )
    print( '  module name:', __name__ )
    print( '  parent process:', os.getppid() )
    print( '  process id:', os.getpid()  )
    print( '  run time: ', time.time()-runtime_init )
    print( "________________________________________" )
    print( '' )
#######################################################

# FILE I/O related
def read_txt2i(filename):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(int(ar[0]))
            arr2.append(int(ar[1]))
        i+=1
    return np.array(arr1), np.array(arr2)

def read_txt2l(filename):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(long(ar[0]))
            arr2.append(long(ar[1]))
        i+=1
    return np.array(arr1), np.array(arr2)

def read_txt1f(filename):
    import fileinput
    arr1 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
        i+=1
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1)

def read_txt2f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
        i+=1
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2)

def read_txt3f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
        i+=1
    return np.array(arr1), np.array(arr2), np.array(arr3)

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
    return np.array(arr1), np.array(arr2), np.array(arr3), np.array(arr4)

def read_txt5f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5)

def read_txt7f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
            arr6.append(float(ar[5]))
            arr7.append(float(ar[6]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5),np.array(arr6),np.array(arr7)

def read_txt11f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []
    arr11 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=0:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
            arr6.append(float(ar[5]))
            arr7.append(float(ar[6]))
            arr8.append(float(ar[7]))
            arr9.append(float(ar[8]))
            arr10.append(float(ar[9]))
            arr11.append(float(ar[10]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5),np.array(arr6),np.array(arr7),np.array(arr8),np.array(arr9),np.array(arr10),np.array(arr11)

def read_txt4f_uptoN(filename, n):
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
        if i>n: break
    return np.array(arr1), np.array(arr2), np.array(arr3), np.array(arr4)
   
def read_txt2f_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
        i+=1
#        if i == 30000: break
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2)

def read_txt2f_skipn_uptoN(filename,n, m):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
        i+=1
        if i>m: break
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2)

def read_txt2f_cvs_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if (i>=n):
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
        i+=1
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2)

def read_txt3f_cvs_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
#        print( i, n, line )
        if (i>=n):
#            print( i, n, line )
            ar = line.split(",")
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
        i+=1
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2), np.array(arr3)

def read_txt3f_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
        i+=1
#        if i == 30000: break
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2), np.array(arr3)

def read_txt4f_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        print( i, line )
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
        i+=1
#        if i == 30000: break
    print( 'read from ' )
    print( '   ', filename )
    print( '    num= ', len(arr1) )
    return np.array(arr1), np.array(arr2), np.array(arr3), np.array(arr4)

def read_txt9f_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
            arr6.append(float(ar[5]))
            arr7.append(float(ar[6]))
            arr8.append(float(ar[7]))
            arr9.append(float(ar[8]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5), \
        np.array(arr6),np.array(arr7),np.array(arr8),np.array(arr9)

def read_txt11f_skipn(filename,n):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []
    arr11 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=n:
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
            arr6.append(float(ar[5]))
            arr7.append(float(ar[6]))
            arr8.append(float(ar[7]))
            arr9.append(float(ar[8]))
            arr10.append(float(ar[9]))
            arr11.append(float(ar[10]))
        i+=1
    return np.array(arr1),np.array(arr2),np.array(arr3),np.array(arr4),np.array(arr5), \
        np.array(arr6),np.array(arr7),np.array(arr8),np.array(arr9),np.array(arr10),np.array(arr11)

def write_txt5f(fname,array1,array2,array3,array4,array5):
    nb = len(array1)
    f = open(fname, "w")
    for i in range(0,nb):
        f.write('%f %f %f %f %f\n'
                % (array1[i],array2[i],array3[i],array4[i],array5[i]))
    print( '[WRITE TEXT FILE]: ', fname )
    f.close()

def write_txt4f(fname,array1,array2,array3,array4):
    nb = len(array1)
    f = open(fname, "w")
    for i in range(0,nb):
        f.write('%f %f %f %f\n' % (array1[i],array2[i],array3[i],array4[i]))
    print( '[WRITE TEXT FILE]: ', fname )
    f.close()

def write_txt3f(fname,array1,array2,array3):
    nb = len(array1)
    f = open(fname, "w")
    for i in range(0,nb):
        f.write('%f %f %f \n' % (array1[i],array2[i],array3[i]))
    print( '[WRITE TEXT FILE]: ', fname )
    f.close()

def write_txt2f(fname,array1,array2):
    nb = len(array1)
    f = open(fname, "w")
    for i in range(0,nb):
        f.write('%f %f \n' % (array1[i], array2[i]))
    print( '[WRITE TEXT FILE]: ', fname )
    f.close()

def write_txtf(fname,array):
    nb = len(array)
    f = open(fname, "w")
    for i in range(0,nb):
        f.write('%f \n' % (array[i]))
    f.close()

#
def open_png(xscale,yscale):
    #xscale=2.
    #yscale=2.
    F = gcf()
    DPI = F.get_dpi()
    DefaultSize = F.get_size_inches()
    F.set_size_inches( (DefaultSize[0]*xscale, DefaultSize[1]*yscale) )
    return DPI

def value_list(x):
    if isinstance(x, dict):
        return list(set(x.values()))
    elif isinstance(x, basestring):
        return [x]
    else:
        return None

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def wraparound_npi(x,n_):
    n_ = np.float(n_)
    if len(x)==1: 
        n_wrap = np.int(x/(n_*pi))
        if x<0: n_wrap-=1 
    if len(x)>1: 
        n_wrap = np.int_(x/(n_*pi))
        ind = np.where((x<0)) 
        if len(ind[0]) != 0: n_wrap[ind[0]]-=1
    return x-n_wrap*n_*pi 
#--------------------------------------------------------------------------------------------------------
def wraparound_2pi(x):
    if len(x)==1: 
        n = np.int(x/(2.*pi))
        if x<0: n-=1 
    if len(x)>1: 
        n = np.int_(x/(2.*pi))
        ind = np.where((x<0)) 
        if len(ind[0]) != 0: n[ind[0]]-=1
    return x-n*2.*pi 
#--------------------------------------------------------------------------------------------------------
def wraparound_pi(x):
    if len(x)==1: 
        n = np.int(x/(pi))
        if (x<0): n-=1 
    if len(x)>1: 
        n = np.int_(x/(pi))
        ind = np.where((x<0)) 
        if len(ind[0]) != 0: n[ind[0]]-=1
    return x-n*pi 
#--------------------------------------------------------------------------------------------------------
def wraparound_halfpi(x):
    if len(x)==1: 
        n = np.int(x/(pi/2.))
        if (x<0): n-=1 
    if len(x)>1: 
        n = np.int_(x/(pi/2.))
        ind = np.where((x<0)) 
        if len(ind[0]) != 0: n[ind[0]]-=1
    return x-n*pi/2. 
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def weight_mean_std(x,sigma):
    num = len(x)
    sum_x = 0
    val_x = 0
    for i in range(num):
        sum_x += x[i]/sigma[i]**2
        val_x += 1./sigma[i]**2
    return sum_x/val_x, np.sqrt(1./val_x)
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

def read_ptg(filename):
    print( "" )
    print( "[READ PTG]: BEGIN reading "+filename )
    ptg = h.mrdfits(filename,hdu=1)
    ptg_package = {'ra':ptg[2], 'dec':ptg[1], 'pa':ptg[0], 'hwp':ptg[3]}
    print( "[READ PTG]: END reading " )
    return ptg_package

def convert_JD2MJD(JD):
    MJD = JD - 2400000.5
    return np.array(MJD)

def get_lst(JD,e_long):
    MJD = convert_JD2MJD(JD)
    nb = len_all(JD)
    dT = sla.sla_dt(2000)
    if nb>1:
        lst = np.zeros(nb)
        for i in range(0,nb):
            gmst = sla.sla_gmst(MJD[i]) # [radian]
            MJD_eoe = MJD[i] + dT
            equation_of_equinoxes = sla.sla_eqeqx(MJD_eoe) # [radian]
            lst[i] = gmst + e_long + equation_of_equinoxes # [radian]
    if nb==1:
        MJD_eoe = MJD + dT
        equation_of_equinoxes = sla.sla_eqeqx(MJD_eoe)
        gmst = sla.sla_gmst(MJD)
        lst = gmst + e_long + equation_of_equinoxes
    lst = lst / pi * 180. / 15. # [hour]
    lst = lst % 24 # our within 0~24
    return np.array(lst) # [hour]

def azel2radec(az,el,jd,obs_lon,obs_lat):
    ''' 
    azel2radec.py
    inputs: az[rad], el[rad], jd, obs_lat[rad]
    outputs: ra[rad], dec[rad]
    '''
    lst = get_lst(jd, obs_lon)
    ha,dec = sla.sla_dh2e(az, el, obs_lat)
    ra = lst*15./180.*pi - ha
    if ra < 0: ra = ra+2.*pi
    return ra, dec

def radec2azel(ra,dec,jd,obs_lon,obs_lat):
    '''
    radec2azel.py
    inputs: ra[rad], dec[rad], jd, obs_lat[rad]
    outputs: az[rad], el[rad]
    '''
    lst = get_lst(jd, obs_lon)
    ha = lst*15./180.*pi - ra
    az,el = sla.sla_de2h(ha, dec, obs_lat)
    return az, el

def polarvec(theta, phi):
    p = [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]
    return np.array(p)

def polarvec_astro(dec, ra):
    p = [np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)]
    return np.array(p)

def vec2NP_astro(dec, ra):
    p = [-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)]
    return np.array(p)

def compute_nhits(nside, ra, dec, nhits):
    dec = pi/2.-dec
    ipix = h.ang2pix(nside,dec,ra)
    nhits[ipix] += 1.
    return np.array(nhits)

def compute_nhits_crosslink(nside, ra, dec, hwpang, map_nhits, map_sin, map_cos, map_ang, order):
    theta = pi/2.-dec
    ipix = h.ang2pix(int(nside),theta,ra)
    nb = len(ra)
    fp_angle = np.zeros(nb)
    p_np = vec2NP_astro(dec, ra)
    idx = range(1,nb)
    idx_ = range(0,nb-1)
    p = polarvec_astro(dec, ra)
    dp = np.zeros([3,nb])
    dp[:,idx_] = p[:,idx] - p[:,idx_]
    dp[:,nb-1] =  dp[:,nb-2]
    for i in range(0,nb):
        norm1 = np.sqrt(dot(dp[:,i],dp[:,i]))
        norm2 = np.sqrt(dot(p_np[:,i],p_np[:,i]))
        tmp = np.arccos(dot(p_np[:,i],dp[:,i])/norm1/norm2)
        if (tmp > 0.5*pi): tmp = pi - tmp
        map_sin[ipix[i]] += np.sin(float(order)*tmp+hwpang)
        map_cos[ipix[i]] += np.cos(float(order)*tmp+hwpang)
#        map_sin[ipix[i]] = tmp*180./pi
        map_nhits[ipix[i]] += 1.
#        pa[i] = arccos(tmp)
        fp_angle[i] = tmp
        map_ang[ipix[i]] += tmp
    return np.array(map_nhits), np.array(map_sin), np.array(map_cos), np.array(map_ang), np.array(fp_angle)

def DistOnSphere(az,el):
    '''
    input: az = [az0, az1], el = [el0, el1] in [rad]
    output: separation angle of two pointing vectors, dist in [rad]
    '''
    i = 0
    dist = np.arccos(np.cos(az[i])*np.cos(el[i])*np.cos(az[i+1])*np.cos(el[i+1])
                  + np.sin(az[i])*np.cos(el[i])*np.sin(az[i+1])*np.cos(el[i+1])
                  + np.sin(el[i])*np.sin(el[i+1]))
    return abs(dist)

def dist_sph2pnt(az1,el1,az2,el2):
    x1 = np.cos(el1)*np.cos(az1)
    y1 = np.cos(el1)*np.sin(az1)
    z1 = np.sin(el1)
    x2 = np.cos(el2)*np.cos(az2)
    y2 = np.cos(el2)*np.sin(az2)
    z2 = np.sin(el2)
    cos_theta = (x1*x2+y1*y2+z1*z2)/np.sqrt(x1**2+y1**2+z1**2)/np.sqrt(x2**2+y2**2+z2**2)
    theta = np.arccos(cos_theta)
    return np.array(theta)

def convert_Gregorian2Julian(year, month, day, hour, minute, sec):
    a = int((14-month)/12)
    y = year + 4800 - a
    m = month + 12*a - 3
    JDN = day+int((153*m+2)/5)+365*y+int(y/4)-int(y/100)+int(y/400)-32045
    JD = JDN + (hour-12)/24. + minute/1440. + sec/86400.
    return np.array(JD)

def convert_Dublin2Julian(DJD):
    JD = DJD + 2415020.
    return np.array(JD)

def convert_Julian2Dublin(JD):
    DJD = JD - 2415020.
    return np.array(DJD)

def convert_JD2MJD(JD):
    MJD = JD - 2400000.5
    return np.array(MJD)

def convert_JD2TJD(JD):
    TJD = JD - 2440000.5
    return np.array(TJD)

def convert_hour2rad(hour):
    rad = hour*15./180.*pi
    return np.array(rad)

def convert_hour2deg(hour):
    deg = hour*15.
    return np.array(deg)

def angle_separation(ra1,dec1,ra2,dec2):
    """
    INPUT: ra1,dec1 and ra2,dec2 in radian
    OUTPUT: separation angle in radian
    """
    cos_del = np.cos(dec1)*np.cos(ra1)*np.cos(dec2)*np.cos(ra2)+np.cos(dec1)*np.sin(ra1)*np.cos(dec2)*np.sin(ra2)+np.sin(dec1)*np.sin(dec2)
    delta = np.arccos(cos_del)
    return np.array(delta)

def JD2lst(JD,w_long):
    '''
    convert JD to lst based on wikipedia
    '''
    TJD = JD - 2440000.5
    sidereal = 24. * (0.671262 + 1.0027379094*TJD) # [h]
    sidereal = sidereal%24.
    lst = sidereal - w_long/pi*180./15.
    return lst

def get_lst(JD,e_long):
    '''
    Inputs: JD [day], e_long [degrees], east longitude
    Output: LST [hour]
    '''
    TJD = JD - 2440000.5
    GST = 24.*(0.671262+1.0027379094*TJD)
#    print( 'GST', GST, GST%24.
    LST = GST - e_long/15.
    return np.array(LST%24.) # [hour]

def daysJ2000(year, month, day):
    dwhole =367*year-int(7*(year+int((month+9)/12))/4)+int(275*month/9)+day-730531.5
    return np.array(dwhole)

def daysJ2000_decimal(year, month, day, hour, minute, sec):
    dwhole =367*year-int(7*(year+int((month+9)/12))/4)+int(275*month/9)+day-730531.5
    dfrac = (hour + minute/60 + sec/3600)/24
    d = dwhole + dfrac
    return np.array(d)

def hms2frac(hour, minute, sec):
    frac = hour+minute/60.+sec/60./60.
    return np.array(frac)

def ha2ra(lst,ha):
    ra = lst - ha
    return np.array(ra) # radian

def convert_UTC2lst(year, month, day, hour, minute, sec, e_long):
    JD = convert_Gregorian2Julian(year, month, day, hour, minute, sec)
    lst = get_lst(np.array([JD]),e_long)
    return lst

def convert_azel2radec(JD,az,el,obs_lon,obs_lat):
    """
    INPUT: 
      JD: Julian day [sec]
      az: [rad]
      el: [rad]
      obs_lon: east longitude [rad] (0~2pi)
      obs_lat: latitude [rad] (-pi~pi)
    OUTPUT:
      ra: [rad]
      dec: [rad]
    """
    ha,dec = convert_azel2hadec(az,el,obs_lat)
    lst = get_lst(JD, obs_lon)
    lst = convert_hour2rad(lst)
    ra = ha2ra(lst,ha)
    ind = where(ra < 0.)
    if len(ind)>0:
        ra[ind[0]] = ra[ind[0]] + 2.*pi
    ra = ra % (2.*pi)
    return np.array(ra), np.array(dec)

def convert_azel2hadec(az,el,lat):
    ha = np.arctan( -np.sin(az)*np.cos(el) / ( -np.cos(az)*np.sin(lat)*np.cos(el)+np.sin(el)*np.cos(lat) ))
    ind = np.where(ha < 0.)
    if len(ind)>0:
        ha[ind[0]] = ha[ind[0]] + 2.*pi
    ha = ha % (2.*pi)
    # Find declination (positive if north of Celestial Equator, negative if south)
    sindec = np.sin(lat)*np.sin(el) + np.cos(lat)*np.cos(el)*np.cos(az)
    dec = np.arcsin(sindec)
    return np.array(ha), np.array(dec)

def convert_radec2galactic(ra,dec):
    dec = pi/2.-dec
    x = np.cos(ra)*np.sin(dec)
    y = np.sin(ra)*np.sin(dec)
    z = np.cos(dec)
    c1 = [-0.0548755, -0.873437, -0.483835]
    c2 = [ 0.49411,   -0.44483,   0.746982]
    c3 = [-0.867666,  -0.198076,  0.455984]
    xp = x*c1[0]+y*c1[1]+z*c1[2]
    yp = x*c2[0]+y*c2[1]+z*c2[2]
    zp = x*c3[0]+y*c3[1]+z*c3[2]
    glat = np.arctan2(zp,np.sqrt(xp**2+yp**2))
    glon = np.arctan2(yp,xp)
    return np.array(glon), np.array(glat)

def convert_galactic2radec(glon,glat):
    glat = pi/2.-glat
    x = np.cos(glon)*np.sin(glat)
    y = np.sin(glon)*np.sin(glat)
    z = np.cos(glat)
    c1 = [-0.05487615,  0.49410936, -0.86766598]
    c2 =  [-0.87343707, -0.4448295,  -0.19807669]
    c3 =  [-0.48383516,  0.74698201,  0.45598419]
    xp = x*c1[0]+y*c1[1]+z*c1[2]
    yp = x*c2[0]+y*c2[1]+z*c2[2]
    zp = x*c3[0]+y*c3[1]+z*c3[2]
    dec = np.arctan(zp/np.sqrt(xp**2+yp**2))
    ra = np.arctan(yp/xp)
    return np.array(ra), np.array(dec)

def TrackPatch_radec(ra,dec,JD,obs_lon,obs_lat):
    """
    INPUT ra: [rad], dec: [rad], JD: [day], obs_lon: east longitude [rad], obs_lat: [rad]
    """
    lst = get_lst(JD, obs_lon) # lst in [hour]
    ha = lst*15./180.*pi - ra # ha in [rad]
    az,el = sla.sla_de2h(ha, dec, obs_lat)
    return az, el

def Scanset_eloffset(ra,dec,period,JD,obs_lon,obs_lat):
    nb = int(period/100.)
    JD_arr = JD + np.arange(nb)/24./3600. * 100.
    lst = get_lst(JD_arr, obs_lon)
    ha = lst*15./180.*pi - ra
    az = np.zeros(nb)
    el = np.zeros(nb)
    for i in range(0,nb):
        az[i],el[i] = sla.sla_de2h(ha[i], dec, obs_lat)
    # accending
    if (el[0] < el[nb-1]): 
        offset_el = (max(el) - min(el))/2.
    # deccending
    if (el[0] > el[nb-1]): 
        offset_el = -(max(el) - min(el))/2.
    return offset_el

def EULER(ai,bi,select,FK4):
    ''';+
    ; NAME:
    ;     EULER
    ; PURPOSE:
    ;     Transform between Galactic, celestial, and ecliptic coordinates.
    ; EXPLANATION:
    ;     Use the procedure ASTRO to use this routine interactively
    ;
    ; CALLING SEQUENCE:
    ;      EULER, AI, BI, AO, BO, [ SELECT, /FK4, SELECT = ] 
    ;
    ; INPUTS:
    ;       AI - Input Longitude in DEGREES, scalar or vector.  If only two 
    ;               parameters are supplied, then  AI and BI will be modified to 
    ;               contain the output longitude and latitude.
    ;       BI - Input Latitude in DEGREES
    ;
    ; OPTIONAL INPUT:
    ;       SELECT - Integer (1-6) specifying type of coordinate transformation.  
    ;
    ;      SELECT   From          To        |   SELECT      From            To
    ;       1     RA-Dec (2000)  Galactic   |     4       Ecliptic      RA-Dec    
    ;       2     Galactic       RA-DEC     |     5       Ecliptic      Galactic  
    ;       3     RA-Dec         Ecliptic   |     6       Galactic      Ecliptic  
    ;
    ;      If not supplied as a parameter or keyword, then EULER will prompt for 
    ;      the value of SELECT
    ;      Celestial coordinates (RA, Dec) should be given in equinox J2000 
    ;      unless the /FK4 keyword is set.
    ; OUTPUTS:
    ;       AO - Output Longitude in DEGREES
    ;       BO - Output Latitude in DEGREES
    ;
    ; INPUT KEYWORD:
    ;       /FK4 - If this keyword is set and non-zero, then input and output 
    ;             celestial and ecliptic coordinates should be given in equinox 
    ;             B1950.
    ;       /SELECT  - The coordinate conversion integer (1-6) may alternatively be 
    ;              specified as a keyword
    ; NOTES:
    ;       EULER was changed in December 1998 to use J2000 coordinates as the 
    ;       default, ** and may be incompatible with earlier versions***.
    ; REVISION HISTORY:
    ;       Written W. Landsman,  February 1987
    ;       Adapted from Fortran by Daryl Yentis NRL
    ;       Converted to IDL V5.0   W. Landsman   September 1997
    ;       Made J2000 the default, added /FK4 keyword  W. Landsman December 1998
    ;       Add option to specify SELECT as a keyword W. Landsman March 2003
    ;-
    '''
    twopi   =   2.0*pi
    fourpi  =   4.0*pi
    deg_to_rad = (180.0/pi)

#;   J2000 coordinate conversions are based on the following constants
#;   (see the Hipparcos explanatory supplement).
#;  eps = 23.4392911111d              Obliquity of the ecliptic
#;  alphaG = 192.85948d               Right Ascension of Galactic North Pole
#;  deltaG = 27.12825d                Declination of Galactic North Pole
#;  lomega = 32.93192d                Galactic longitude of celestial equator  
#;  alphaE = 180.02322d              Ecliptic longitude of Galactic North Pole
#;  deltaE = 29.811438523d            Ecliptic latitude of Galactic North Pole
#;  Eomega  = 6.3839743d              Galactic longitude of ecliptic equator              

    if 'B1950' in FK4:
        equinox = '(B1950)' 
        psi = [ 0.57595865315, 4.9261918136, 0.00000000000, 0.0000000000, 0.11129056012, 4.7005372834]     
        stheta =[ 0.88781538514,-0.88781538514, 0.39788119938,-0.39788119938, 0.86766174755,-0.86766174755]    
        ctheta =[ 0.46019978478, 0.46019978478, 0.91743694670, 0.91743694670, 0.49715499774, 0.49715499774]    
        phi = [ 4.9261918136,  0.57595865315, 0.0000000000, 0.00000000000, 4.7005372834, 0.11129056012]
    elif 'J2000' in FK4:
        equinox = '(J2000)'
        psi = [ 0.57477043300, 4.9368292465, 0.00000000000, 0.0000000000, 0.11142137093, 4.71279419371]     
        stheta =[ 0.88998808748,-0.88998808748, 0.39777715593,-0.39777715593, 0.86766622025,-0.86766622025]    
        ctheta =[ 0.45598377618, 0.45598377618, 0.91748206207, 0.91748206207, 0.49714719172, 0.49714719172]    
        phi = [ 4.9368292465,  0.57477043300, 0.0000000000, 0.00000000000, 4.71279419371, 0.11142137093]
    else:
        print( 'no FK4 is set' )

    if ((select<0) or (select>6)):
        print( ' ' )
        print( ' 1 RA-DEC ' + equinox + ' to Galactic' )
        print( ' 2 Galactic       to RA-DEC' + equinox )
        print( ' 3 RA-DEC ' + equinox + ' to Ecliptic' )
        print( ' 4 Ecliptic       to RA-DEC' + equinox )
        print( ' 5 Ecliptic       to Galactic' )
        print( ' 6 Galactic       to Ecliptic' )
        return 0, 0

    i = select - 1                         # IDL offset
    a = ai/deg_to_rad - phi[i]
    b = bi/deg_to_rad
    sb = np.sin(b)
    cb = np.cos(b)
    cbsa = cb * np.sin(a)
    b = -stheta[i] * cbsa + ctheta[i] * sb
    bo = np.arcsin(b)*deg_to_rad

    a = np.arctan2( ctheta[i] * cbsa + stheta[i] * sb, cb * np.cos(a) )
    ao = ( (a+psi[i]+fourpi) % twopi) * deg_to_rad

    return ao, bo

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def plot_bin(x,y,binsize,symbol):
    nbData = len(x)
    nbData_bin = int((np.max(x)-np.min(x))/binsize)
    l=2
    x_ = [0.,0.]
    y_ = [0.,0.]
    er = [0.,0.]

    for i in range(0,nbData_bin):
        ind_tmp = np.where( (x >= np.min(x)+binsize*i) & (x < np.min(x)+binsize*(i+1)) )
        if len(ind_tmp[0]) > 1:
            x_.append( np.min(x)+binsize*i+binsize/2. )
            y_.append( np.mean(y[ind_tmp]) )
        #y_.append( np.median(y[ind_tmp]) )
            er.append( np.std(y[ind_tmp]) )
            l += 1

    errorbar(x_[2:l], y_[2:l], er[2:l], fmt=symbol)

    return np.array(x_[2:l]),np.array(y_[2:l]),np.array(er[2:l])


#def plot_hist(x,nbin,par=-1,fit=False,init_auto=False,xtitle=-1,no_plot=False,plot=True,normed=False,label='', fitlinecolor='r',histcolor='blue'):
def plot_hist(x,nbin,par=-1,fit=False,init_auto=False,xtitle=-1,no_plot=False,plot=True,label='', fitlinecolor='r',histcolor='blue'):
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
#    non, bins, patches = py.hist(x, nbin, histtype='step', normed=normed, label=label, color=histcolor)#, normed=1, facecolor='green', alpha=0.75)
    non, bins, patches = py.hist(x, nbin, histtype='step', label=label, color=histcolor)#, normed=1, facecolor='green', alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])

    func_gauss = lambda p, xin: p[0]*np.exp(-(xin-p[1])**2/(2.*p[2]**2))
    chi_nosigma = lambda p, xin, d: ((func_gauss(p,xin)-d)**2).sum()

    if fit: 
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        print( '+++  Fit the histogram with Gaussian +++' )
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print( 'initial guess:', par0 )
#        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/float(nbin))
        par, fopt,iterout,funcalls,warnflag=fmin(chi_nosigma,par0,args=(bincenters,non),maxiter=10000,maxfun=10000,xtol=0.01,full_output=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),fitlinecolor, linewidth=1)
        if plot == True: py.plot(x,func_gauss(par,x),fitlinecolor, linewidth=1)
        print( 'fitted parameters:', par )
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
    py.grid(True)

#    py.show()

    return np.array(par)

def plot_hist_weight(x,nbin,weight,par=-1,fit=False,init_auto=False,xtitle=-1,no_plot=False,normed=False):
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
         weight: weight = 1/sigma^2
     output:
         fit parameters
    """
    # the histogram of the data
    non, bins, patches = py.hist(x, nbin, weights=weight,histtype='step')#, normed=1, facecolor='green', alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])

    func_gauss = lambda p, xin: p[0]*np.exp(-(xin-p[1])**2/(2.*p[2]**2))
    chi_nosigma = lambda p, xin, d: ((func_gauss(p,xin)-d)**2).sum()

    if fit: 
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        print( '+++  Fit the histogram with Gaussian +++' )
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print( 'initial guess:', par0 )
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        par = fmin(chi_nosigma, par0, args=(bincenters,non), maxiter=10000, maxfun=10000, xtol=0.01)
#        if no_plot == False: py.plot(bincenters,func_gauss(par,bincenters),'r', linewidth=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),'r', linewidth=1)
        #y = mlab.normpdf(bincenters, par[1], par[2])
        #l = py.plot(bincenters, y, 'r--', linewidth=1)
        print( 'fitted parameters:', par )
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
#    py.grid(True)

#    py.show()

    return np.array(par)

def reject_sigma(x_in,sigma_limit):
    num_data = len(x_in)
    num_tmp_prev = num_data
    num_tmp = 1e10
    while (num_tmp != num_tmp_prev ):
        num_tmp_prev = num_tmp
        ind = np.where((np.std(x_in)*sigma_limit > x_in))
        num_tmp = len(ind[0])
    return x_in[ind[0]]

def reject_sigma_v2(x_in,sigma_limit):
    num_data = len(x_in)
    num_tmp_prev = num_data
    num_tmp = 1e10
    while (num_tmp != num_tmp_prev ):
        num_tmp_prev = num_tmp
        ind = np.where((np.std(x_in)*sigma_limit > x_in))
        num_tmp = len(ind[0])
    return ind[0], x_in[ind[0]]

##############################################################################################################
def Gauss_2D(x,y,sigma_x,sigma_y):
    return np.exp(-(x**2/(2.*sigma_x**2)))*np.exp(-(y**2/(2.*sigma_y**2)))

def Gauss_1D(par,x):
    return par[0]*np.exp(-((x-par[1])**2/(2.*par[2]**2)))

def chi2_nosigma(d_model, d_data):
    chi2_sum = ((d_model-d_data)**2).sum()
    return chi2_sum

def chi2_sigma(d_model, d_data, sigma):
    chi2_sum = ((d_model-d_data)**2 / sigma**2).sum()
    return chi2_sum

def fit_1DGauss_nosigma(p, x, d_data):
    model = Gauss_1D(p,x)
    chi2_sum = ((model-data)**2).sum()
    return chi2_sum

def Gauss1D_fit_nosigma(p,x,d,fit=False,autoguess=False):
    '''
    fit 1D Gaussian
    p: par, x: xin, d: data to fit
    option
    fit: if true, return the fitted line with the same x
    autoguess: if true, it does initial guess automatically
    '''
    if autoguess: p = np.array([max(d),np.median(d),np.std(x)])
    func = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2.*p[2]**2))
    chi2 = lambda p, x, d: ((func(p,x)-d)**2).sum()
    par = fmin(chi2, p, args=(x,d), maxiter=10000, maxfun=10000, xtol=0.01)
    if fit: return par, func(par,x)
    return par

def Gauss2D_fit_nosigma(p,x,data):
    Gauss_2D = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2.*p[3]**2))*np.exp(-(y-p[2])**2/(2.*p[4]**2))
    model = Gauss_2D(p,x,y)
    par = fmin(chi2_nosigma, p, args=(model,data), maxiter=10000, maxfun=10000, xtol=0.01)
    return par

##############################################################################################################

def calPSD( datain, samplerate, outputunit):
    """ this program calculate the power spectrum density [V/rtHz]
        caution!  Normalization for k=1,...N/2-1 and k=0&nyquist is 
        is different by factor of sqrt(2)
        
        output unit
        1: sum squared amplitude K^2 (NRC 13.4.1)
        2: mean squared amplitude K^2/nbData (NRC 13.4.2)
        3: time integral squared amplitude K^2 sec (NRC 13.4.3)
        4: sqrt of 1
        5: sqrt of 2
        6: sqrt of 3
        7: normalized to 1 using the average of first 5 data points at low freq
        unit is still in K, not K^2
    """
  
    nbData = len(datain)
    nrowh = nbData//2
    delta = 1./samplerate
    PSD = np.zeros(nrowh+1)
    
#; inverse FT, direction=1, does not include normalization
#; same as (NRC-13.4.4)
    Ck = abs(fft.fft(datain)) 
  
    PSD[0] = Ck[0]**2 / np.double(nbData**2.)       #;PSD at DC
    PSD[nrowh] = Ck[nrowh]**2 / np.double(nbData**2.) #; PSD at Nyquist freq
    
    for i in range(1,nrowh): #i=1L, nrowh-1L do begin
        PSD[i] = (Ck[i]**2 + Ck[nbData-i]**2) / np.double(nbData**2.)
        
    freq = np.arange(0,nrowh+1)/np.double(nrowh)*samplerate/2.
    
    if (outputunit == 1): PSD = PSD*nbData
    if (outputunit == 2): PSD = PSD
    if (outputunit == 3): PSD = PSD*delta*nbData
        
    if (outputunit == 4): PSD = np.sqrt(PSD*nbData)
    if (outputunit == 5): PSD = np.sqrt(PSD)
    if (outputunit == 6): PSD = np.sqrt(PSD*delta*nbData)

    if (outputunit == 7): PSD = np.sqrt(PSD)/mean(np.sqrt(PSD[0:4]))
  
    return np.array(freq),np.array(PSD)


def calPSD_cross( datain1, datain2, samplerate, outputunit):
    """ this program calculate the cross power spectrum density [V/rtHz]
        caution!  Normalization for k=1,...N/2-1 and k=0&nyquist is 
        is different by factor of sqrt(2)
        
        output unit
        1: sum squared amplitude K^2 (NRC 13.4.1)
        2: mean squared amplitude K^2/nbData (NRC 13.4.2)
        3: time integral squared amplitude K^2 sec (NRC 13.4.3)
        4: sqrt of 1
        5: sqrt of 2
        6: sqrt of 3
        7: normalized to 1 using the average of first 5 data points at low freq
        unit is still in K, not K^2
    """
  
    nbData = len(datain1)
    nrowh = nbData//2
    delta = 1./samplerate
    CPSD = np.zeros(nrowh+1)
    
#; inverse FT, direction=1, does not include normalization
#; same as (NRC-13.4.4)
    Ck1 = fft.fft(datain1) 
    Ck2 = fft.fft(datain2) 
  
    CPSD[0] = Ck1[0]*Ck2[0].conj() / np.double(nbData**2.)       #;PSD at DC
    CPSD[nrowh] = Ck1[nrowh]*Ck2[nrowh].conj() / np.double(nbData**2.) #; PSD at Nyquist freq

    for i in range(1,nrowh): #i=1L, nrowh-1L do begin
        CPSD[i] = (Ck1[i]*Ck2[i].conj() + Ck1[nbData-i]*Ck2[nbData-i].conj()) / np.double(nbData**2.)
        
    freq = np.arange(0,nrowh+1)/np.double(nrowh)*samplerate/2.
    
    if (outputunit == 1): CPSD = CPSD*nbData
    if (outputunit == 2): CPSD = CPSD
    if (outputunit == 3): CPSD = CPSD*delta*nbData
        
    if (outputunit == 4): CPSD = np.sqrt(CPSD*nbData)
    if (outputunit == 5): CPSD = np.sqrt(CPSD)
    if (outputunit == 6): CPSD = np.sqrt(CPSD*delta*nbData)

    if (outputunit == 7): CPSD = np.sqrt(CPSD)/mean(np.sqrt(CPSD[0:4]))
  
    return np.array(freq),np.array(CPSD)

def calPSD_nfft( time, data, Nf, outputunit):
    """ this program calculate the power spectrum density [V/rtHz]
         caution!  Normalization for k=1,...N/2-1 and k=0&nyquist is 
         is different by factor of sqrt(2)

         output unit
         1: sum squared amplitude K^2 (NRC 13.4.1)
         2: mean squared amplitude K^2/nbData (NRC 13.4.2)
         3: time integral squared amplitude K^2 sec (NRC 13.4.3)
         4: sqrt of 1
         5: sqrt of 2
         6: sqrt of 3
         7: normalized to 1 using the average of first 5 data points at low freq
         unit is still in K, not K^2
    """

    Ck = nf.nfft_adjoint(time, data, Nf, sigma=5, tol=1E-8)

    total_time = (np.max(time)-np.min(time))
    delta_time = total_time / Nf
    samplerate = 1./delta_time

    nrowh = Nf//2
    PSD = np.zeros(nrowh+1)
    PSD[0] = np.abs(Ck[0])**2 / np.double(Nf**2.)       #;PSD at DC
    PSD[nrowh] = np.abs(Ck[nrowh])**2 / np.double(Nf**2.) #; PSD at Nyquist freq

    for i in range(1,nrowh): #i=1L, nrowh-1L do begin
        PSD[i] = (np.abs(Ck[nrowh+i])**2 + np.abs(Ck[-(nrowh+i)])**2) / np.double(Nf**2.)
        
    freq = np.arange(0,nrowh+1)/np.double(nrowh)*samplerate/2.
   
    if (outputunit == 1): PSD = PSD*Nf
    if (outputunit == 2): PSD = PSD
    if (outputunit == 3): PSD = PSD*delta_time*Nf

    if (outputunit == 4): PSD = np.sqrt(PSD*Nf)
    if (outputunit == 5): PSD = np.sqrt(PSD)
    if (outputunit == 6): PSD = np.sqrt(PSD*delta_time*Nf)
 
    return np.array(freq),np.array(PSD)

def AppFilt( datain, samplerate, outputunit, filtertype, par):
    """ this program applys either low or high pass filter

        single pole low pass filter: par[0] = freq_0
        single pole high pass filter: par[0] = freq_0

        cosine low pass filter: par[0] = freq_0, par[1] = width

        output unit
        1: sum squared amplitude K^2 (NRC 13.4.1)
        2: mean squared amplitude K^2/nbData (NRC 13.4.2)
        3: time integral squared amplitude K^2 sec (NRC 13.4.3)
        4: sqrt of 1
        5: sqrt of 2
        6: sqrt of 3
        7: normalized to 1 using the average of first 5 data points at low freq
        unit is still in K, not K^2
    """
  
    nbData = len(datain)
    nrowh = nbData/2
    delta = 1./samplerate
    PSD = np.zeros(nrowh+1)
    PSD_filt = np.zeros(nrowh+1)

    freq = np.arange(0,nrowh+1)/np.double(nrowh)*samplerate/2.

    if filtertype == 'sp_lowpass':
        filt = sp_lowpass(freq,par[0])
    if filtertype == 'sp_highpass':
        filt = sp_highpass(freq,par[0])
    if filtertype == 'cosine_lowpass':
        filt = cosine_lowpass(freq, par[0], par[1])
    if filtertype == 'cosine_highpass':
        filt = cosine_highpass(freq, par[0], par[1])
    if filtertype == 'cosine_bandpass':
        filt = cosine_bandpass(freq, par[0], par[1], par[2], par[3])

#; inverse FT, direction=1, does not include normalization
#; same as (NRC-13.4.4)
    Ck = fft.fft(datain) 
  
    PSD[0] = Ck[0]**2 / np.double(nbData**2.)       #;PSD at DC
    PSD[nrowh] = Ck[nrowh]**2 / np.double(nbData**2.) #; PSD at Nyquist freq

    for i in range(1,nrowh): #i=1L, nrowh-1L do begin
        PSD[i] = ((np.abs(Ck[i]))**2 + (np.abs(Ck[nbData-i]))**2) / np.double(nbData**2.)
        PSD_filt[i] = ((np.abs(Ck[i]*filt[i]))**2 + (np.abs(Ck[nbData-i]*filt[i]))**2) / np.double(nbData**2.)

    Ck_out = np.zeros(nbData,dtype='complex')
    Ck_out[0] = Ck[0]*filt[0]
    Ck_out[nrowh] = Ck[nrowh]*filt[nrowh]
    for i in range(1,nrowh):
        Ck_out[i] = Ck[i]*filt[i]
        Ck_out[nbData-i] = Ck[nbData-i]*filt[i]

    tod_filt = fft.ifft(Ck_out)
  
    if (outputunit == 1): 
        PSD = PSD*nbData; PSD_filt = PSD_filt*nbData
    if (outputunit == 2): 
        PSD = PSD; PSD_filt = PSD_filt
    if (outputunit == 3): 
        PSD = PSD*delta*nbData; PSD_filt = PSD_filt*delta*nbData
        
    if (outputunit == 4): 
        PSD = np.sqrt(PSD*nbData); PSD_filt = np.sqrt(PSD_filt*nbData)
    if (outputunit == 5): 
        PSD = np.sqrt(PSD); PSD_filt = np.sqrt(PSD_filt)
    if (outputunit == 6): 
        PSD = np.sqrt(PSD*delta*nbData); PSD_filt = np.sqrt(PSD_filt*delta*nbData)

    if (outputunit == 7): 
        PSD = np.sqrt(PSD)/mean(np.sqrt(PSD[0:4])); PSD_filt = np.sqrt(PSD_filt)/mean(np.sqrt(PSD_filt[0:4]))
  
    return np.array(freq), np.array(PSD), np.array(PSD_filt), np.array(tod_filt), np.array(filt)

def multiply_complexarr(arr1,arr2):
    nb1 = np.size(arr1)
    nb2 = np.size(arr2)
    if nb1 != nb2: sys.exit('multiply_complexarr: # of array1 != # of array2')

    arr1_real = arr1.real
    arr1_imag = arr1.imag
    
    arr2_real = arr2.real
    arr2_imag = arr2.imag

    out_arr = np.zeros(nb1,complex)

    out_arr.real = arr1_real*arr2_real-arr2_imag*arr2_imag
    out_arr.imag = arr1_imag*arr2_real+arr1_real*arr2_imag    

    return np.array(out_arr)


def divide_complexarr(arr1,arr2):
    nb1 = np.size(arr1)
    nb2 = np.size(arr2)
    if nb1 != nb2: sys.exit('divide_complexarr: # of array1 != # of array2')

    arr1_real = arr1.real
    arr1_imag = arr1.imag
    
    arr2_real = arr2.real
    arr2_imag = arr2.imag

    out_arr = np.zeros(nb1,complex)

    denom = (arr2_real*arr2_real+arr2_imag*arr2_imag)
    out_arr.real = (arr1_real*arr2_real+arr1_imag*arr2_imag)/denom
    out_arr.imag = -(arr1_real*arr2_imag-arr1_imag*arr2_real)/denom

    return np.array(out_arr)


def complex_arr(arr1,arr2):
    nb1 = np.size(arr1)
    nb2 = np.size(arr2)

    if nb1 == nb2:
        out_arr = np.zeros(nb1,complex)
        out_arr.real=arr1
        out_arr.imag=arr2

    if nb1!=nb2 and nb1==1:
        out_arr = np.zeros(nb2,complex)
        i = np.arange(1,nb2+1)/np.arange(1,nb2+1)
        out_arr.real=i*arr1
        out_arr.imag=arr2

    if nb1!=nb2 and nb2==1:
        out_arr = np.zeros(nb1,complex)
        i = np.arange(1,nb1+1)/np.arange(1,nb1+1)
        out_arr.real=arr1
        out_arr.imag=i*arr2

    if nb1==1 and nb2==1:
        out_arr = np.eros(nb1,complex)
        out_arr.real=arr1
        out_arr.imag=arr2

    return np.array(out_arr)

def cosine_lowpass(freq,freq_0, freq_w):
    omega = pi / freq_w
    num = len(freq)
    tf = np.ones(num)
    tf = 0.5 * np.cos(omega*(freq-freq_0+0.5*freq_w)) + 0.5
    ind = np.where(freq < (freq_0-0.5*freq_w))
    tf[ind[0]] = 1.
    ind = np.where(freq > (freq_0+0.5*freq_w))
    tf[ind[0]] = 0.
    return tf

def cosine_highpass(freq,freq_0, freq_w):
    omega = pi / freq_w
    num = len(freq)
    tf = np.ones(num)
    tf = - 0.5 * np.cos(omega*(freq-freq_0+0.5*freq_w)) + 0.5
    ind = np.where(freq < (freq_0-0.5*freq_w))
    tf[ind[0]] = 0.
    ind = np.where(freq > (freq_0+0.5*freq_w))
    tf[ind[0]] = 1.
    return tf

def cosine_bandpass(freq,freq_l,freq_wl,freq_h,freq_wh):
    num = len(freq)
    tf = np.ones(num)

    omega = pi / freq_wl
    tf = 0.5 * np.cos(omega*(freq-(freq_l-0.5*freq_wl))) + 0.5
    ind = np.where(freq < (freq_l-0.5*freq_wl))
    tf[ind[0]] = 1.
    ind = np.where(freq > (freq_l+0.5*freq_wl))
    tf[ind[0]] = 0.

    ind = np.where( (freq > (freq_h-0.5*freq_wh)) & (freq < (freq_h+0.5*freq_wh)) )
    omega = pi / freq_wh
    tf[ind[0]] = - 0.5 * np.cos(omega*(freq-(freq_h-freq_wh))) + 0.5
    ind = np.where(freq <= (freq_h-0.5*freq_wh))
    tf[ind[0]] = 0.

    return tf

def sp_lowpass(f,freq_0):
    tau = 1./(2.*pi*freq_0)
    a = 2.*pi*f*tau
    b = 1.+a**2
    tf = complex_arr(1./b,-a/b)
    return tf

def sp_highpass(f,freq_0):
    tau = 1./(2.*pi*freq_0)
    a = 2.*pi*f*tau
    b = 1.+a**2
    tf = complex_arr(1./b,-a/b)
    return tf

def median_filter(array,n):
    nb = len(array)
    array_out = np.zeros(nb-n)
    for i in range(0,nb-n):
        idx = range(i,i+n)
        array_out[i] = np.median(array[idx])
    return array_out

def cal_N2tpn(nbData):
    pn = np.log10(float(nbData)+1.)/np.log10(2.)
    pn_max = int(pn)
    tranc_nbData = 2.**pn_max
    if (tranc_nbData < nbData): tranc_nbData = 2.**(pn_max+1)
    return tranc_nbData

def pad4fft_po2(data):
    N = np.size(data)
    Npad = int( m.pow(2, 1+int(np.log(N)/np.log(2)) )) 
    data_pad = np.concatenate((data,np.zeros(Npad-N)))
    return data_pad

#######################################################
'''
ang1D2pix(x,res)
Same functionality to ang2pix in healpix routine
INPUT
x: array of angles (ra or dec of pointing)
res: bin resolution
'''
def ang1D2pix(x,res):
    nbData = len(x)
    x_max = np.max(x);    x_min = np.min(x)
    n = np.zeros(nbData,int)
    n = np.int_((x-x_min)/res)
    return np.array(n), np.max(n)+1
#######################################################
'''
ang1D2pix_fixwidth(x,res)
Same functionality to ang2pix in healpix routine
INPUT
x: array of angles (ra or dec of pointing)
width_fix: fixed width
res: bin resolution
'''
def ang1D2pix_fixwidth(x,width_fix_arr,res):
    x_max = np.max(width_fix_arr)
    x_min = np.min(width_fix_arr)
    n = np.int_((x-x_min)/res)
    return np.array(n), len(width_fix_arr)
#######################################################
'''
bin2Dmap(tod, x, y, res)
  INPUT
    tod: time stream
    x, y: corresponding pointing (eg. ra, dec)
    res: resolution to bin in degree
  OUTPUT
    Xout, Yout, mapout
  Use the output to show ra dec 2D map
    py.pcolor(Xout,Yout,map)
    py.colorbar()
'''
def bin2Dmap(tod, ra, dec, res):
    nbData = len(tod)
    print( '[lib_m.py bin2Dmap] nbData', nbData )
    res = res/180.*pi
    i_x,nbx=ang1D2pix(ra,res)
    i_y,nby=ang1D2pix(dec,res)
    mapout = np.zeros((nby,nbx))
    hits = np.zeros((nby,nbx),int)
    xx = np.zeros(nbx)
    yy = np.zeros(nby)
    for i in range(0,nbData):
        mapout[i_y[i],i_x[i]] += tod[i]
        hits[i_y[i],i_x[i]] += 1
        xx[i_x[i]] = ra[i]
        yy[i_y[i]] = dec[i]
    Xout,Yout=py.meshgrid(xx,yy)
    return Xout/pi*180., Yout/pi*180., mapout, hits

#######################################################
'''
bin2Dmap_fixedrange(tod, x, y, x_width, y_width, res)
  INPUT
    tod: time stream
    x, y: corresponding pointing (eg. ra, dec) in [rad]
    x_width, x_width: corresponding width of the cambus
    res: resolution to bin in [rad]
  OUTPUT
    Xout, Yout, mapout [rad]
  Use the output to show ra dec 2D map
    py.pcolor(Xout,Yout,map)
    py.colorbar()
'''
def bin2Dmap_fixedrange(tod, ptg_x, ptg_y, width_x, width_y, res):
    nbData = len(tod)

#    print( ptg_x
#    print( ptg_y

    print( '[lib_m.py bin2Dmap_fixedrange] nbData', nbData )
    i_x, nb_x = ang1D2pix_fixwidth(ptg_x,width_x,res)
    i_y, nb_y = ang1D2pix_fixwidth(ptg_y,width_y,res)

#    print( ptg_x, nb_x
#    print( ptg_y, nb_y

    mapout = np.zeros((nb_y,nb_x))
    hits = np.zeros((nb_y,nb_x),int)
    xx = np.zeros(nb_x)
    yy = np.zeros(nb_y)

    for i in range(0,nbData):
        mapout[i_y[i],i_x[i]] += tod[i]
        hits[  i_y[i],i_x[i]] += 1
        xx[i_x[i]] = ptg_x[i]
        yy[i_y[i]] = ptg_y[i]
#        print( 'lib_m.py: bin2Dmap_fixedrange', i, '(',i_x[i], nb_x, xx[i_x[i]]*radeg, ptg_x[i]*radeg, '), (', i_y[i], nb_y, yy[i_y[i]]*radeg, ptg_y[i]*radeg, ')', hits[i_y[i],i_x[i]]

    xx = np.linspace(np.min(width_x),np.max(width_x),nb_x)
    yy = np.linspace(np.min(width_y),np.max(width_y),nb_y)

    # py.figure()
    # py.plot(xx*radeg,yy*radeg,'.')
    # py.plot(ptg_x*radeg,ptg_y*radeg,'.')
    # py.grid()
    # py.xlim(-5,5)
    # py.ylim(-5,5)

    # print( '[lib_m.py bin2Dmap_fixedrange] Done for-loop'
    Xout,Yout=py.meshgrid(xx,yy)

    # print( '[lib_m.py bin2Dmap_fixedrange] Generate a plot'
    # fig = py.figure()
    # ax = fig.add_subplot(111)
    # im = ax.pcolor(Xout*radeg, Yout*radeg, hits, vmin=0, vmax=np.max(hits))
    # ax.axis([-5,5,-5,5])
    # fig.colorbar(im, ax=ax)
    # ax.set_xlabel("RA [degs]"); ax.set_ylabel('DEC [degs]')
    # py.show()
    # sys.exit()
    return Xout, Yout, mapout, hits

def bin1Dmap(tod, x, res):
    nbData = len(tod)
    res = res/180.*pi
    nx,nbx=ang1D2pix(x,res)
    mapout = np.zeros(nbx)
    xx = np.zeros(nbx)
    #    i = range(0,nbData)
    for i in range(0,nbData):
        mapout[nx[i]] += tod[i]
        xx[nx[i]] = x[i]
    return xx, mapout

def bin1Dmap_rms(tod, x, res):
    nbData = len(tod)
#    res = res/180.*pi
    nx,nbx=ang1D2pix(x,res)
    mapout = np.zeros((3,nbx))
    xx = np.zeros(nbx)
    #    i = range(0,nbData)
    for i in range(0,nbData):
        mapout[0,nx[i]] += tod[i]
        mapout[1,nx[i]] += 1.
        mapout[2,nx[i]] += tod[i]**2
        xx[nx[i]] += x[i]
    xxout = xx/mapout[1]
    mean = mapout[0]/mapout[1]
    rms = np.sqrt(mapout[2] / mapout[1] - (mapout[0]/mapout[1])**2)
    return xxout, mean, rms

def bin1Dmap_histsigma(tod, x, res, plot=False, path='/Users/tomotake_matsumura/Qsync/Projects/tmp_dump/'):
    nbData = len(tod)
#    res = res/180.*pi
    nx,nbx=ang1D2pix(x,res)
    mapout = np.zeros((3,nbx))
    xx = np.zeros(nbx)
    par_arr = np.zeros((3,nbx))
    for j in range(nbx):
        idx = np.where(nx == j)
        num_tmp = len(idx[0])
        xx[j] = np.mean(x[idx[0]])
        if num_tmp == 1: 
            par_arr[0,j] = tod[idx[0]]
            par_arr[1,j] = tod[idx[0]]
            par_arr[2,j] = par_arr[2,j-1]
            break
        if plot==True:
            py.figure()
            par_tmp = plot_hist(tod[idx[0]],int(np.sqrt(float(num_tmp))),fit=True,init_auto=True,plot=True,label='', fitlinecolor='r',histcolor='blue')
            py.savefig(path+'/tmp_hist'+str(j)+'.png')
        if plot==False:
            par_tmp = plot_hist(tod[idx[0]],int(np.sqrt(float(num_tmp))),fit=True,init_auto=True,plot=False,label='', fitlinecolor='r',histcolor='blue')
        par_arr[0,j] = par_tmp[0]
        par_arr[1,j] = par_tmp[1]
        par_arr[2,j] = par_tmp[2]
    return xx, par_arr

def boxxcarr_filter(x, y, num_ave):
    num = len(x)
    num_new = num - num_ave + 1
    out_arr = np.zeros((2,num_new))
    for i in range(num_new):
        out_arr[0,i] = np.mean(x[i:i+num_ave])
        out_arr[1,i] = np.mean(y[i:i+num_ave])
    return out_arr

def box_binning(x, y, delta_x):
    num = len(x)
    x_width = x[-1] - x[0]
    num_bin = int(x_width/delta_x)
    out_arr = np.zeros((2,num_bin))
    for i in range(num_bin):
        ind = np.where( ( x >= x[0]+delta_x*i ) & ( x < x[0]+delta_x*(i+1) ) )
        out_arr[0,i] = np.mean(x[ind[0]])
        out_arr[1,i] = np.mean(y[ind[0]])
    return out_arr

def box_binning_weighted(x, y, sigma, delta_x):
    num = len(x)
    x_width = x[-1] - x[0]
    num_bin = int(x_width/delta_x)
    out_arr = np.zeros((2,num_bin))
    for i in range(num_bin):
        ind = np.where( ( x >= x[0]+delta_x*i ) & ( x < x[0]+delta_x*(i+1) ) )
        out_arr[0,i] = np.sum(x[ind[0]]*sigma[ind[0]])/np.sum(sigma[ind[0]])
        out_arr[1,i] = np.sum(y[ind[0]]*sigma[ind[0]])/np.sum(sigma[ind[0]])
    return out_arr

def index_Nsigma(y, N):
    num = len(y)
    y_rms = np.std(y)
    ind = np.where( np.abs(y) <= N*y_rms) 
    ind_out = np.array(ind[0])
    if len(ind_out) == 0: ind_out = np.linspace(0,num-1,num)
    print(ind_out)
    return ind_out

''' +++++++++++++++++++++++++++++++++++++++++++++
    index_diffsigma(y)
    
    algorithm: 
        - read in del_y ( = y1-y2)
        - take sigma = np.abs(del_y)

    +++++++++++++++++++++++++++++++++++++++++++++
''' 
def index_diffsigma(y):
    num = len(y)
    sigma = np.abs(y)
    return sigma

''' +++++++++++++++++++++++++++++++++++++++++++++
    index_diffsigma_Nsigma(y, N)
    +++++++++++++++++++++++++++++++++++++++++++++
''' 
def index_diffsigma_Nsigma(y, sigma, N):
    w_mean, w_sigma = weight_mean_std(y,sigma)
    
    y_rms = np.std(y)
    ind = np.where( np.abs(y) <= N*y_rms) 
    ind_out = np.array(ind[0])
    if len(ind_out) == 0: ind_out = np.linspace(0,num-1,num)
    print(ind_out)
    return ind_out

if __name__ == '__main__': _main()


