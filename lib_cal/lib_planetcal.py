import numpy as np
import pylab as py
import sys 
import os
import ephem
import sqlite3 as sq
from scipy.interpolate import interp1d
#from scipy.weave import inline 
from scipy.optimize import fmin
import scipy.special as sp
from numpy import fft
from numpy.fft import ifft
from numpy import random
import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d, Axes3D
#import matplotlib.pyplot as plt
import pylab as py

pi = np.pi
h = 6.67e-34
c = 2.99792458e8 
kb = 1.38e-23
radeg = (180./pi)

def meshgrid2array_XY(X,Y):
    num_X = len(X[0,:])
    num_Y = len(Y[:,0])
    x_arr = []; y_arr=[]
    for i in range(num_X):
        x_arr.append(X[0,i])
    for j in range(num_Y):
        y_arr.append(Y[j,0])
    return x_arr, y_arr

def meshgrid2array_Z(Z):
    num_x = len(Z[:,0])
    num_y = len(Z[0,:])
    z_arr = []
    for i in range(num_x):
        for j in range(num_y):
            z_arr.append(Z[i,j])
    return z_arr

def sort(x_in,y_in):
    ii = 0;
    yp = np.zeros(len(x_in))
    xp = np.zeros(len(x_in))
    idx_sort = []
    for i in sorted(enumerate(x_in), key=lambda x:x_in[1]):
        idx_sort.append(i[0])
        xp[ii] = x_in[i[0]]
        yp[ii] = y_in[i[0]]
        ii += 1
    return idx_sort, xp, yp

def rotation2D(x, y, theta_rad):
    return np.cos(theta_rad)*x-np.sin(theta_rad)*y, \
        np.sin(theta_rad)*x+np.cos(theta_rad)*y

def ellipticity(sigma_x, sigma_y):
    return (sigma_x-sigma_y)/(sigma_x+sigma_y)

# samplerate in [Hz] NOT in [sec]

def read_log(dir_in):
    out = np.load(dir_in+'/runlog.npz')
    print(out['title'])
    print(out['sample_rate'])
    print(out['total_time'])
    print(out['theta_antisun'])
    print(out['freq_antisun'])
    print(out['theta_boresigiht'])
    print(out['freq_boresight'])
    print(out['ydays'])
    print(out['nside'])
    print(out['today_julian'])
    print(out['option_gen_ptg'])
    print(out['dir_out'])

class planet_loc():
    def __init__(self):
        self.name_planet = 'sun'
        self.time_JD = np.zeros(1)

    def planet_radec(self):
        num = len(self.time_JD)
        self.radec = np.zeros([2,num])
        if self.name_planet == 'Mars':
            out = ephem.Mars()
        if self.name_planet == 'Jupiter':
            out = ephem.Jupiter()
        if self.name_planet == 'Saturn':
            out = ephem.Saturn()
        if self.name_planet == 'Venus':
            out = ephem.Venus()
        if self.name_planet == 'Uranus':
            out = ephem.Uranus()
        if self.name_planet == 'Neptune':
            out = ephem.Neptune()
        if self.name_planet == 'Moon':
            out = ephem.Moon()
        if self.name_planet == 'Sun':
            out = ephem.Sun()
#        print 'current source is ', self.name_planet
        for i in range(num):
            out.compute(self.time_JD[i]-2415020.)  # convert from JD to DJD
            self.radec[0,i] = float(out.ra) # return in radian
            self.radec[1,i] = float(out.dec) # return in radian
        return self.radec

    def planet_interp(self,new_time_in):
        num = len(new_time_in)
        print(num, np.min(new_time_in), np.max(new_time_in))
        radec_new = np.zeros([2,num])
        ra_interp = interp1d(self.time_JD,self.radec[0], kind='cubic')
        dec_interp = interp1d(self.time_JD,self.radec[1], kind='cubic')
        radec_new[0] = ra_interp(new_time_in)
        radec_new[1] = dec_interp(new_time_in)
        return radec_new

    def non_planet_radec(self,new_time):
        if self.name_planet == 'TauA': 
            RA = 5./24.*360. + 34./(24.*60.)*360. + 32./(24.*3600)*360.
            Dec = 22. + 00./60. + 52./3600.
        if self.name_planet == 'CenA':
            RA = (13.401)/24.*360. 
            Dec = 43.25
        if self.name_planet == 'CasA':
            RA = 23./24.*360. + 23./(24.*60.)*360. + 24./(24.*3600)*360.
            Dec = 58. + 48./60. + 54./3600.
        if self.name_planet == 'CygA':
            RA = 19./24.*360. + 59./(24.*60.)*360. + 28./(24.*3600)*360.
            Dec = 40. + 44./60. + 02./3600.
        if self.name_planet == '3C58':
            RA = 2./24.*360. + 05./(24.*60.)*360. + 38./(24.*3600)*360.
            Dec = 64. + 49./60. + 42./3600.
        if self.name_planet == '3C274':
            RA = 12./24.*360. + 30./(24.*60.)*360. + 49./(24.*3600)*360.
            Dec = 12. + 23./60. + 28./3600.
        num = len(new_time)
        radec = np.zeros([2,num])
        radec[0] = np.ones(num)*RA/radeg
        radec[1] = np.ones(num)*Dec/radeg
        return radec

class LB_ptg():
    def __init__(self):
        self.dir_in = './tmp'
        self.dbname = './tmp'
        self.runID = np.arange(1,dtype='int')

    def read_LBptgdb(self):
        num = len(self.runID)
        self.runID_out = []
        self.JD_out = []
        self.fname_out = []
        print(self.dir_in+'/'+self.dbname)
        conn = sq.connect(self.dir_in+'/'+self.dbname)
        c = conn.cursor()
        for i in range(num):
            print('select * from LBSimPtg where id == '+str(self.runID[i])+';')
            c.execute('select * from LBSimPtg where id == '+str(self.runID[i])+';')
            for ar in c:
                self.runID_out.append(ar[0])
                self.JD_out.append(ar[1])
                self.fname_out.append(ar[2])
        c.close()
        return self.runID_out, self.JD_out, self.fname_out

    def read_LBptg(self):
        num = len(self.fname_out)
        self.julian_t0 = []
        self.lat = np.zeros(0); self.lon = np.zeros(0); self.pa = np.zeros(0)
        for i in range(num):
            ptg_tmp = np.load(self.dir_in+'/'+self.fname_out[i]+'.npz')
            self.samplerate = ptg_tmp['sample_rate']
            self.julian_t0.append(ptg_tmp['julian_t0'])
            self.lat = np.hstack((self.lat,ptg_tmp['lat']))
            self.lon = np.hstack((self.lon,ptg_tmp['lon']))
            self.pa = np.hstack((self.pa,ptg_tmp['pa']))
        return self.julian_t0, self.samplerate, self.lat, self.lon, self.pa

    def gen_next1sample(self):
        num = len(self.fname_out[1])
        self.julian_t0 = []
        ptg_tmp = np.load(self.dir_in+'/'+self.fname_out[1]+'.npz')
        ra, dec = EULER(ptg_tmp['lon']*radeg,ptg_tmp['lon']*radeg,4,'J2000')
        return ra[0]/radeg, dec[0]/radeg

    def gen_time(self):
        num = len(self.lon)
        self.time = np.arange(num)/self.samplerate+float(self.JD_out[0]*(24.*3600.))
        return self.time

    def gen_newtime(self,new_samplerate):
        num = len(self.lon)
        num_new = int(num*new_samplerate/self.samplerate)
        time_new = np.arange(num_new)/float(new_samplerate)+float(self.JD_out[0]*(24.*3600.))
        return time_new

    def convert_eclip2radec(self):
        self.ra, self.dec = EULER(self.lon*radeg,self.lat*radeg,4,'J2000')
        return self.ra/radeg, self.dec/radeg

    def convert_elicp2gal(self):
        glon, glat = EULER(self.lon*radeg,self.lat*radeg,4,'J2000')
        return glon/radeg, glat/radeg

class read_LBPtg():
    def __init__(self):
        self.dir_in = './tmp'
        self.dbname = 'tmp.db'
        self.runID = np.array([1,2])

    def read_upsamp_ptg(self, samplerate):
        tmp2 = LB_ptg()
        tmp2.dir_in = self.dir_in
        tmp2.dbname = self.dbname
        tmp2.runID = self.runID

        runid, JD_i, fname = tmp2.read_LBptgdb()
        tmp2.read_LBptg()
        self.time = tmp2.gen_time()
        if np.abs(self.time[1]-self.time[0]) != samplerate:
            time_new = tmp2.gen_newtime(samplerate)
            ra,dec = tmp2.convert_eclip2radec()
            dec_dewrap = dewrap(dec,pi)
            ra_dewrap = dewrap(ra,2.*pi)
            ind = np.where(time_new < self.time[len(self.time)-1]) # take the tail off
            ra_int,dec_int = interp_ptg(time_new[ind[0]],self.time,ra_dewrap,dec_dewrap)
            alpha = scan_direc(ra_int, dec_int)
            return time_new[ind[0]], ra_int, dec_int, alpha, runid, JD_i
        elif np.abs(self.time[1]-self.time[0]) == samplerate:
            ra,dec = tmp2.convert_eclip2radec()
            dec_dewrap = dewrap(dec,pi)
            ra_dewrap = dewrap(ra,2.*pi)
            alpha = scan_direc(ra_dewrap, dec_dewrap)
            return self.time, ra, dec, alpha, runid, JD_i

    def gen_time(self):
        return self.time

def interp_ptg(time_new,time,ra,dec):
    ra_interp = interp1d(time, ra) #, kind='cubic')
    dec_interp = interp1d(time, dec) #, kind='cubic')
    return ra_interp(time_new), dec_interp(time_new)

def interp_time(time,samplerate_new):
    samplerate = 1./(time[1]-time[0])
    num = len(time)
    num_new = int(num*samplerate_new/float(samplerate))
    x = time[0] + np.arange(num_new)/samplerate_new
    return x #ra_interp(time_new)

# def dewrap(lon,wrap):
#     num = len(lon)
#     lon_c = np.copy(lon)
#     lon_out = np.copy(lon)
#     code = '''
#         int i, kp=0, km=0;
#         double diff;
#         for (i=1;i<num;i++){
#            diff = lon_c[i]-lon_c[i-1];
#            if (diff > 0.9*wrap){
#                km-=1;
#            }
#            if (diff < -0.9*wrap){
#                kp+=1;
#            }
#            lon_out[i] += wrap*kp;
#            lon_out[i] -= wrap*km;
# //        printf("%lf %lf %lf %d %d \\n", lon_c[i]*180./3.14, diff*180./3.14, lon_out[i]*180./3.14, kp, km);
#         }
#         '''
#     inline(code,['num','lon_c','lon_out','wrap'])
#     return lon_out

def ang2pos_3D(theta,phi):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def ang2_scandirec_3D(x,y,z):
    xx = np.diff(x)
    yy = np.diff(y)
    zz = np.diff(z)
    return np.array([xx, yy, zz])

def ang2_deriv_theta_3D(theta,phi):
    nb = len(theta)
    x =  np.cos(theta)*np.cos(phi)
    y =  np.cos(theta)*np.sin(phi)
    z = -np.sin(theta)
    return np.array([x[0:nb-1],y[0:nb-1],z[0:nb-1]])

def ang2_deriv_phi_3D(theta,phi):
    nb = len(theta)
    x = -np.sin(phi)
    y =  np.cos(phi)
    z =  np.zeros(nb)
    return np.array([x[0:nb-1],y[0:nb-1],z[0:nb-1]])

def scan_direc(ra,dec):
    num = len(ra)
    nbPix_scan = num - 1
    theta_out = pi/2. - dec
    phi_out = ra
    xp,yp,zp = ang2pos_3D(theta_out,phi_out)
    dpvec = ang2_scandirec_3D(xp,yp,zp)
    dtvec = ang2_deriv_theta_3D(theta_out,phi_out)
    dphivec = ang2_deriv_phi_3D(theta_out,phi_out)
    
    alpha = np.zeros(nbPix_scan)
    alpha = np.arccos( (dpvec[0,:]*dtvec[0,:]+dpvec[1,:]*dtvec[1,:]+dpvec[2,:]*dtvec[2,:])
                       /np.sqrt(dpvec[0,:]*dpvec[0,:]+dpvec[1,:]*dpvec[1,:]+dpvec[2,:]*dpvec[2,:])
                       /np.sqrt(dtvec[0,:]*dtvec[0,:]+dtvec[1,:]*dtvec[1,:]+dtvec[2,:]*dtvec[2,:]) )
    
    beta = np.zeros(nbPix_scan)
    beta = np.arccos( (dpvec[0,:]*dphivec[0,:]+dpvec[1,:]*dphivec[1,:]+dpvec[2,:]*dphivec[2,:])
                      /np.sqrt(dpvec[0,:]*dpvec[0,:]+dpvec[1,:]*dpvec[1,:]+dpvec[2,:]*dpvec[2,:])
                      /np.sqrt(dphivec[0,:]*dphivec[0,:]+dphivec[1,:]*dphivec[1,:]+dphivec[2,:]*dphivec[2,:]) )
    
    for i in range(nbPix_scan):    
        if ((beta[i] >= 90./180.*pi) and (beta[i] < 180./180.*pi)):
            alpha[i] = - alpha[i]


    return alpha

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
    pi = np.pi
    twopi   =   2.0*pi
    fourpi  =   4.0*pi
    deg_to_rad = (180.0/pi)

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
        print('no FK4 is set')

    if ((select<0) or (select>6)):
        print(' ')
        print( ' 1 RA-DEC ' + equinox + ' to Galactic')
        print( ' 2 Galactic       to RA-DEC' + equinox)
        print( ' 3 RA-DEC ' + equinox + ' to Ecliptic')
        print( ' 4 Ecliptic       to RA-DEC' + equinox)
        print( ' 5 Ecliptic       to Galactic')
        print( ' 6 Galactic       to Ecliptic')
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

def ellipGauss(X,Y,x0,y0,sigma_x, sigma_y, theta):
    return np.exp(-0.5*(np.cos(theta)*(X-x0)-np.sin(theta)*(Y-y0))**2/sigma_x**2 \
                       - 0.5*(np.sin(theta)*(X-x0)+np.cos(theta)*(Y-y0))**2/sigma_y**2)

def curvefitfunc_ellipGauss(XY, x0,y0,sigma_x,sigma_y,theta,A):
    X=XY[0]
    Y=XY[1]
    out = A*np.exp(-0.5*(np.cos(theta)*(X-x0)-np.sin(theta)*(Y-y0))**2/sigma_x**2 \
                       - 0.5*(np.sin(theta)*(X-x0)+np.cos(theta)*(Y-y0))**2/sigma_y**2)
    return out.ravel()
    
class ellipticalGaussian():
    def __init__(self):
        self.par = np.array([0.,0.,1.,1.,0.,1.])
        self.resol_rad = 1.
        self.map_width_rad = 10.
        self.beam_sigma = 1.

    def gen_flatellip_map(self):
        x0 = self.par[0]
        y0 = self.par[1]
        sigma_x = self.par[2]
        sigma_y = self.par[3]
        theta = self.par[4]
        A = self.par[5]
        self.x = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.y = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.X,self.Y = np.meshgrid(self.x, self.y)
        self.Z = A*ellipGauss(self.X,self.Y,x0,y0,sigma_x,sigma_y,theta)
        return self.X, self.Y, self.Z

    def gen_flatairy_map(self,freq,radius):
        x0 = self.par[0]
        y0 = self.par[1]
        sigma_x = self.par[2]
        sigma_y = self.par[3]
        theta = self.par[4]
        A = self.par[5]
        self.x = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.y = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.X,self.Y = np.meshgrid(self.x, self.y)
        theta_r = np.sqrt(self.X**2+self.Y**2)
        self.Z = AiryFunc(theta_r,freq,radius)
        return self.X, self.Y, self.Z

    def gen_flatTruncGauss_map(self,freq,edgetaper,radius):
        wavelength = c/freq
        alpha = np.log(edgetaper)/(-2.)
        self.x = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.y = np.arange(-0.5*self.map_width_rad, 0.5*self.map_width_rad, self.resol_rad)
        self.X,self.Y = np.meshgrid(self.x, self.y)
        theta_r = np.sqrt(self.X**2+self.Y**2)
        beta = 2.*pi/wavelength * radius * np.sin(theta_r)
        self.Z = flatTruncGauss(alpha,beta)
        return self.X, self.Y, self.Z

    def gen_flatTruncGauss_2D(self,freq,edgetaper,radius):
        wavelength = c/freq
        alpha = np.log(edgetaper)/(-2.)
        theta_r = np.arange(0.001,90,1e-2)/radeg
        beta = 2.*pi/wavelength * radius * np.sin(theta_r)
        out = flatTruncGauss(alpha,beta)
        return theta_r, out

    def gen_axis(self):
        return self.x, self.y

    def plot_ellipticalGaussian(self):
        xmin, xmax, ymin, ymax = \
            np.amin(self.X*radeg), np.amax(self.X*radeg), np.amin(self.Y*radeg), np.amax(self.Y*radeg)
        extent = xmin, xmax, ymin, ymax
#        im2 = py.imshow(self.Z, alpha=.9, interpolation='bilinear', extent=extent)
        im2 = py.imshow(self.Z, extent=extent)

    def plot_TGAFbeam(self):
        xmin, xmax, ymin, ymax = np.amin(self.X*radeg), np.amax(self.X*radeg), np.amin(self.Y*radeg), np.amax(self.Y*radeg)
        extent = xmin, xmax, ymin, ymax
#        im2 = py.imshow(self.Z, alpha=.9, interpolation='bilinear', extent=extent)
        im2 = py.imshow(self.Z, extent=extent)

    def plot_bin2Dmap(self,tod,ra,dec):
        xout, yout, map, hits = bin2Dmap(tod, ra, dec, self.resol_rad)
        return xout, yout, map, nhits
        
    def fft_2d(self,dx,dy):
        self.Bk = np.fft.fft2(self.Z)
        num = len(self.Bk)
        if num%2 == 0: 
            kx1 = np.arange(0,num/2+1)/float(num)
            kx2 = - np.arange(1,num/2)/float(num)
            self.kx = np.hstack((kx1,kx2[::-1])) / dx
            self.ky = np.hstack((kx1,kx2[::-1])) / dy
        if num%2 == 1: 
            kx1 = np.arange(0,num/2+1/2)/float(num)
            kx2 = np.reverse(- np.arange(1,num/2-1/2)/float(num))
            self.kx = np.hstack((kx1,kx2[::-1])) / dx
            self.ky = np.hstack((kx1,kx2[::-1])) / dy
        return self.Bk, self.kx, self.ky

    def plot_Bk2D(self):
        xmin, xmax, ymin, ymax = np.amin(self.kx), np.amax(self.kx), np.amin(self.ky), np.amax(self.ky)
        extent = xmin, xmax, ymin, ymax
        im2 = py.imshow(np.real(self.Bk), extent=extent)        

    def cal_Bk(self, num, g):
        KX,KY = np.meshgrid(self.kx, self.ky)

        kr = np.sqrt(KX**2 + KY**2)
        xmin, xmax, ymin, ymax = np.amin(KX), np.amax(KX), np.amin(KY), np.amax(KY)
        extent = xmin, xmax, ymin, ymax
        im2 = py.imshow(np.real(kr), extent=extent)

        kr_max = np.amax(kr); kr_min = np.amin(kr)
        kr_del = (kr_max - kr_min)/float(num)
        self.Bkout_kr = []
        self.Bkout_mean = []
        self.Bkout_std = []
        self.Bkout_median = []
        kr_i = np.arange(num)/float(num)*(kr_max-kr_min)+kr_min
        for i in range(num):
            ind = np.where( (kr>=kr_i[i]-kr_del*0.5) & (kr<kr_i[i]+kr_del*0.5) )
            self.Bkout_kr.append(np.mean(kr[ind]))
            self.Bkout_mean.append(np.mean(np.abs(self.Bk[ind])**2))
            self.Bkout_std.append(np.std(np.abs(self.Bk[ind])**2))
            self.Bkout_median.append(np.median(np.abs(self.Bk[ind])**2))

        return self.Bkout_kr, self.Bkout_mean, self.Bkout_std, self.Bkout_median #, self.Bk_fit, par_fit

    def plot_Bk1D(self,label,option=''):
        if option=='norm':
            xx = np.array(self.Bkout_kr)
            yy = np.array(self.Bkout_mean)/max(np.array(self.Bkout_mean))
            eerr = np.array(self.Bkout_std)/max(np.array(self.Bkout_mean))
            ind = np.where(np.isfinite(yy) & np.isfinite(eerr))
            x = xx[ind[0]]
            y = yy[ind[0]]
            err = eerr[ind[0]]
            py.subplot(121)
            py.errorbar(x,y,yerr=err,fmt='o')
#            py.plot(x,self.Bk_fit[ind[0]],'-r')
            py.ylabel('Normalized $B_k$')
            py.xlabel('k [1/rad]')
            py.title(label)
            py.subplot(122)
            py.errorbar(x,y,yerr=err,fmt='o')
#            py.plot(x,self.Bk_fit[ind[0]],'-r')
            py.ylabel('Normalized $B_k$')
            py.xlabel('k [1/rad]')
            py.semilogy()
        if option=='':
            py.errorbar(x,y,yerr=err, fmt='o')
            py.xlabel('k [1/deg]')

'''
 [lib_planetcal.py: fft_2d] 
 input: dx, dy in [rad]
         Z (n x m) matrix
'''
def fft_2d(dx,dy,Z):
    Bk = np.fft.fft2(Z)
    num = len(Bk)
    if num%2 == 0:
        kx1 = np.arange(0,num/2+1)/float(num)
        kx2 = - np.arange(1,num/2)/float(num)
        kx = np.hstack((kx1,kx2[::-1])) / dx * (2.*pi)
        ky = np.hstack((kx1,kx2[::-1])) / dy * (2.*pi)
    if num%2 == 1:
        kx1 = np.arange(0,num/2+1/2)/float(num)
        tmp = - np.arange(1,num/2-1/2)/float(num)
        kx2 = tmp[::-1]
        kx = np.hstack((kx1,kx2[::-1])) / dx * (2.*pi)
        ky = np.hstack((kx1,kx2[::-1])) / dy * (2.*pi)
    return Bk, kx, ky

def cal_Bk(num, kx,ky,Bk):
    KX,KY = np.meshgrid(kx, ky)
    
    kr = np.sqrt(KX**2 + KY**2)
    xmin, xmax, ymin, ymax = np.amin(KX), np.amax(KX), np.amin(KY), np.amax(KY)
    extent = xmin, xmax, ymin, ymax
#    im2 = py.imshow(np.real(kr), extent=extent)
    
    kr_max = np.amax(kr); kr_min = np.amin(kr)
    kr_del = (kr_max - kr_min)/float(num)
    Bkout_kr = []
    Bkout_mean = []
    Bkout_std = []
    Bkout_median = []
    kr_i = np.arange(num)/float(num)*(kr_max-kr_min)+kr_min
    for i in range(num):
        ind = np.where( (kr>=kr_i[i]-kr_del*0.5) & (kr<kr_i[i]+kr_del*0.5) )
        if len(ind[0]) ==0: 
            continue
        Bkout_kr.append(np.mean(kr[ind]))
        Bkout_mean.append(np.mean(Bk[ind]))
        Bkout_std.append(np.std(Bk[ind])/np.sqrt(len(ind)))
        Bkout_median.append(np.median(Bk[ind]))
    return Bkout_kr, Bkout_mean, Bkout_std, Bkout_median #, self.Bk_fit, par_fit


def minimize_fit1DGaussian(par,x,y_data,y_err):
    y_model = fit1DGaussian(par,x)
    out = np.sum((y_data - y_model)**2/y_err**2)
    return out

def fit1DGaussian(par,x):
    y_model = np.exp(- par * x**2)
    return y_model

class plot_3DEllipGaussian():
    def __init__(self):
        self.ra_dif = np.arange(10)
        self.dec_dif = np.arange(10)
        self.par = np.arange(5)
        self.label = ''
        self.range = [0,1]
    def plot(self):
        mpl.rcParams['legend.fontsize'] = 10
        fig = py.figure()
#        ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        z = self.par[5]*ellipGauss( self.ra_dif, self.dec_dif, \
                            self.par[0], self.par[1], \
                            self.par[2], self.par[3], \
                            self.par[4])
        ax.plot(self.ra_dif*radeg, self.dec_dif*radeg, z, '.', markersize=2, label=self.label)
#        ax.set_xlim3d(self.range[0],self.range[1])
#        ax.set_ylim3d(self.range[0],self.range[1])
#        ax.set_title(self.label)
#        ax.set_zlim3d(0, 10)
#        ax.set_xlabel('X axis')
#        ax.set_ylabel('Y axis')
#        ax.set_zlabel('Z axis')
#        ax.text2D(0.05, 0.95, self.label, transform=ax.transAxes)
        ax.legend()

def Bk1D_fit(x, y, yerr, beam_sigma):
    num = len(x)
    par_init = beam_sigma**2 *(1.+np.random.normal(0.,1.,1))
    par_fit = fmin(minimize_fit1DGaussian, par_init, \
                       args=( np.array(x[1:num]), \
                                  np.array(y[1:num]/max(y[1:num])), \
                                  np.array(y[1:num]/max(yerr[1:num])) ), \
                       maxiter=10000,maxfun=10000,xtol=0.001 )
    Bk_fit = fit1DGaussian(par_fit,np.array(x))
    return Bk_fit, par_fit

def gen_TOD(ipix_x,ipix_y,Z):
    tod = Z[ipix_x,ipix_y]
#    num = len(ipix_x)
#    tod = np.zeros(num)
#    code = """
#    long i, j;
#    for (i=0;i<num;i++){
#       tod[i] = Z[ipix_x[i],ipix_y[i]];
#    }
#    """
#    inline(code,['ipix_x','ipix_y','num','Z','tod'])
    return tod
#######################################################
def fit_beam(par_init,ipix_x,ipix_y,tod_data,noise,g):
    elip = ellipticalGaussian()
    elip.resol_rad = g.resol_rad
    elip.map_width_rad = g.map_width_rad
    elip.par = par_init
    X, Y, Z = elip.gen_flatellip_map()
    model = gen_TOD(ipix_x,ipix_y,Z)
    chi2 = chi2_ellipbeam(tod_data,model,noise)
    return chi2

def chi2_ellipbeam(tod_data,model,noise):
    chi2 = np.sum((tod_data - model)**2/noise**2)
    return chi2

def fit_elipbeam(par_init,ipix_x,ipix_y,pa,tod_data,noise,g):
    elip = ellipticalGaussian()
    elip.resol_rad = g.resol_rad
    elip.map_width_rad = g.map_width_rad
    par_init[4] = par_init[4] - pa
    elip.par = par_init
    X, Y, Z = elip.gen_flatellip_map()
    model = gen_TOD(ipix_x,ipix_y,Z)
    chi2 = chi2_ellipbeam(tod_data,model,noise)
    return chi2
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
ang1D2pix2(x,x_min,x_max,res)                                       
Same functionality to ang2pix in healpix routine       
INPUT                                             
x: array of angles (ra or dec of pointing)
res: bin resolution
'''
def ang1D2pix2(x,x_min,x_max,res):
    num = int((x_max - x_min)/res)
    width = x_max-x_min
    nbData = len(x)
    if nbData == 0: 
        print('[ang1D2pix2] No sample falls within the range.')
        return 0, 0
    n = np.zeros(nbData,int)
    ind = np.where((x<2.*pi+width/2.) & (x>2*pi-width/2.))
    x[ind[0]] = x[ind[0]]-2.*pi
    n = np.int_((x-x_min)/res)
    return np.array(n), np.max(n)+1
#######################################################
'''
bin2Dmap(tod, x, y, res)
  INPUT
    tod: time stream
    x, y: corresponding pointing (eg. ra, dec)
    res: resolution to bin in degree
  OUTPUT
    Xout, Yout, map
  Use the output to show ra dec 2D map
  
    py.pcolor(Xout,Yout,map)
    py.colorbar()
'''
def bin2Dmap(tod, ra, dec, res):
    nbData = len(tod)
    res = res/180.*pi
    nx,nbx=ang1D2pix(ra,res)
    ny,nby=ang1D2pix(dec,res)
    map = np.zeros((nby,nbx))
    hits = np.zeros((nby,nbx),int)
    xx = np.zeros(nbx)
    yy = np.zeros(nby)
    for i in range(0,nbData):
        map[ny[i],nx[i]] += tod[i]
        hits[ny[i],nx[i]] += 1
        xx[nx[i]] = ra[i]
        yy[ny[i]] = dec[i]
        Xout,Yout=py.meshgrid(xx,yy)
    return Xout/pi*180., Yout/pi*180., map, hits

def gen_wnoise(NET_rtsec,num,samplerate):
    return NET_rtsec * np.sqrt(samplerate) * np.random.normal(0.,1.,num)

def gen_1ofnoise(NET_rtsec,fknee,alpha,num,samplerate,seed):
    out = TODgen()
    out.net = NET_rtsec*np.sqrt(samplerate)
    out.fknee = fknee
    out.power = alpha
    out.seed = seed
    out.fsample = samplerate
    out.nbData = num
    out.gen_basic()
    freq = out.gen_freq()
    num, noise_t = out.NoiseGen_auto()
    return noise_t

def plot_2D(x,y,z):
    xx, yy = np.meshgrid(x,y)
#    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
#    extent = xmin, xmax, ymin, ymax
    im2 = py.imshow(z) #, extent=extent)
    py.imshow(z)

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
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print( '+++  Fit the histogram with Gaussian +++')
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print( 'initial guess:', par0)
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        par, fopt,iterout,funcalls,warnflag=fmin(chi_nosigma,par0,args=(bincenters,non),maxiter=2000,maxfun=10000,xtol=0.01,full_output=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),'r', linewidth=1)
#        if no_plot == False: py.plot(bincenters,func_gauss(par,bincenters),'r', linewidth=1)
        #y = mlab.normpdf(bincenters, par[1], par[2])
        #l = py.plot(bincenters, y, 'r--', linewidth=1)
        print( 'fitted parameters:', par)
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
    py.grid(True)

#    py.show()

    return np.array(par)

def DelTcmb2DelTantenna( T, nu):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    x = h/k*nu/T
    Bnu = x**2. * np.exp(x)/(np.exp(x)-1.)**2.
    return Bnu

def antennaT2flux(T_A, nu):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    flux = 2.*nu**2*k*T_A/c**2
    return flux

def dBBdT( T, nu):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    dBBdT = 2.*(h*nu**3.)/c**2./(np.exp(h/k*nu/T)-1.)**2.*(-h/k*nu/T**2)*np.exp(h/k*nu/T)
    return dBBdT

def antennaT2Power(T_A, nu_arr):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    AOmega = (c/nu_arr)**2
    power = 2.*k*T_A/c**2 * np.sum(nu_arr**2.*AOmega)*(nu_arr[1]-nu_arr[0])
    return power

def dPowerdT( T, nu_arr):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    AOmega = (c/nu_arr)**2
    dPdT = np.sum(2.*(h*nu_arr**3.)/c**2. \
                      /(np.exp(h/k*nu_arr/T)-1.)**2. \
                      *(-h/k*nu_arr/T**2)*np.exp(h/k*nu_arr/T) * AOmega) \
                      *(nu_arr[1]-nu_arr[0])
    return dPdT

def AiryFunc(theta,freq,radius):
    k = 2.*pi/c*freq
    beta = k*radius*np.sin(theta)
    airy = (2.*sp.jv(1.,beta)/beta)**2
    return airy

def flatTruncGauss(alpha, beta):
    sumout = np.zeros(len(beta))
    m_max = 10
    n_max = 10
    for m in range(1,m_max):
        for n in range(1,n_max):
            sumout = sumout + (2.*alpha)**(m+n)*sp.jv(m,beta)/beta**m * sp.jv(n,beta)/beta**n
#    print sumout
    out = np.exp(-2.*alpha)/(1.-np.exp(-alpha))**2 * sumout
    return out

def beam_solidangle_AirySymmetric(freq,Diameter):
    radius = Diameter/2.
    num = 10000
    theta = np.arange(num)/float(num)*pi/10. + 0.0000001
    #    return the solidangle in steradian
    return 2.*pi*np.sum(np.sin(theta)*AiryFunc(theta,freq,radius))*(theta[1]-theta[0])

def beam_solidangle_PvsTheta(theta,PvsTheta):
    #    return the solidangle in steradian
    return 2.*pi*np.sum(np.sin(theta)*PvsTheta(theta))*(theta[1]-theta[0])

def DelTcmb2DelTantenna( T_thermo, nu):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    Tcmb = 2.725
    x = h/k*nu/Tcmb
    Bnu = x**2. * np.exp(x)/(np.exp(x)-1.)**2.
    return Bnu*T_thermo

def DelTantenna2DelTcmb( T_antenna, nu):
    h = 6.626068e-34
    k = 1.3806503e-23
    c = 2.99792458e8 
    Tcmb = 2.725
    x = h/k*nu/Tcmb
    Bnu = x**2. * np.exp(x)/(np.exp(x)-1.)**2.
    return (1./Bnu)*T_antenna
    
# below the information is coming from Weiland et al. (WMAP7)
def planet_info(src_planet, nu_obs, sigma_r_rad, details=True):
    ''' src_planet: Jupiter, Mars, Saturn
        nu_obs: in unit of Hz, e.g. 100e9 Hz
        sigma_r: beam size
    '''

    # below the information is coming from Weiland et al. (WMAP7)
    if src_planet == 'Jupiter':
        T_antenna = 170.
        planet_apparent_str = 2.481e-8

    if src_planet == 'Mars':
        T_antenna = 200.
        planet_apparent_str = 2.*np.arccos(1.-7.153e-10/(2.*pi))

    if src_planet == 'Saturn':
        T_antenna = 150.
        planet_apparent_str = 2.*np.arccos(1.-5.096e-9/(2.*pi))

    if src_planet == 'Uranus':
        T_antenna = 110.; 
        solid_ref = 2.482e-10

    if src_planet == 'Neptune':
        T_antenna = 150.; 
        solid_ref = 1.006e-10

    planet_apparentfullangle_rad = 2.*np.arccos(1.-planet_apparent_str/(2.*pi))
    beam_appearnt_str = 2.*pi*(1.-np.cos(sigma_r_rad))
    A_original = DelTantenna2DelTcmb( T_antenna, nu_obs)
    A = A_original*1e6 * (planet_apparent_str/beam_appearnt_str)

    if details==True:
        print('+++++++++++++++++++++++')
        print( '    Band:', nu_obs*1e-9, 'GHz')
        print( '    FWHM:', np.sqrt(8.*np.log(2))*sigma_r_rad*10800./pi, 'arcmin')
        print( '    sigma_r:', sigma_r, 'rad')
        #print '    np.sqrt(sigma_x**2+sigma_y**2):', np.sqrt(sigma_x**2+sigma_y**2), 'rad'
        print( '   src_planet:', src_planet)
        print( '            A=', A_original, 'Kcmb') # convert to uK
        print( ' Planet appearent str', planet_apparent_str)
        print( ' LB beam str', beam_appearnt_str)
        print( ' beam dilution factor', planet_apparent_str/beam_appearnt_str)
        print( '   apparent planet diameter in arcsec', planet_apparentfullangle_rad * radeg * 3600.)
        #print '  beam dilution ratio:', (apparent/np.sqrt(sigma_x**2+sigma_y**2))**2
        print( ' appearent amplitude of a planet in uKcmb', A) # convert to uK in apparent amplitude

    return A

# below the information is coming from Weiland et al. (WMAP7)
#def planet_info(src_planet,nu_obs,sigma_x,sigma_y):
# def planet_info(src_planet,nu_obs,beam_solid_str):
#     Tcmb = 2.725
#     num_nu = 1000
#     bandwidth = 0.3
#     nu_obs_arr = np.arange(num_nu)/float(num_nu)*bandwidth*nu_obs + nu_obs*(1.-bandwidth/2.)

#     if src_planet== 'Jupiter':
#         T_antenna = 173.5; 
#         solid_ref =  2.481e-8
#         beam_dil = solid_ref / beam_solid_str
#         P = antennaT2Power(T_antenna, nu_obs_arr)
#         delT = 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)) * P * beam_dil
#         print ''
#         print '--in func--'
#         print src_planet,nu_obs,beam_solid_str
#         print 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)), P, 1./np.abs(dPowerdT(Tcmb,nu_obs_arr))*P, beam_dil, delT, delT * 1.e6
#         print ''
#         A = delT * 1.e6
            
#     if src_planet == 'Mars':
#         T_antenna = 200.; 
#         solid_ref = 7.153e-10
#         beam_dil = solid_ref / beam_solid_str
#         P = antennaT2Power(T_antenna, nu_obs_arr)
#         delT = 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)) * P * beam_dil
#         A = delT * 1.e6 
        
#     if src_planet == 'Saturn':
#         T_antenna = 150.; 
#         solid_ref = 5.096e-9
#         beam_dil = solid_ref / beam_solid_str
#         P = antennaT2Power(T_antenna, nu_obs_arr)
#         delT = 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)) * P * beam_dil
#         A = delT * 1.e6 

#     if src_planet == 'Uranus':
#         T_antenna = 110.; 
#         solid_ref = 2.482e-10
#         beam_dil = solid_ref / beam_solid_str
#         P = antennaT2Power(T_antenna, nu_obs_arr)
#         delT = 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)) * P * beam_dil
#         A = delT * 1.e6 

#     if src_planet == 'Neptune':
#         T_antenna = 150.; 
#         solid_ref = 1.006e-10
#         beam_dil = solid_ref / beam_solid_str
#         P = antennaT2Power(T_antenna, nu_obs_arr) 
#         delT = 1./np.abs(dPowerdT(Tcmb,nu_obs_arr)) * P * beam_dil
#         A = delT * 1.e6
#     return A

def gen_InputParams(nu_obs,Dapt_mm):
    if nu_obs == 60.e9: beam_arcmin = 75. * 300./Dapt_mm
    if nu_obs == 78.e9: beam_arcmin = 58. * 300./Dapt_mm
    if nu_obs == 94.e9: beam_arcmin = 5.8 # wmap w-band from 2e-5 str
    if nu_obs == 100.e9: beam_arcmin = 45. * 300./Dapt_mm
    if nu_obs == 140.e9: beam_arcmin = 32. * 300./Dapt_mm
    if nu_obs == 195.e9: beam_arcmin = 24. * 300./Dapt_mm
    if nu_obs == 280.e9: beam_arcmin = 16. * 300./Dapt_mm

    x0 = 0.
    y0 = 0.
    sigma_x = (beam_arcmin/np.sqrt(8.*np.log(2.))/60./radeg)
    sigma_y = (beam_arcmin/np.sqrt(8.*np.log(2.))/60./radeg)
    phi = (0./radeg)
    
    samplerate = 10.*Dapt_mm/300. # in unit of Hz

    return x0, y0, sigma_x, sigma_y, phi, samplerate


def weightedMean(x,w):
    weighted_mean = np.sum(x*w)/np.sum(w) * np.ones(len(x))
    err = np.sqrt(1./np.sum(w)) * np.ones(len(x))
    return weighted_mean, err


class TODgen():
    def __init__(self):
        self.net = 50.
        self.fknee = 0.1
        self.power = 2.
        self.seed = 1
        self.fsample = 10.
        self.nbData = 100.

    def gen_basic(self):
        self.nbf = int((self.nbData)/2.)+1
        self.freq = np.arange(self.nbf)/np.double(self.nbf-1)*(self.fsample)/2.
        self.delta = 1./self.fsample

    def gen_time(self):
        self.time_axis = np.arange(self.nbData)*self.delta
        return self.time_axis

    def gen_freq(self):
        return self.freq

    def gen_ell(self, scanspeed):
        self.scanspeed = scanspeed # degs/sec
        ell = 180./(self.scanspeed*self.time_axis)   # 180/theta
        return ell
    
    def NoiseGen_auto(self):
        psd = np.sqrt(self.net**2 * (1.+(self.fknee/self.freq)**self.power))
# if nbData = 10000
#        0, 1     2,          4999,     5000,    5001,          , 9999
#        0, 1,    2,    ..., N/2-1,      N/2,   N/2+1,          , N-1
#        0, 1,    2,    ..., nbf-2,    nbf-1,     nbf,          , 2*(nbf-1)
#        0, (        idx         ), 
#        0, 1/NT, 2/NT, ..., (N/2-1)/NT, 1/(2T), -(N/2-1)/NT, ..., -1/NT

        idx = range(1,self.nbf-1) #
        real_psd = np.concatenate(( psd[idx], np.array([psd[self.nbf-1]]), psd[idx[::-1]] ))
        real_psd = np.hstack([0,real_psd])
        real_psd[self.nbf-1] = 0.

        np.random.seed(self.seed)
        f_rand = np.random.uniform(low=0.0, high=2.*pi, size=self.nbData)
        real_psdout = real_psd*np.cos(f_rand)
        imag_psdout = real_psd*np.sin(f_rand)
        psd_complex = complex_arr(real_psdout,imag_psdout)
        top_noise = fft.ifft(psd_complex)
        # taking only the real part is effectively dividing 2
#        if self.fknee != 0:
        top_noise = np.array(top_noise.real)*np.sqrt(float(self.nbData)) * 2. /np.sqrt( 2.)
        # + self.net*random.randn(self.nbData)
        return len(top_noise.real), np.array(top_noise.real)

    def Plot_NoiseGen_auto(self, outputunit, logx=False, logy=False, breakdown=False):
        psd_net2 = self.net**2 * np.ones(self.nbf)
        psd_1of = self.net**2 * (self.fknee/self.freq)**self.power
        psd_plot = psd_net2 + psd_1of

        if outputunit==1:
            psd_plot = psd_plot*self.nbData
            psd_net2_plot = psd_net2*self.nbData
            psd_1of_plot = psd_1of*self.nbData
        if outputunit==2:
            psd_plot = psd_plot
            psd_net2_plot = psd_net2
            psd_1of_plot = psd_1of
        if outputunit==3:
            psd_plot = psd_plot*self.delta*self.nbData
            psd_net2_plot = psd_net2*self.delta*self.nbData
            psd_1of_plot = psd_1of*self.delta*self.nbData
        if outputunit==4:
            psd_plot = np.sqrt(psd_plot*self.nbData)
            psd_net2_plot = np.sqrt(psd_net2*self.nbData)
            psd_1of_plot = np.sqrt(psd_1of*self.nbData)
        if outputunit==5:
            psd_plot = np.sqrt(psd_plot)
            psd_net2_plot = np.sqrt(psd_net2)
            psd_1of_plot = np.sqrt(psd_1of)
        if outputunit==6:
            psd_plot = np.sqrt(psd_plot*self.delta*self.nbData)
            psd_net2_plot = np.sqrt(psd_net2*self.delta*self.nbData)
            psd_1of_plot = np.sqrt(psd_1of*self.delta*self.nbData)

        py.plot(self.freq, psd_plot)
        if breakdown==True: py.plot(self.freq, psd_net2_plot)
        if breakdown==True: py.plot(self.freq, psd_1of_plot)
        if logx==True and logy==False: py.semilogx()
        if logy==True and logx==False: py.semilogy()
        if logx==True and logy==True: py.loglog()
#        py.show()

class CalPSD():
    def __init__(self, todin, fsample):
        self.todin = todin
        self.fsample = fsample
        self.nbData = len(todin)
        self.nbf = int((self.nbData)/2.)+1
        self.nrowh = int((self.nbData)/2.)
        self.freq = np.arange(self.nbf)/np.double(self.nbf-1)*(self.fsample)/2.
        self.delta = 1./fsample

    def gen_freq(self):
        return self.freq

    def calPSD(self, outputunit):
        self.outputunit = outputunit
        PSD = np.zeros(self.nrowh+1)
        Ck = abs(fft.fft(self.todin)) 
        PSD[0] = Ck[0]**2 / np.double(self.nbData**2.)       #;PSD at DC
        PSD[self.nrowh] = Ck[self.nrowh]**2 / np.double(self.nbData**2.) #; PSD at Nyquist freq
    
        for i in range(1,self.nrowh): #i=1L, nrowh-1L do begin
            PSD[i] = (Ck[i]**2 + Ck[self.nbData-i]**2) / np.double(self.nbData**2.)

        if (self.outputunit == 1): PSD = PSD*self.nbData
        if (self.outputunit == 2): PSD = PSD
        if (self.outputunit == 3): PSD = PSD*self.delta*self.nbData
        if (self.outputunit == 4): PSD = PSD*self.delta
        
        if (self.outputunit == 5): PSD = np.sqrt(PSD*self.nbData)
        if (self.outputunit == 6): PSD = np.sqrt(PSD)
        if (self.outputunit == 7): PSD = np.sqrt(PSD*self.delta*self.nbData)
        if (self.outputunit == 8): PSD = np.sqrt(PSD*self.delta)

        if (self.outputunit == 9): PSD = np.sqrt(PSD)/mean(np.sqrt(PSD[0:4]))

        self.PSD = PSD
        return np.array(PSD)

    def define_parin(self,parin):
        self.parin = parin

    def plot_PSD(self, logx=False, logy=False, fit=False):
        py.plot(self.freq, self.PSD)
        if logx==True and logy==False: py.semilogx()
        if logy==True and logx==False: py.semilogy()
        if logx==True and logy==True: py.loglog()
        if fit==True:
            if self.outputunit < 5:
                parin = np.array([np.sqrt(self.parin[0]),self.parin[1],self.parin[2]])
                func_1of = lambda p, xin: p[0]**2*(1.+(p[1]/xin)**p[2])
            if self.outputunit >= 5:
                parin = np.array([self.parin[0],self.parin[1],self.parin[2]])
                func_1of = lambda p, xin: np.sqrt(p[0]**2*(1.+(p[1]/xin)**p[2]))
            chi_nosigma = lambda p, xin, d: ((func_1of(p,xin)-d)**2).sum()
            parout=fmin(chi_nosigma,parin,args=(self.freq,self.PSD),maxiter=2000,maxfun=10000,xtol=0.01)
            py.plot(self.freq,func_1of(parout,self.freq),'-r')
            return parout
#        py.show()


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
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print( '+++  Fit the histogram with Gaussian +++')
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print( 'initial guess:', par0)
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        par, fopt,iterout,funcalls,warnflag=fmin(chi_nosigma,par0,args=(bincenters,non),maxiter=10000,maxfun=10000,xtol=0.01,full_output=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),'r', linewidth=1)
#        if no_plot == False: py.plot(bincenters,func_gauss(par,bincenters),'r', linewidth=1)
        #y = mlab.normpdf(bincenters, par[1], par[2])
        #l = py.plot(bincenters, y, 'r--', linewidth=1)
        print( 'fitted parameters:', par)
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
    py.grid(True)

#    py.show()

    return np.array(par)

def beam_params_LBnote25(band_index):
    '''
        Check also LiteBIRD note #20.
    '''
    # index, band GHz, pixel, FWHM degs, A, B, C, D, E, F, crossing angle in deg
    if band_index == 1: par = np.array([1, 40, 30.0, 1.151213, 42.414715, -8.210130, -2.642630, 26.85410, -2.962870, 1.790068e-3, 1.221568 ])
    if band_index == 2: par = np.array([2, 50, 30.0, 0.951332, 45.191733, -12.04290, -5.576840, 25.60290, -2.969680, 1.331703e-3, 1.054150 ])
    if band_index == 3: par = np.array([3, 60, 30.0, 0.827540, 47.111145, -15.88390, -9.924060, 23.45920, -2.959210, 8.306155e-4, 0.9611415 ]) 
    if band_index == 4: par = np.array([4, 68, 30.0, 0.752678, 48.215969, -19.48310, -12.50710, 21.41130, -2.955750, 5.195117e-4, 0.9246329  ])
    if band_index == 5: par = np.array([5, 78, 30.0, 0.686078, 49.179578, -23.82650, -14.91230, 18.21780, -2.945130, 2.546737e-4, 0.9072797  ])
    if band_index == 6: par = np.array([6, 89, 30.0, 0.635835, 49.869756, -28.27300, -14.94860, 14.38570, -2.950970, 1.045138e-4, 0.9173471  ])
    if band_index == 7: par = np.array([7, 68, 18.0, 0.685500, 47.063467, -22.85730, -23.55510, 24.55000, -2.963020, 1.055691e-3, 0.7221173  ])
    if band_index == 8: par = np.array([8, 78, 18.0, 0.607320, 48.837312, -29.26370, -36.68410, 23.86840, -2.967010, 8.951047e-4, 0.6584738  ])
    if band_index == 9: par = np.array([9, 89, 18.0, 0.544341, 50.344897, -36.54990, -55.18130, 22.77170, -2.968990, 6.920898e-4, 0.6094436  ])
    if band_index == 10: par = np.array([10, 100, 18.0, 0.497153, 51.535607, -43.90380, -77.91140, 21.35990, -2.968730, 5.001038e-4, 0.5750968  ])
    if band_index == 11: par = np.array([11, 119, 18.0, 0.435342, 52.998024, -58.82040, -99.48800, 18.49050, -2.969740, 2.583253e-4, 0.5490679  ])
    if band_index == 12: par = np.array([12, 140, 18.0, 0.393605, 53.980168, -73.37160, -112.3490, 14.40750, -2.962450, 1.024772e-4, 0.5454522  ])
    if band_index == 13: par = np.array([13, 100, 12.0, 0.622009, 47.823867, -27.58230, -36.60280, 24.15160, -2.965280, 9.589551e-4, 0.6488819  ])
    if band_index == 14: par = np.array([14, 119, 12.0, 0.526998, 50.070640, -39.42940, -56.55770, 23.34550, -2.976730, 7.782928e-4, 0.5802754  ])
    if band_index == 15: par = np.array([15, 140, 12.0, 0.462321, 51.867803, -51.31920, -93.87680, 21.84470, -2.977200, 5.505875e-4, 0.5298993  ])
    if band_index == 16: par = np.array([16, 166, 12.0, 0.406232, 53.445069, -67.30340, -137.2620, 19.39190, -2.974620, 3.145551e-4, 0.4971988  ])
    if band_index == 17: par = np.array([17, 195, 12.0, 0.365010, 54.598336, -84.67660, -171.1620, 15.95050, -2.967470, 1.446127e-4, 0.4845037  ])
    if band_index == 18: par = np.array([18, 235, 12.0, 0.330527, 55.486262, -105.5770, -169.9600, 10.45530, -2.978080, 4.009473e-5, 0.4941475  ])
    if band_index == 19: par = np.array([19, 280, 5.2, 0.218559, 56.866803, -230.2250, -1829.830, 19.87910, -2.982110, 3.470806e-4, 0.2345765  ])
    if band_index == 20: par = np.array([20, 337, 5.2, 0.187673, 59.186542, -308.8270, -3753.200, 18.80370, -2.983450, 2.704545e-4, 0.2062461  ])
    if band_index == 21: par = np.array([21, 402, 5.2, 0.162053, 61.080655, -421.8090, -5590.800, 16.96070, -2.981430, 1.774943e-4, 0.1904191 ])
    return par

def beam_LBnote25(theta_deg,par):
    num = len(theta_deg)
    Pmain = np.zeros(num)

    A=par[4]; B=par[5]; C=par[6]; D=par[7]; E=par[8]; F=par[9]; theta_crs=par[10]

    ind_i = np.where( theta_deg < theta_crs)
    num_i = len(ind_i[0])

    Pout = np.zeros(num)
    Pout[ind_i[0]] = 10.**((A+B*theta_deg[ind_i[0]]**2+C*theta_deg[ind_i[0]]**4)/10.)
    Pout[num_i-1::] = 10.**(D/10.)*theta_deg[num_i-1::]**E + F

    return Pout

def TGAFbeam_LBnote25_xy(x_deg,y_deg,par):
    ''' 
        this was checked and confirmed.
        make sure to implement any change here to above beam_LBnote25 when it is used for a serious calculation.
    '''
    num = len(x_deg)
    Pmain = np.zeros((num,num))

    A=par[4]; B=par[5]; C=par[6]; D=par[7]; E=par[8]; F=par[9]; theta_crs=par[10]

    theta_deg = np.sqrt(x_deg**2+y_deg**2)
    Pout = np.zeros((num,num))

#    print ''
#    print ''
#    print ''
    for i in range(num):
#        print ''
        ind_in = np.where( theta_deg[i,:] < theta_crs )
        ind_out = np.where( theta_deg[i,:] >= theta_crs )
        num_in = len(ind_in[0])
        num_out = len(ind_out[0])
#        print i, num, theta_deg[i,:], ind_in, num_in, len(ind_in[0])+len(ind_out[0])
        if num_in == 0: 
            Pout[i,:] = 10.**(D/10.)*theta_deg[i,:]**E + F
        elif num_out == 0:
            Pout[i,:] = 10.**((A+B*theta_deg[i,:]**2+C*theta_deg[i,:]**4)/10.)
        else:
            Pout[i,ind_in[0]] = 10.**((A+B*theta_deg[i,ind_in[0]]**2+C*theta_deg[i,ind_in[0]]**4)/10.)
            Pout[i,ind_out[0]] = 10.**(D/10.)*theta_deg[i,ind_out[0]]**E + F
#        print '->', max(Pout[i,:])
#    print 'max', np.max(Pout[:,:])
    ind = np.where(Pout[:,:] == np.max(Pout[:,:]))
#    print ind
#    print ind[0]
#    print x_deg[ind[0],ind[0]]
#    print y_deg[ind[0]]
#    sys.exit()
    return Pout



