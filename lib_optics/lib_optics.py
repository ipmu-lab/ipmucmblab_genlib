import numpy as np
import pylab as py
from scipy.optimize import fmin
from scipy.optimize import curve_fit
import sys
from scipy.interpolate import interp1d
#import matsumulib as mylib

pi = np.pi
#c = 3.e8                         # [m/s]
c = 299792458.                         # [m/s]
ep0 = 8.8542e-12   # [Fm^-1]=[C^2 N^-1 m^-2] dielectric const in vaccum

#################################################################################################################################
######### Multi-reflection related codes

def chisq_FitIndex_1layer_RvsFreq(par, d, freq, angle_i, incpol, data_trans, data_transerr):
    n = np.array([1.,par,1.])
    RT = fit_oblique_basic_multilayer_r_t( n, d, freq, angle_i, incpol)
    chisq = np.sum((data_trans - np.abs(RT[1])**2)**2/data_transerr**2)
    return chisq

def chisq_FitIndex_1layer_TvsFreq(par, d, freq, angle_i, incpol, data_trans, data_transerr):
    n = np.array([1.,par,1.])
    RT = fit_oblique_basic_multilayer_r_t( n, d, freq, angle_i, incpol)
    chisq = np.sum((data_trans - np.abs(RT[2])**2)**2/data_transerr**2)
    return chisq

def fit_oblique_basic_multilayer_r_t( n, d, freq_in, angle_i, incpol):
    '''
        input:
            n: array of indices  
            d: array of thickness in m
            freq: array of frequency in Hz
            angle_i: incident angle in rad, scalar
            incpol: 1 or -1 
                1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident
        output:
            freq, reflection, transmission

        Example:
            n_arr = np.array([1.,3.,1.])
            d_arr = np.array([2.e-3])
            freq_in = np.arange(10e9,500e9,1e9)
            angle_i = 0.
            incpol = 1
            output = lib_o.fit_oblique_basic_multilayer_r_t( n_arr, d_arr, freq_in, angle_i, incpol)

            py.plot(output[0]*1e-9,np.abs(output[1])**2,label='reflectance')
            py.plot(output[0]*1e-9,np.abs(output[2])**2,label='transmittance')
            py.ylim([0,1.1])
            py.xlabel('Frequency [GHz]')
            py.grid()
            py.legend(loc='best')
            py.show()
    '''

    num=len(d) #; the number of layer not including two ends                                                                                       
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)                                                          

    # ;-----------------------------------------------------------------------------------                                                         
    # ; angle of refraction                                                                                                                        
    angle = np.zeros(num+2)          # ; angle[0]=incident angle                                                                                   
    angle[0] = angle_i
    for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------                                                         
    # ; define the frequency span                                                                                                                  
    l = len(freq_in)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)                                                                                 

    # ;-----------------------------------------------------------------------------------                                                         
    # ; define the effective thickness of each layer                                                                                               
    h = np.zeros(num)
    for i in range(0,num): h[i] = n[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer                                               

    # ;-----------------------------------------------------------------------------------                                                         
    # ; for loop for various thickness of air gap between each layer                                                                               
    for j in range(0,l):
        freq = freq_in[j]
        k = 2.*pi*freq/c

        # ;===========================================                                                                                             
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side                                                                                   
        Y = np.zeros(num+2)
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = const*n[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = const*n[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ;===========================================                                                                                             
        # ; define matrix for single layer                                                                                                         
        m = np.identity((2),'complex')    # ; net matrix                                                                                           
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...                                                                                    
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

        output[0,j] = freq+0.j #; unit of [Hz]                                                                                                     
        output[1,j] = r
        output[2,j] = t

    return output

def fit_oblique_basic_multilayer_r_t_incloss( n, losstan, d, freq_in, angle_i, incpol):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; angle of refraction
    angle = np.zeros(num+2)          # ; angle[0]=incident angle
    angle[0] = angle_i
    for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    l = len(freq_in)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; define the effective thickness of each layer
    h = np.zeros(num,'complex')
    n_comparr = np.zeros(len(n),'complex')
    n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
    n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

    # ;-----------------------------------------------------------------------------------
    # ; for loop for various thickness of air gap between each layer
    for j in range(0,l):
        for i in range(0,num): 
            n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
            h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

        freq = freq_in[j]
        k = 2.*pi*freq/c
        
        # ;===========================================
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2,'complex')
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = const*n_comparr[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = const*n_comparr[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ;===========================================
        # ; define matrix for single layer
        m = np.identity((2),'complex')    # ; net matrix
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

        output[0,j] = freq+0.j #; unit of [Hz]
        output[1,j] = r
        output[2,j] = t

    return output


def oblique_basic_multilayer_r_t( n, d, freq_i, freq_f, freq_int, angle_i, incpol):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; angle of refraction
    angle = np.zeros(num+2)          # ; angle[0]=incident angle
    angle[0] = angle_i
    for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    l = int((freq_f - freq_i)/freq_int + 1.)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; define the effective thickness of each layer
    h = np.zeros(num)
    for i in range(0,num): h[i] = n[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

    # ;-----------------------------------------------------------------------------------
    # ; for loop for various thickness of air gap between each layer
    for j in range(0,l):
        freq = freq_int * j + freq_i
        k = 2.*pi*freq/c
        
        # ;===========================================
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2)
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = const*n[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = const*n[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ;===========================================
        # ; define matrix for single layer
        m = np.identity((2),'complex')    # ; net matrix
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

        output[0,j] = freq+0.j #; unit of [Hz]
        output[1,j] = r
        output[2,j] = t

    return output


def twoplates_multilayer_r_t( n, d, freq_i, freq_f, freq_int, angle_plate, d_sep, incpol):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; angle of refraction
    angle1 = np.zeros(num+2)          # ; angle[0]=incident angle
    angle2 = np.zeros(num+2)          # ; angle[0]=incident angle
    angle1[0] = 0.
    angle2[0] = angle_plate
    for i in range(0,num+1): angle1[i+1] = np.arcsin(np.sin(angle1[i])*n[i]/n[i+1])
    for i in range(0,num+1): angle2[i+1] = np.arcsin(np.sin(angle2[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    l = int((freq_f - freq_i)/freq_int + 1.)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; define the effective thickness of each layer
    h1 = np.zeros(num)
    h2 = np.zeros(num)
    for i in range(0,num): h1[i] = n[i+1]*d[i]*np.cos(angle1[i+1]) # ;effective thickness of 1st layer
    for i in range(0,num): h2[i] = n[i+1]*d[i]*np.cos(angle2[i+1]) # ;effective thickness of 1st layer

    # ;-----------------------------------------------------------------------------------
    # ; for loop for various thickness of air gap between each layer
    for j in range(0,l):
        freq = freq_int * j + freq_i
        k = 2.*pi*freq/c
        
        # ;===========================================
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y1 = np.zeros(num+2)
        for i in range(0,num+2):
            if (incpol == 1): # out of the plane
                Y1[i] = const*n[i]*np.cos(angle1[i])
                cc = 1.
            if (incpol == -1): # in the plane
                Y1[i] = const*n[i]/np.cos(angle1[i])
                cc = np.cos(angle1[num+1])/np.cos(angle1[0])

        Y2 = np.zeros(num+2)
        for i in range(0,num+2):
            if (incpol == 1): # out of the plane
                Y2[i] = const*n[i]*np.cos(angle2[i])
                cc = 1.
            if (incpol == -1): # in the plane
                Y2[i] = const*n[i]/np.cos(angle2[i])
                cc = np.cos(angle2[num+1])/np.cos(angle2[0])

        # ;===========================================
        # ; define matrix for single layer
        m = np.identity((2),'complex')    # ; net matrix
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h1[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h1[i])/Y1[i+1])
            me[0,1] = complex(0., np.sin(k*h1[i])*Y1[i+1])
            me[1,1] = complex(np.cos(k*h1[i]), 0.)
            m = np.dot(m,me)

        delta = pi/c*freq*1.*d_sep \
            *(1.+np.cos(2.*angle_plate)/np.cos(angle_plate)**2 \
                *(1.-np.sin(4.*angle_plate)/np.cos(3.*angle_plate)) )

        if  (incpol == 1): # out of the plane
            me[0,0] = complex(np.cos(delta), 0.)
            me[1,0] = complex(0., np.cos(angle_plate)*np.sin(delta)/const)
            me[0,1] = complex(0., np.sin(delta)*const)
            me[1,1] = complex(np.cos(angle_plate)*np.cos(delta), 0.)
            m = np.dot(m,me)
        if  (incpol == -1): # in the plane
            me[0,0] = complex(np.cos(delta)/np.cos(angle_plate), 0.)
            me[1,0] = complex(0., np.sin(delta)/const/np.cos(angle_plate))
            me[0,1] = complex(0., np.sin(delta)*const/np.cos(angle_plate))
            me[1,1] = complex(np.cos(delta)/np.cos(angle_plate), 0.)
            m = np.dot(m,me)

        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h2[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h2[i])/Y2[i+1])
            me[0,1] = complex(0., np.sin(k*h2[i])*Y2[i+1])
            me[1,1] = complex(np.cos(k*h2[i]), 0.)
            m = np.dot(m,me)

        r = (Y1[0]*m[0,0]*cc+Y1[0]*Y2[num+1]*m[1,0]-m[0,1]*cc-Y2[num+1]*m[1,1]) \
            / (Y1[0]*m[0,0]*cc+Y1[0]*Y2[num+1]*m[1,0]+m[0,1]*cc+Y2[num+1]*m[1,1])
        t = 2.*Y1[0] / (Y1[0]*m[0,0]*cc+Y1[0]*Y2[num+1]*m[1,0]+m[0,1]*cc+Y2[num+1]*m[1,1])

        output[0,j] = freq+0.j #; unit of [Hz]
        output[1,j] = r
        output[2,j] = t

    return output

def oblique_basic_multilayer_r_t_incloss( n, losstan, d, freq_i, freq_f, freq_int, angle_i, incpol):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; angle of refraction
    angle = np.zeros(num+2)          # ; angle[0]=incident angle
    angle[0] = angle_i
    for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    l = int((freq_f - freq_i)/freq_int + 1.)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; define the effective thickness of each layer
    h = np.zeros(num,'complex')
    n_comparr = np.zeros(len(n),'complex')
    n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
    n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

    # ;-----------------------------------------------------------------------------------
    # ; for loop for various thickness of air gap between each layer
    for j in range(0,l):
        for i in range(0,num): 
            n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
            h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

        freq = freq_int * j + freq_i
        k = 2.*pi*freq/c
        
        # ;===========================================
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2,'complex')
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = const*n_comparr[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = const*n_comparr[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ;===========================================
        # ; define matrix for single layer
        m = np.identity((2),'complex')    # ; net matrix
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

        output[0,j] = freq+0.j #; unit of [Hz]
        output[1,j] = r
        output[2,j] = t

    return output


def basic_multilayer_r_t_1plate1freq( n, d, freq):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    #    l = int((freq_f - freq_i)/freq_int + 1.)
    output = np.zeros(3,'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; define the effective thickness of each layer
    h = np.zeros(num)
    for i in range(0,num): 
        h[i] = n[i+1]*d[i]  #*np.cos(angle[i+1]) # ;effective thickness of 1st layer

    k = 2.*pi*freq/c
    # ;===========================================
    # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
    Y = np.zeros(num+2)
    for i in range(0,num+2):
        Y[i] = const*n[i] #*np.cos(angle[i])
    cc = 1.
    
    # ;===========================================
    # ; define matrix for single layer
    m = np.identity((2),'complex')    # ; net matrix
    me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
    for i in range(0,num):
        me[0,0] = complex(np.cos(k*h[i]), 0.)
        me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
        me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
        me[1,1] = complex(np.cos(k*h[i]), 0.)
        m = np.dot(m,me)

    r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
    t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

    output[0] = freq+0.j #; unit of [Hz]
    output[1] = r
    output[2] = t
    return output


def oblique_basic_multilayer_r_t_z( n, d, z_i, z_f, z_int, freq_in, angle_i, incpol):

    num=len(d) #; the number of layer not including two ends
    const = np.sqrt((8.85e-12)/(4.*pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

    # ;-----------------------------------------------------------------------------------
    # ; angle of refraction
    angle = np.zeros(num+2)          # ; angle[0]=incident angle
    angle[0] = angle_i
    for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

    # ;-----------------------------------------------------------------------------------
    # ; define the frequency span
    l = int((z_f - z_i)/z_int + 1.)
    output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)
    
    # ;-----------------------------------------------------------------------------------
    # ; for loop for various thickness of air gap between each layer
    for j in range(0,l):
        freq = freq_in
        z = z_int * j + z_i
        k = 2.*pi*freq/c

        d[0] = z

        # ;-----------------------------------------------------------------------------------
        # ; define the effective thickness of each layer
        h = np.zeros(num)
        for i in range(0,num): h[i] = n[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer
        
        # ;===========================================
        # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
        Y = np.zeros(num+2)
        for i in range(0,num+2):
            if (incpol == 1):
                Y[i] = const*n[i]*np.cos(angle[i])
                cc = 1.
            if (incpol == -1):
                Y[i] = const*n[i]/np.cos(angle[i])
                cc = np.cos(angle[num+1])/np.cos(angle[0])

        # ;===========================================
        # ; define matrix for single layer
        m = np.identity((2),'complex')    # ; net matrix
        me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
        for i in range(0,num):
            me[0,0] = complex(np.cos(k*h[i]), 0.)
            me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
            me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
            me[1,1] = complex(np.cos(k*h[i]), 0.)
            m = np.dot(m,me)

        r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
        t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

#        output[0,j] = freq+0.j #; unit of [Hz]
        output[0,j] = z+0.j #; unit of [Hz]
        output[1,j] = r
        output[2,j] = t

    return output

def transmission_w_abs(n,tandel,d,nu,theta_i,incpol):
    k_0 = 2.*pi/c * nu
    T = 4.*n/(1.+n)**2
    R = (1.-n)**2/(1.+n)**2
    if incpol == 1:
        delta = 2.*k_0*n*d/np.cos(theta_i)
        mu = np.exp(-k_0*n*d*tandel/np.cos(theta_i))
    if incpol == -1:
        delta = 2.*k_0*n*d*np.cos(theta_i)
        mu = np.exp(-k_0*n*d*tandel*np.cos(theta_i))
    a_t = T*mu*np.exp(-1.j*delta)/(1.-R*mu**2*np.exp(-1.j*delta))
    return np.abs(a_t)**2

def band_integration(freq_c, band_frachalf, RT):
    band = np.where((np.abs(RT[0]) > freq_c*(1.-band_frachalf)) & (np.abs(RT[0]) < freq_c*(1.+band_frachalf)))
    R = np.mean(np.abs(RT[1,band])**2.)
    T = np.mean(np.abs(RT[2,band])**2.)
    return R, T


def SWG_frac2index(freq, n1, n2, f, pitch):
    '''
    physical dimension of pitch
    width of n2 (nominally take as a material)cal
    '''
    r = pitch/c*freq
    ep1 = n1**2.*ep0
    ep2 = n2**2.*ep0

    ep01 = (1.-f)*ep1+f*ep2
    ep02 = 1./((1.-f)/ep1+f/ep2)

    ep21 = ep01*(1.+pi**2/3.*r**2.*f**2*(1.-f)**2*(ep2-ep1)**2./(ep0*ep01))
    ep22 = ep02*(1.+pi**2/3.*r**2.*f**2*(1.-f)**2*(ep2-ep1)**2.*ep01/ep0*(ep02/ep2/ep1)**2.)

    ep_that = (1.-f)*ep1+f*ep22
    ep_lhat = 1./((1.-f)/ep1+f/ep21)

    n_ave = (1.-f**2.)*n1+f**2*n2
    n_that=np.sqrt(ep_that/ep0)
    n_lhat=np.sqrt(ep_lhat/ep0)

    n = .2*(n_ave+2.*n_that+2.*n_lhat)
    w = f*pitch

    return n, w

def SWG_index2frac(index_arr, freq, n1, n2, f, pitch, option_template=False):
    n, w = SWG_frac2index(freq, n1, n2, f, pitch)
    func = interp1d(n, w, kind='cubic')
    if option_template == False: return func(index_arr)
    if option_template == True: return func(index_arr), n, w

def SWG_frac2index_2polfrac(freq, n1, n2, f1, f2, pitch):
    '''
    physical dimension of pitch
    width of n2 (nominally take as a material)
    '''
    r = pitch/c*freq
    ep1 = n1**2.*ep0
    ep2 = n2**2.*ep0

    ep01 = (1.-f1)*ep1+f1*ep2
    ep02 = 1./((1.-f2)/ep1+f2/ep2)

    ep21 = ep01*(1.+pi**2/3.*r**2.*f1**2*(1.-f1)**2*(ep2-ep1)**2./(ep0*ep01))
    ep22 = ep02*(1.+pi**2/3.*r**2.*f2**2*(1.-f2)**2*(ep2-ep1)**2.*ep01/ep0*(ep02/ep2/ep1)**2.)

    ep_lhat = 1./((1.-f1)/ep1+f1/ep21)
    ep_that = (1.-f2)*ep1+f2*ep22

    n_ave = (1.-f1**2.)*n1+f2**2*n2
    n_that=np.sqrt(ep_that/ep0)
    n_lhat=np.sqrt(ep_lhat/ep0)

    n = .2*(n_ave+2.*n_that+2.*n_lhat)
    w1 = f1*pitch
    w2 = f2*pitch

    return n_lhat, n_that, w1, w2

def SWG_frac2index_2pol(freq, n1, n2, f, pitch):
    '''
    physical dimension of pitch
    width of n2 (nominally take as a material)
    '''
    r = pitch/c*freq
    ep1 = n1**2.*ep0
    ep2 = n2**2.*ep0

    ep01 = (1.-f)*ep1+f*ep2
    ep02 = 1./((1.-f)/ep1+f/ep2)

    ep21 = ep01*(1.+pi**2/3.*r**2.*f**2*(1.-f)**2*(ep2-ep1)**2./(ep0*ep01))
    ep22 = ep02*(1.+pi**2/3.*r**2.*f**2*(1.-f)**2*(ep2-ep1)**2.*ep01/ep0*(ep02/ep2/ep1)**2.)

    n1 = np.sqrt(ep21/ep0)
    n2 = np.sqrt(ep22/ep0)

    w = f*pitch

    return n1, n2, w

def RT_angulardep( n, d, freq_i, freq_f, freq_int, figure=False, freq_c1=False, freq_c2=False, freq_c3=False, band_frachalf=False):
    num_angle = 30
    angle_i = np.arange(num_angle)/float(num_angle)*50. / 180.*pi

    for i in range(0,num_angle):
        RT1 = oblique_basic_multilayer_r_t( n, d, freq_i, freq_f, freq_int, angle_i[i], -1)
        RT2 = oblique_basic_multilayer_r_t( n, d, freq_i, freq_f, freq_int, angle_i[i], 1)
#        py.plot(np.abs(RT1[1]))
#        py.show()
        if freq_c1 != False:
            R11, T11 = np.zeros(num_angle), np.zeros(num_angle)
            R12, T12 = np.zeros(num_angle), np.zeros(num_angle)
            R11[i], T11[i] = band_integration(freq_c1, band_frachalf, RT1)
            R12[i], T12[i] = band_integration(freq_c1, band_frachalf, RT2)
        if freq_c2 != False:
            R21, T21 = np.zeros(num_angle), np.zeros(num_angle)
            R22, T22 = np.zeros(num_angle), np.zeros(num_angle)
            R21[i], T21[i] = band_integration(freq_c2, band_frachalf, RT1)
            R22[i], T22[i] = band_integration(freq_c2, band_frachalf, RT2)
        if freq_c3 != False:
            R31, T31 = np.zeros(num_angle), np.zeros(num_angle)
            R32, T32 = np.zeros(num_angle), np.zeros(num_angle)
            R31[i], T31[i] = band_integration(freq_c3, band_frachalf, RT1)
            R32[i], T32[i] = band_integration(freq_c3, band_frachalf, RT2)

    print(R11, R12)
    if figure==False: figure=1
    py.figure(figure)
    py.subplot(311)
    if freq_c1 != False:
        py.plot(angle_i/pi*180,R11)
        py.plot(angle_i/pi*180,T11)
        py.plot(angle_i/pi*180,R12)
        py.plot(angle_i/pi*180,T12)
        py.semilogy()
    py.subplot(312)
    if freq_c2 != False:
        py.plot(angle_i/pi*180,R21)
        py.plot(angle_i/pi*180,T21)
        py.plot(angle_i/pi*180,R22)
        py.plot(angle_i/pi*180,T22)
        py.semilogy()
    py.subplot(313)
    if freq_c3 != False:
        py.plot(angle_i/pi*180,R31)
        py.plot(angle_i/pi*180,T31)
        py.plot(angle_i/pi*180,R32)
        py.plot(angle_i/pi*180,T32)
        py.semilogy()
    

#################################################################################################################################
######### (A)HWP related codes
def Mueller_rot(theta):
    nb = 4
    m = np.zeros([nb,nb])
    m[0,0]=1.
    m[1,1]= np.cos(2.*theta)
    m[1,2]=-np.sin(2.*theta)
    m[2,1]= np.sin(2.*theta)
    m[2,2]= np.cos(2.*theta)
#    m[0,1]=m[0,2]=m[0,3]=m[1,0]=m[1,3]=m[2,0]=m[2,3]=m[3,0]=m[3,1]=m[3,2]=0.
    m[3,3]=1.
    return m

def Mueller_retard(delta):
    nb = 4
    m = np.zeros([nb,nb])
    m[0,0]=1.
    m[1,1]=1.
    m[2,2]= np.cos(delta)
    m[2,3]=-np.sin(delta)
    m[3,2]= np.sin(delta)
    m[3,3]= np.cos(delta)
#    m[0,1]=m[0,2]=m[0,3]=m[1,0]=m[1,2]=m[1,3]=m[2,0]=m[2,1]=m[3,0]=m[3,1]=0.
    return m

def Mueller_retard_trans(tx, ty): # , delta):
    nb = 4
    m = np.zeros([nb,nb],'complex')

    m[0,0]=0.5*(tx*np.conjugate(tx)+ty*np.conjugate(ty))
    m[0,1]=0.5*(tx*np.conjugate(tx)-ty*np.conjugate(ty))
    m[0,2]=0.
    m[0,3]=0.

    m[1,0]=0.5*(tx*np.conjugate(tx)-ty*np.conjugate(ty))
    m[1,1]=0.5*(tx*np.conjugate(tx)+ty*np.conjugate(ty))
    m[1,2]=0.
    m[1,3]=0.

    m[2,0]=0.
    m[2,1]=0.
    m[2,2]=0.5*(tx*np.conjugate(ty)+ty*np.conjugate(tx))
    m[2,3]=0.5*(tx*np.conjugate(ty)-ty*np.conjugate(tx))/1.j

    m[3,0]=0.
    m[3,1]=0.
    m[3,2]=0.5j*(tx*np.conjugate(ty)-ty*np.conjugate(tx))
    m[3,3]=0.5*(tx*np.conjugate(ty)+ty*np.conjugate(tx))
    return m


def Mueller_xgrid():
    nb = 4
    m = np.zeros([nb,nb])
    m[0,0]=m[0,1]=m[1,0]=m[1,1]=1.
#    m[0,2]=m[0,3]=m[1,2]=m[1,3]=0.
#    m[2,0]=m[2,1]=m[2,2]=m[2,3]=0.
#    m[3,0]=m[3,1]=m[3,2]=m[3,3]=0.
    return 0.5*m

def Mueller_unit():
    nb = 4
    m = np.zeros([nb,nb])
    m[0,0]=m[1,1]=m[2,2]=m[3,3]=1.
    return m

def calSout_IVAout(hwp_angle,theta_offset,delta,Sin):
    nb_angle = hwp_angle.shape[0]
    nb_ahwp = theta_offset.shape[0]
    grid = Mueller_xgrid()
    num_freq = len(delta)
    Sout = np.zeros((4,nb_angle))
    for i in range(0,nb_angle):
        m = Mueller_unit()
        for j in range(0,nb_ahwp):
            rot = Mueller_rot(hwp_angle[i]+theta_offset[j])
            hwp = Mueller_retard(delta[j])
            rot_inv = Mueller_rot(-(hwp_angle[i]+theta_offset[j]))
            m = np.dot(rot,m)
            m = np.dot(hwp,m)
            m = np.dot(rot_inv,m)
        m = np.dot(grid,m)
        # Sout_tmp = np.dot(m,Sin)
        Sout[:,i] = np.dot(m,Sin)
        # Sout_tmp[0]
    return Sout

def calHWP_IVAout(hwp_angle,theta_offset,delta,Sin):
    nb_angle = hwp_angle.shape[0]
    nb_ahwp = theta_offset.shape[0]
    grid = Mueller_xgrid()
    Sout = np.zeros(nb_angle)
    for i in range(0,nb_angle):
        m = Mueller_unit()
        for j in range(0,nb_ahwp):
            rot = Mueller_rot(hwp_angle[i]+theta_offset[j])
            hwp = Mueller_retard(delta[j])
            rot_inv = Mueller_rot(-(hwp_angle[i]+theta_offset[j]))
            m = np.dot(rot,m)
            m = np.dot(hwp,m)
            m = np.dot(rot_inv,m)
        m = np.dot(grid,m)
        Sout_tmp = np.dot(m,Sin)
        Sout[i] = Sout_tmp[0]
    return Sout

def calHWP_IVAout_phasecancel(hwp_angle,theta_offset,delta,Sin):
    nb_angle = hwp_angle.shape[0]
    nb_ahwp = (theta_offset.shape[0])/2
    grid = Mueller_xgrid()
    Sout = np.zeros(nb_angle)
    for i in range(0,nb_angle):
        m = Mueller_unit()
        for j in range(0,nb_ahwp):
            rot = Mueller_rot(hwp_angle[i]+theta_offset[j])
            hwp = Mueller_retard(delta[j])
            rot_inv = Mueller_rot(-(hwp_angle[i]+theta_offset[j]))
            m = np.dot(rot,m)
            m = np.dot(hwp,m)
            m = np.dot(rot_inv,m)
        for j in range(0,nb_ahwp):
            rot = Mueller_rot(theta_offset[j+nb_ahwp])
            hwp = Mueller_retard(delta[j+nb_ahwp])
            rot_inv = Mueller_rot(-theta_offset[j+nb_ahwp])
#            rot = Mueller_rot(theta_offset[j])
#            hwp = Mueller_retard(delta[j])
#            rot_inv = Mueller_rot(-theta_offset[j])
            m = np.dot(rot,m)
            m = np.dot(hwp,m)
            m = np.dot(rot_inv,m)
        m = np.dot(grid,m)
        Sout_tmp = np.dot(m,Sin)
        Sout[i] = Sout_tmp[0]
    return Sout

def calHWP_IVAout_trans(hwp_angle,theta_offset,tx,ty,Sin):
    nb_angle = hwp_angle.shape[0]
    nb_ahwp = theta_offset.shape[0]
    grid = Mueller_xgrid()
    Sout = np.zeros(nb_angle)
    for i in range(0,nb_angle):
        m = Mueller_unit()
        for j in range(0,nb_ahwp):
            rot = Mueller_rot(hwp_angle[i]+theta_offset[j])
            hwp = Mueller_retard_trans(tx[j],ty[j])
            rot_inv = Mueller_rot(-(hwp_angle[i]+theta_offset[j]))
            m = np.dot(rot,m)
            m = np.dot(hwp,m)
            m = np.dot(rot_inv,m)
        m = np.dot(grid,m)
        Sout_tmp = np.dot(m,Sin)
        Sout[i] = np.abs(Sout_tmp[0])
    return Sout

def IVA_Mean_RMS_inband(freq_c, band_frachalf, freq, IVA):
    nb_freq, nb_angle = IVA.shape
    y_mean, y_std = np.zeros(nb_angle), np.zeros(nb_angle)
    band = np.where((freq > freq_c*(1.-band_frachalf)) & (freq < freq_c*(1.+band_frachalf)))
    band = band[0]
    for i in range(0,nb_angle):
        y_mean[i] = np.mean(IVA[band,i])
        y_std[i] = np.std(IVA[band,i])
    print('[IVA_Mean_RMS_inband] band_low, band_c, band_high', freq_c*(1.-band_frachalf)*1e-9, freq_c*1e-9, freq_c*(1.+band_frachalf)*1e-9)
    return y_mean, y_std

def Mean_RMS_inband(freq_c, band_frachalf, freq, y, wraparound=False):
    band = np.where((freq > freq_c*(1.-band_frachalf)) & (freq < freq_c*(1.+band_frachalf)))
    band = band[0]
    y_mean = np.mean(y[band])
    y_std = np.std(y[band])
    return y_mean, y_std

def IVA2PolEffPhase(IVA,rho):
    IVA_max = max(IVA)
    IVA_min = min(IVA)
    Poleff = (IVA_max-IVA_min)/(IVA_max+IVA_min)
    ind = np.where(IVA == max(IVA))
    rho_max = np.mean(rho[ind])
    return Poleff, rho_max

def IVA2PolEffPhase_Ip(IVA,rho):
    IVA_max = max(IVA)
    IVA_min = min(IVA)
    Poleff = (IVA_max-IVA_min)
    ind = np.where(IVA == max(IVA))
    rho_max = np.mean(rho[ind])
    return Poleff, rho_max

def IVA_model(par,x):
    return 0.5*(par[0]+par[1]*np.cos(4.*x+4.*par[2]) )

def IVA_model2(par,x):
    return 0.5*(par[0]+par[1]*np.cos(2.*x+2.*par[2])+par[3]*np.cos(4.*x+4.*par[4]) )

def IVA_model_CF(t,p1,p2,p3,p4):
    return 0.5*(p1+p2*np.cos(4.*p3*t+2.*p4))

def IVA_model2_CF(t,p1,p2,p3,p4,p5):
    return 0.5*(p1+p2*np.cos(4.*t+4.*p3)+p4*np.cos(2.*t+2.*p5) )

def IVT_model_CF(t,p1,p2,p3,p4):
    return 0.5*(p1+p2*np.cos(4.*p3*t+4.*p4) )

def IVT_model2_CF(t,p1,p2,p3,p4,p5,p6):
    return 0.5*(p1+p2*np.cos(4.*p3*t+4.*p4)+p5*np.cos(2.*p3*t+2.*p6) )

def Chi_IVA_model(par,x,data):
    return np.sum( (IVA_model(par,x) - data)**2 )

def Chi_IVA_model2(par,x,data):
    return np.sum( (IVA_model2(par,x) - data)**2 )

#def IVA2PolEffPhase_fit(IVA,rho):
#    xopt = fmin(Chi_IVA_model, par, args=(x,data), xtol=1e-1, maxiter=10)
#    return xopt

def IVA2PolEffPhase_fit(x,data):
    par = np.array([0.5,1.,0.])
    xopt = fmin(Chi_IVA_model, par, args=(x,data), xtol=1e-1, maxiter=1000)
    poleff = xopt[1]/xopt[0]
    phase = par[2]
    return poleff, phase

def IVAtrans2PolEffPhase_fit(x,data):
    par = np.array([0.2,0.2,0.,.2,0.])
#    print x*180/pi
#    print Chi_IVA_model2(par,x,data)
#    print IVA_model2(par,x)
#    print data
#    sys.exit()
    xopt = fmin(Chi_IVA_model2, par, args=(x,data), xtol=1e-1, maxiter=1000)
    poleff = xopt[3]/xopt[0]
    phase = xopt[4]
    if xopt[3] < 0:
        poleff = -xopt[3]/xopt[0]
        phase = phase+pi
    return poleff, phase

def IVAtrans2PolEffPhase_CFfit(x,data):
#    par = np.array([0.2,0.2,0.,.2,0.])
    par = np.array( [np.mean(data), 0.5*(np.max(data)-np.min(data)), 0., 1., 0.] ) 
    popt, pcov = curve_fit(IVA_model2_CF, x, data, p0=par)
    poleff = popt[1]/popt[0]
    phase = popt[2]
    if popt[1] < 0:
        poleff = -popt[1]/popt[0]
        phase = phase+pi
    return poleff, phase



def genplot_Poleff(par, n_o, n_e, freq_i, freq_f, freq_int, Sin,
                   freq_c1=False, freq_c2=False, freq_c3=False, 
                   band_frachalf=False, plot=False, option_dump='', option_IVAout=False):
    num = len(par)
    theta_offset = par[0:num/2]
    thick = par[num/2:num]
    num_angle = 360*2*8
#    num_angle = 90
    hwp_angle = np.arange(0,num_angle)/float(num_angle)*pi/2.
    num_plate = len(theta_offset)
    i_angle = range(0,num_angle)
    num_freq = int((freq_f - freq_i)/freq_int + 1.)
    i_freq = range(0,num_freq)
    freq = np.arange(0,num_freq)/float(num_freq)*(freq_f-freq_i)+freq_i
    IVA = np.zeros((num_freq,num_angle)) 

    Poleff, phase = np.zeros(num_freq), np.zeros(num_freq)
    i_ahwp = range(0,num_plate)

    for j in range(0,num_freq):
        delta_arr = []
        for i in range(0,num_plate):
            delta = 2.*pi*(n_e-n_o)*thick[i]/c*freq[j]
            delta_arr = np.hstack((delta_arr,delta))
        IVA[j,:] = calHWP_IVAout(hwp_angle,theta_offset,delta_arr,Sin)
        Poleff[j], phase[j] = IVA2PolEffPhase_Ip(IVA[j,:],hwp_angle)

    if option_dump!='':
        np.savez(option_dump,freq,Poleff,phase)
#        np.savez(option_dump+'/PoleffPhaseFreq_oneset',freq,Poleff,phase)
#    np.savez('/home/tmatsumu/work/develop/PBII/PBII_hardware/Optics/AHWP_design/Poleff/PBII_prototypeDesign/freq',freq, Poleff)
#    sys.exit()
    if(freq_c1 != False):
        IVA_mean1, IVA_std1 = IVA_Mean_RMS_inband(freq_c1, band_frachalf, freq, IVA)
        print('Poleff and phase at freq', freq_c1*1e-9, IVA2PolEffPhase_Ip(IVA_mean1,hwp_angle), IVA2PolEffPhase(IVA_mean1,hwp_angle))
    if(freq_c2 != False):
        IVA_mean2, IVA_std2 = IVA_Mean_RMS_inband(freq_c2, band_frachalf, freq, IVA)
        print('Poleff and phase at freq', freq_c2*1e-9, IVA2PolEffPhase_Ip(IVA_mean2,hwp_angle), IVA2PolEffPhase(IVA_mean2,hwp_angle))
    if(freq_c3 != False):
        IVA_mean3, IVA_std3 = IVA_Mean_RMS_inband(freq_c3, band_frachalf, freq, IVA)
        print('Poleff and phase at freq', freq_c3*1e-9, IVA2PolEffPhase_Ip(IVA_mean3,hwp_angle), IVA2PolEffPhase(IVA_mean3,hwp_angle))

    if plot==True:
        py.subplot(221)
        py.plot(freq*1e-9,Poleff)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
        py.ylim(np.min(Poleff)*0.95,np.max(Poleff)*1.05)
        py.ylabel('Mod eff')

        py.subplot(223)
        py.plot(freq*1e-9,phase*180./pi)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-y')
        py.ylim(np.min(phase*180./pi)*0.95,np.max(phase*180./pi)*1.05)
        py.ylabel('Phase [degs]')

        py.subplot(222)
        for j in range(0,num_freq):
            py.plot(hwp_angle,IVA[j,:])
        py.ylabel('IVA')

        py.subplot(224)
        if(freq_c1 != False):
            py.plot(hwp_angle,IVA_mean1,label=str(freq_c1*1e-9)+'GHz')
        if(freq_c2 != False):
            py.plot(hwp_angle,IVA_mean2,label=str(freq_c2*1e-9)+'GHz')
        if(freq_c3 != False):
            py.plot(hwp_angle,IVA_mean3,label=str(freq_c3*1e-9)+'GHz')
        if((freq_c1 != False) or (freq_c2 != False) or (freq_c3 != False)):
            py.legend(loc='best')

        py.show()

    if option_IVAout==False: return freq, Poleff, phase
    if option_IVAout==True: return freq, Poleff, phase, IVA, hwp_angle

def genplot_Poleff2(par, n_o, n_e, freq_i, freq_f, freq_int, Sin,
                   freq_c1=False, freq_c2=False, freq_c3=False, 
                   band_frachalf1=False, band_frachalf2=False, band_frachalf3=False, 
                   plot=False, option_dump='', option_IVAout=False):
    '''
    output includes the multiple IVAs 
    '''
    num = len(par)
    theta_offset = par[0:num/2]
    thick = par[num/2:num]
    num_angle = 360*2*8
#    num_angle = 90
    hwp_angle = np.arange(0,num_angle)/float(num_angle)*pi/2.
    num_plate = len(theta_offset)
    i_angle = range(0,num_angle)
    num_freq = int((freq_f - freq_i)/freq_int + 1.)
    i_freq = range(0,num_freq)
    freq = np.arange(0,num_freq)/float(num_freq)*(freq_f-freq_i)+freq_i
    IVA = np.zeros((num_freq,num_angle)) 

    Poleff, phase = np.zeros(num_freq), np.zeros(num_freq)
    i_ahwp = range(0,num_plate)

    for j in range(0,num_freq):
        delta_arr = []
        for i in range(0,num_plate):
            delta = 2.*pi*(n_e-n_o)*thick[i]/c*freq[j]
            delta_arr = np.hstack((delta_arr,delta))
        IVA[j,:] = calHWP_IVAout(hwp_angle,theta_offset,delta_arr,Sin)
        Poleff[j], phase[j] = IVA2PolEffPhase_Ip(IVA[j,:],hwp_angle)

    if option_dump!='':
        np.savez(option_dump,freq,Poleff,phase)
#        np.savez(option_dump+'/PoleffPhaseFreq_oneset',freq,Poleff,phase)
#    np.savez('/home/tmatsumu/work/develop/PBII/PBII_hardware/Optics/AHWP_design/Poleff/PBII_prototypeDesign/freq',freq, Poleff)
#    sys.exit()
    if(freq_c1 != False):
        IVA_mean1, IVA_std1 = IVA_Mean_RMS_inband(freq_c1, band_frachalf1, freq, IVA)
        print('Poleff and phase at freq', freq_c1*1e-9, IVA2PolEffPhase_Ip(IVA_mean1,hwp_angle), IVA2PolEffPhase(IVA_mean1,hwp_angle))
    if(freq_c2 != False):
        IVA_mean2, IVA_std2 = IVA_Mean_RMS_inband(freq_c2, band_frachalf2, freq, IVA)
        print('Poleff and phase at freq', freq_c2*1e-9, IVA2PolEffPhase_Ip(IVA_mean2,hwp_angle), IVA2PolEffPhase(IVA_mean2,hwp_angle))
    if(freq_c3 != False):
        IVA_mean3, IVA_std3 = IVA_Mean_RMS_inband(freq_c3, band_frachalf3, freq, IVA)
        print('Poleff and phase at freq', freq_c3*1e-9, IVA2PolEffPhase_Ip(IVA_mean3,hwp_angle), IVA2PolEffPhase(IVA_mean3,hwp_angle))

    if plot==True:
        py.subplot(221)
        py.plot(freq*1e-9,Poleff)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf1)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf1)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf2)*1e-9,np.array([0.,1.2]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf2)*1e-9,np.array([0.,1.2]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf3)*1e-9,np.array([0.,1.2]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf3)*1e-9,np.array([0.,1.2]), '-y')
        py.ylim(np.min(Poleff)*0.95,np.max(Poleff)*1.05)
        py.ylabel('Mod eff')

        py.subplot(223)
        py.plot(freq*1e-9,phase*180./pi)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf1)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf1)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf2)*1e-9,np.array([0.,360.]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf2)*1e-9,np.array([0.,360.]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf3)*1e-9,np.array([0.,360.]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf3)*1e-9,np.array([0.,360.]), '-y')
        py.ylim(np.min(phase*180./pi)*0.95,np.max(phase*180./pi)*1.05)
        py.ylabel('Phase [degs]')

        py.subplot(222)
        for j in range(0,num_freq):
            py.plot(hwp_angle,IVA[j,:])
        py.ylabel('IVA')

        py.subplot(224)
        if(freq_c1 != False):
            py.plot(hwp_angle,IVA_mean1,label=str(freq_c1*1e-9)+'GHz')
        if(freq_c2 != False):
            py.plot(hwp_angle,IVA_mean2,label=str(freq_c2*1e-9)+'GHz')
        if(freq_c3 != False):
            py.plot(hwp_angle,IVA_mean3,label=str(freq_c3*1e-9)+'GHz')
        if((freq_c1 != False) or (freq_c2 != False) or (freq_c3 != False)):
            py.legend(loc='best')

        py.show()

    if option_IVAout==False: return freq, Poleff, phase
    if option_IVAout==True: return freq, Poleff, phase, IVA_mean1, IVA_mean2, IVA_mean3, hwp_angle

def genplot_Poleff_phasecancel_twosets(par, n_o, n_e, freq_i, freq_f, freq_int, Sin,
                                       freq_c1=False, freq_c2=False, freq_c3=False, 
                                       band_frachalf=False, plot=False, option_dump=''):
    num = len(par)
    theta_offset = par[0:num/2]
    thick = par[num/2:num]
    num_angle = 360*2
    hwp_angle = np.arange(0,num_angle)/float(num_angle)*pi/2.
    num_plate = len(theta_offset)
    i_angle = range(0,num_angle)
    num_freq = int((freq_f - freq_i)/freq_int + 1.)
    i_freq = range(0,num_freq)
    freq = np.arange(0,num_freq)/float(num_freq)*(freq_f-freq_i)+freq_i
    IVA = np.zeros((num_freq,num_angle)) 

    Poleff, phase = np.zeros(num_freq), np.zeros(num_freq)
    i_ahwp = range(0,num_plate)

    for j in range(0,num_freq):
        delta_arr = []
        for i in range(0,num_plate):
            delta = 2.*pi*(n_e-n_o)*thick[i]/c*freq[j]
            delta_arr = np.hstack((delta_arr,delta))
        IVA[j,:] = calHWP_IVAout_phasecancel(hwp_angle,theta_offset,delta_arr,Sin)
#        IVA[j,:] = calHWP_IVAout(hwp_angle,theta_offset,delta_arr,Sin)
        Poleff[j], phase[j] = IVA2PolEffPhase_Ip(IVA[j,:],hwp_angle)

    if option_dump!='':
#        np.savez(option_dump+'/PoleffPhaseFreq_twoset',freq,Poleff,phase)
        np.savez(option_dump,freq,Poleff,phase)

    if(freq_c1 != False):
        IVA_mean1, IVA_std1 = IVA_Mean_RMS_inband(freq_c1, band_frachalf, freq, IVA)
        Poleff_tmp, phase_tmp = IVA2PolEffPhase_Ip(IVA_mean1,hwp_angle)
        print('Poleff and phase at freq', freq_c1*1e-9, IVA2PolEffPhase(IVA_mean1,hwp_angle))
        if option_dump!='':
            np.savez(option_dump+'_MeanIVA_freq_c1_twoset',hwp_angle,IVA_mean1,IVA_std1)
            np.savez(option_dump+'_MeanPoleffPhase_freq_c1_twoset',Poleff_tmp,phase_tmp)
    if(freq_c2 != False):
        IVA_mean2, IVA_std2 = IVA_Mean_RMS_inband(freq_c2, band_frachalf, freq, IVA)
        Poleff_tmp, phase_tmp = IVA2PolEffPhase_Ip(IVA_mean2,hwp_angle)
        print('Poleff and phase at freq', freq_c2*1e-9, IVA2PolEffPhase(IVA_mean2,hwp_angle))
        if option_dump!='':
            np.savez(option_dump+'_MeanIVA_freq_c2_twoset',hwp_angle,IVA_mean2,IVA_std2)
            np.savez(option_dump+'_MeanPoleffPhase_freq_c2_twoset',Poleff_tmp,phase_tmp)
    if(freq_c3 != False):
        IVA_mean3, IVA_std3 = IVA_Mean_RMS_inband(freq_c3, band_frachalf, freq, IVA)
        Poleff_tmp, phase_tmp = IVA2PolEffPhase_Ip(IVA_mean3,hwp_angle)
        print('Poleff and phase at freq', freq_c3*1e-9, IVA2PolEffPhase(IVA_mean3,hwp_angle))
        print('')
        if option_dump!='':
            np.savez(option_dump+'_MeanIVA_freq_c3_twoset',hwp_angle,IVA_mean3,IVA_std3)
            np.savez(option_dump+'_MeanPoleffPhase_freq_c3_twoset',Poleff_tmp,phase_tmp)

    if(plot == True):
        py.subplot(221)
        py.plot(freq*1e-9,Poleff)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
        py.ylim(np.min(Poleff)*0.95,np.max(Poleff)*1.05)
        py.ylabel('Mod eff')
        py.xlabel('Frequency [GHz]')

        py.subplot(223)
        py.plot(freq*1e-9,phase*180./pi)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-y')
        py.ylim(np.min(phase*180./pi)*0.95,np.max(phase*180./pi)*1.05)
        py.ylabel('Phase [degs]')
        py.xlabel('Frequency [GHz]')

        py.subplot(222)
        for j in range(0,num_freq):
            py.plot(hwp_angle/pi*180.,IVA[j,:])
        py.ylabel('IVA')
        py.xlabel('HWP angle [degs]')

        py.subplot(224)
        if(freq_c1 != False):
            py.plot(hwp_angle,IVA_mean1,label=str(freq_c1*1e-9)+'GHz')
        if(freq_c2 != False):
            py.plot(hwp_angle,IVA_mean2,label=str(freq_c2*1e-9)+'GHz')
        if(freq_c3 != False):
            py.plot(hwp_angle,IVA_mean3,label=str(freq_c3*1e-9)+'GHz')
        if((freq_c1 != False) or (freq_c2 != False) or (freq_c3 != False)):
            py.legend(loc='best')
#    py.savefig(option_dump+'IVA.png')
#    if plot==True:
#        py.show()

    return freq, Poleff, phase


def genplot_Poleff_trans(par, narr_o, narr_e, freq_i, freq_f, freq_int, Sin, angle_i, incpol,
                         freq_c1=False, freq_c2=False, freq_c3=False, band_frachalf=False, plot=False, option_dump=''):
    print('[genplot_Poleff_trans]')
    num = len(par)
    theta_offset = par[0:int(num/2)]
    thick = par[int(num/2):num]
    num_angle = 360*2
    hwp_angle = np.arange(0,num_angle)/float(num_angle)*pi
    num_plate = len(theta_offset)
    i_angle = range(0,num_angle)
    num_freq = int((freq_f - freq_i)/freq_int + 1.)
    i_freq = range(0,num_freq)
    freq = np.arange(0,num_freq)/float(num_freq)*(freq_f-freq_i)+freq_i
    IVA = np.zeros((num_freq,num_angle)) 

    Poleff, phase = np.zeros(num_freq), np.zeros(num_freq)
    i_ahwp = range(0,num_plate)

    txx=[]; tyy=[]; fxx = []
    for j in range(0,num_freq):
        delta_arr = []
        tx_arr = []
        ty_arr = []
        for i in range(0,num_plate):
            output_o = basic_multilayer_r_t_1plate1freq( narr_o[i:i+3], [thick[i]], freq[j])
            output_e = basic_multilayer_r_t_1plate1freq( narr_e[i:i+3], [thick[i]], freq[j])
            tx = output_o[2]
            ty = output_e[2]
            tx_arr = np.hstack((tx_arr,tx))
            ty_arr = np.hstack((ty_arr,ty))
#            fxx.append(np.abs(output_o[0]))
#            txx.append(np.abs(output_o[2])**2)
#            tyy.append(np.abs(output_e[2])**2)
        IVA[j,:] = calHWP_IVAout_trans(hwp_angle, theta_offset, tx_arr, ty_arr, Sin)
#        py.plot(hwp_angle,IVA[j,:])
#        Poleff[j], phase[j] = IVA2PolEffPhase(IVA[j,:],hwp_angle)
#        Poleff[j], phase[j] = IVA2PolEffPhase_Ip(IVA[j,:],hwp_angle)
#        Poleff[j], phase[j] = IVAtrans2PolEffPhase_fit(hwp_angle,IVA[j,:])
        Poleff[j], phase[j] = IVAtrans2PolEffPhase_CFfit(hwp_angle,IVA[j,:])
#        Poleff[j], phase[j] = IVA2PolEffPhase_fit(hwp_angle,IVA[j,:])
# IVAtrans2PolEffPhase_fit(x,data):
# IVA2PolEffPhase_fit(x,data)

#    py.show()
#    sys.exit()

#    py.subplot(211)
#    py.plot(freq,Poleff)
#    py.subplot(212)
#    py.plot(fxx,txx)
#    py.plot(fxx,tyy)
#    py.show()
#    sys.exit()
    if option_dump!='':
        np.savez(option_dump,freq,Poleff)

    if(freq_c1 != False):
        IVA_mean1, IVA_std1 = IVA_Mean_RMS_inband(freq_c1, band_frachalf, freq, IVA)
    if(freq_c2 != False):
        IVA_mean2, IVA_std2 = IVA_Mean_RMS_inband(freq_c2, band_frachalf, freq, IVA)
    if(freq_c3 != False):
        IVA_mean3, IVA_std3 = IVA_Mean_RMS_inband(freq_c3, band_frachalf, freq, IVA)

    if plot==True:
        py.subplot(221)
        py.plot(freq*1e-9,Poleff)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,1.2]), '-y')
        py.ylim(np.min(Poleff)*0.95,np.max(Poleff)*1.05)
        py.ylabel('Mod eff')
        py.xlabel('Frequency [GHz]')

        py.subplot(223)
        py.plot(freq*1e-9,phase*180./pi)
        py.plot(np.ones(2)*freq_c1*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c1*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-b')
        py.plot(np.ones(2)*freq_c2*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        py.plot(np.ones(2)*freq_c2*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-r')
        if(freq_c3 != False):
            py.plot(np.ones(2)*freq_c3*(1.-band_frachalf)*1e-9,np.array([0.,360.]), '-y')
            py.plot(np.ones(2)*freq_c3*(1.+band_frachalf)*1e-9,np.array([0.,360.]), '-y')
        py.ylim(np.min(phase*180./pi)*0.95,np.max(phase*180./pi)*1.05)
        py.ylabel('Phase [degs]')
        py.xlabel('Frequency [GHz]')

        py.subplot(222)
        for j in range(0,num_freq):
            py.plot(hwp_angle/pi*180.,IVA[j,:])
        py.ylabel('IVA')
        py.xlabel('HWP angle [degs]')

        py.subplot(224)
        if(freq_c1 != False):
            py.plot(hwp_angle,IVA_mean1,label=str(freq_c1*1e-9)+'GHz')
        if(freq_c2 != False):
            py.plot(hwp_angle,IVA_mean2,label=str(freq_c2*1e-9)+'GHz')
        if(freq_c3 != False):
            py.plot(hwp_angle,IVA_mean3,label=str(freq_c3*1e-9)+'GHz')
        if((freq_c1 != False) or (freq_c2 != False) or (freq_c3 != False)):
            py.legend(loc='best')

        py.show()

    return freq, Poleff, phase



def Poleff_Phase_tolerance(par_in, n_o, n_e, freq_i, freq_f, freq_int, Sin,
                           freq_c1=False, freq_c2=False, freq_c3=False, band_frachalf=False):
    num_par = len(par_in)
    par = np.copy(par_in)
    num_rand = 30
    
    if(freq_c1 != False):
        Poleff1, phase1 = np.zeros(num_rand), np.zeros(num_rand)
    if(freq_c2 != False):
        Poleff2, phase2 = np.zeros(num_rand), np.zeros(num_rand)
    if(freq_c3 != False):
        Poleff3, phase3 = np.zeros(num_rand), np.zeros(num_rand)

    for ii in range(num_rand):
        for i in range(0,num_par):
            par[i] = (1.+0.1*np.random.rand())*par_in[i]
        
        theta_offset = par[0:num_par/2]
        thick = par[num_par/2:num_par]
        num_angle = 360*2
        hwp_angle = np.arange(0,num_angle)/float(num_angle)*pi/2.
        num_plate = len(theta_offset)
        i_angle = range(0,num_angle)
        num_freq = int((freq_f - freq_i)/freq_int + 1.)
        i_freq = range(0,num_freq)
        freq = np.arange(0,num_freq)/float(num_freq)*(freq_f-freq_i)+freq_i
        IVA = np.zeros((num_freq,num_angle)) 

        Poleff, phase = np.zeros(num_freq), np.zeros(num_freq)
        i_ahwp = range(0,num_plate)

        for j in range(0,num_freq):
            delta_arr = []
            for i in range(0,num_plate):
                delta = 2.*pi*(n_e-n_o)*thick[i]/c*freq[j]
                delta_arr = np.hstack((delta_arr,delta))
            IVA[j,:] = calHWP_IVAout(hwp_angle,theta_offset,delta_arr,Sin)
            Poleff[j], phase[j] = IVA2PolEffPhase(IVA[j,:],hwp_angle)

        if(freq_c1 != False):
            IVA_mean1, IVA_std1 = IVA_Mean_RMS_inband(freq_c1, band_frachalf, freq, IVA)
            Poleff1[ii], phase1[ii] = IVA2PolEffPhase(IVA_mean1,hwp_angle)
        if(freq_c2 != False):
            IVA_mean2, IVA_std2 = IVA_Mean_RMS_inband(freq_c2, band_frachalf, freq, IVA)
            Poleff2[ii], phase2[ii] = IVA2PolEffPhase(IVA_mean2,hwp_angle)
        if(freq_c3 != False):
            IVA_mean3, IVA_std3 = IVA_Mean_RMS_inband(freq_c3, band_frachalf, freq, IVA)
            Poleff3[ii], phase3[ii] = IVA2PolEffPhase(IVA_mean3,hwp_angle)

    nbin=num_rand/10.
    par = [0.,1.,2.]
    if(freq_c1 != False):
        py.subplot(231)
        mylib.plot_hist(Poleff1,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
        py.subplot(232)
        mylib.plot_hist(phase1,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
    if(freq_c2 != False):
        py.subplot(233)
        mylib.plot_hist(Poleff2,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
        py.subplot(234)
        mylib.plot_hist(phase2,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
    if(freq_c3 != False):
        py.subplot(235)
        mylib.plot_hist(Poleff3,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
        py.subplot(236)
        mylib.plot_hist(phase3,nbin,par=par,fit=True,init_auto=True,xtitle=-1,no_plot=False,normed=False)
    
    py.show()

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
        print('+++  Fit the histogram with Gaussian +++')
        if init_auto: par0 = [np.max(non),np.median(x),np.std(x)]
        if init_auto == False: par0 = par
        print('initial guess:', par0)
        x = np.arange(min(bincenters),max(bincenters),(max(bincenters)-min(bincenters))/500.)
        par, fopt,iterout,funcalls,warnflag=fmin(chi_nosigma,par0,args=(bincenters,non),maxiter=10000,maxfun=10000,xtol=0.01,full_output=1)
        if no_plot == False: py.plot(x,func_gauss(par,x),'r', linewidth=1)
#        if no_plot == False: py.plot(bincenters,func_gauss(par,bincenters),'r', linewidth=1)
        #y = mlab.normpdf(bincenters, par[1], par[2])
        #l = py.plot(bincenters, y, 'r--', linewidth=1)
        print('fitted parameters:', par)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    if xtitle != -1: py.xlabel(xtitle)
    py.ylabel('Count')
    #py.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    py.xlim(min(bins), max(bins))
#    py.ylim(0, 0.03)
    py.grid(True)

#    py.show()

    return np.array(par)

def calExEy2Stokes(ex,ey):
    I =   np.real(ex*np.conjugate(ex)+ey*np.conjugate(ey))
    Q =   np.real(ex*np.conjugate(ex)-ey*np.conjugate(ey))
    U =   2.*np.real(ex*np.conjugate(ey))
    V = - 2.*np.imag(ex*np.conjugate(ey))
    return np.array([I,Q,U,V])

def calStokes2PolAng(Stokes):
    P = np.sqrt(Stokes[1]**2+Stokes[2]**2+Stokes[3]**2)/Stokes[0]
    angle = 0.5*np.arctan(Stokes[2]/Stokes[1])
    return np.array([P,angle])


