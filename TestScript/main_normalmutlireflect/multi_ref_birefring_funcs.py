import numpy as np
import pylab as py
import lib_optics as lib_o
import sys

pi = np.pi
c = 299792458.
perm0 = (4.*pi*1.e-7)
radeg = (180./pi)
n_air = 1.

def Mueller_HWP_trans1(delretard):
    me = np.zeros((4,4))
    me = np.matrix(me)
    me[0,0] = 1.
    me[1,0] = 0.
    me[2,0] = 0.
    me[3,0] = 0.
    me[0,1] = 0.
    me[1,1] = 1.
    me[2,1] = 0.
    me[3,1] = 0.
    me[0,2] = 0.
    me[1,2] = 0.
    me[2,2] =  np.cos(delretard)
    me[3,2] =  np.sin(delretard)
    me[0,3] = 0.
    me[1,3] = 0.
    me[2,3] = -np.sin(delretard)
    me[3,3] =  np.cos(delretard)
    return me

def cal_Mullermatrix_RotinvHWPRot(rho, offset_plate, delretard):
    mm = np.zeros((4,4))
    mm = np.matrix(mm)
    matrix_HWP = Mueller_HWP_trans1(delretard)
    matrix_rot     = mueller_rotation( rho,  offset_plate)
    matrix_rot_inv = mueller_rotation(-rho, -offset_plate)
#    mm[:,:] = matrix_rot_inv[:,:] * matrix_HWP[:,:] * matrix_rot[:,:]
    mm[:,:] = matrix_rot_inv * matrix_HWP * matrix_rot
    return mm

def retard_n(n, HWP_thick, freq):
    c = 3.e8
    ret_out = 8.*np.arctan(1.) * n * HWP_thick * freq / c
    return ret_out

def unitMullermatrix(a):
    unitmm = np.zeros((4,4))
    unitmm = np.matrix(unitmm)
    unitmm[0,0] = 1.
    unitmm[1,0] = 0.
    unitmm[2,0] = 0.
    unitmm[3,0] = 0.
    unitmm[0,1] = 0.
    unitmm[1,1] = 1.
    unitmm[2,1] = 0.
    unitmm[3,1] = 0.
    unitmm[0,2] = 0.
    unitmm[1,2] = 0.
    unitmm[2,2] = 1.
    unitmm[3,2] = 0.
    unitmm[0,3] = 0.
    unitmm[1,3] = 0.
    unitmm[2,3] = 0.
    unitmm[3,3] = 1.
    return unitmm

#;---
#; this convention of sign is not the same as Shurcliff
def mueller_rotation(rho, offset):
    mrot = np.zeros((4,4))
    mrot = np.matrix(mrot)

    mrot[0,0] = 1.
    mrot[1,0] = 0.
    mrot[2,0] = 0.
    mrot[3,0] = 0.
    mrot[0,1] = 0.
    mrot[1,1] = np.cos( 2. * rho + 2. * offset)
    mrot[2,1] = np.sin( 2. * rho + 2. * offset)
    mrot[3,1] = 0.
    mrot[0,2] = 0.
    mrot[1,2] = -np.sin( 2. * rho + 2. * offset)
    mrot[2,2] =  np.cos( 2. * rho + 2. * offset)
    mrot[3,2] = 0.
    mrot[0,3] = 0.
    mrot[1,3] = 0.
    mrot[2,3] = 0.
    mrot[3,3] = 1.
    return mrot

def mueller_transmission(too, toe, teo, tee):
    mt = np.zeros((4,4),dtype=complex)
    mt = np.matrix(mt)

    mt[0,0] =                          too*np.conj(too) + teo*np.conj(teo) + toe*np.conj(toe) + tee*np.conj(tee)
    mt[0,1] =                          too*np.conj(too) + teo*np.conj(teo) - toe*np.conj(toe) - tee*np.conj(tee)
    mt[0,2] =                          too*np.conj(toe) + teo*np.conj(tee) + toe*np.conj(too) + tee*np.conj(teo)
    mt[0,3] = -np.complex(0.,1.)*(too*np.conj(toe) + teo*np.conj(tee) - toe*np.conj(too) - tee*np.conj(teo))

    mt[1,0] =                          too*np.conj(too) - teo*np.conj(teo) + toe*np.conj(toe) - tee*np.conj(tee)
    mt[1,1] =                          too*np.conj(too) - teo*np.conj(teo) - toe*np.conj(toe) + tee*np.conj(tee)
    mt[1,2] =                          toe*np.conj(too) - tee*np.conj(teo) + too*np.conj(toe) - teo*np.conj(tee)
    mt[1,3] = -np.complex(0.,1.)*(toe*np.conj(too) - tee*np.conj(teo) - too*np.conj(toe) + teo*np.conj(tee))

    mt[2,0] =                          too*np.conj(teo) + teo*np.conj(too) + toe*np.conj(tee) + tee*np.conj(toe)
    mt[2,1] =                          too*np.conj(teo) + teo*np.conj(too) - toe*np.conj(tee) - tee*np.conj(toe)
    mt[2,2] =                          too*np.conj(tee) + teo*np.conj(toe) + toe*np.conj(teo) + tee*np.conj(too)
    mt[2,3] = -np.complex(0.,1.)*(too*np.conj(tee) + teo*np.conj(toe) - toe*np.conj(teo) - tee*np.conj(too))

    mt[3,0] =  np.complex(0.,1.)*(too*np.conj(teo) - teo*np.conj(too) + toe*np.conj(tee) - tee*np.conj(toe))
    mt[3,1] =  np.complex(0.,1.)*(too*np.conj(teo) - teo*np.conj(too) - toe*np.conj(tee) + tee*np.conj(toe))
    mt[3,2] =  np.complex(0.,1.)*(too*np.conj(tee) - teo*np.conj(toe) + toe*np.conj(teo) - tee*np.conj(too))
    mt[3,3] =                          too*np.conj(tee) - teo*np.conj(toe) - toe*np.conj(teo) + tee*np.conj(too)

    return 0.5*mt

def Muller_input(I, Q, U, V):
    m_in = np.zeros((4))
    m_in[0] = I
    m_in[1] = Q
    m_in[2] = U
    m_in[3] = V
    return m_in

def multibire_too(a):
    too = a[1,1]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    return too

def multibire_toe(a):
#    toe = - a[1,0]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    toe = - a[0,1]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    return toe

def multibire_teo(a):
#    teo = - a[0,1]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    teo = - a[1,0]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    return teo

def multibire_tee(a):
    tee = a[0,0]/(a[1,1]*a[0,0]-a[0,1]*a[1,0])
    return tee

def multibire_roo(a):
    roo = (a[1,1]*a[2,0]-a[2,1]*a[1,0])/(a[0,0]*a[1,1]-a[0,1]*a[1,0])
    return roo

def multibire_roe(a):
    roe = - (a[2,0]*a[0,1]-a[0,0]*a[2,1])/(a[0,0]*a[1,1]-a[0,1]*a[1,0])
    return roe

def multibire_reo(a):
    reo = (a[3,0]*a[1,1]-a[3,1]*a[1,0])/(a[0,0]*a[1,1]-a[0,1]*a[1,0])
    return reo

def multibire_ree(a):
    ree = - (a[3,0]*a[0,1]-a[3,1]*a[0,0])/(a[0,0]*a[1,1]-a[0,1]*a[1,0])
    return ree

def multibire_Binv(phi1, lamboI, lambeI, const):
    Binvmat = np.zeros((4,4))
    Binvmat = np.matrix(Binvmat)

    Binvmat[0,0]=  np.cos(phi1)
    Binvmat[0,1]=  np.sin(phi1)
    Binvmat[0,2]= -np.sin(phi1) * lamboI/const
    Binvmat[0,3]=  np.cos(phi1) * lamboI/const

    Binvmat[1,0]= -np.sin(phi1)
    Binvmat[1,1]=  np.cos(phi1)
    Binvmat[1,2]= -np.cos(phi1) * lambeI/const
    Binvmat[1,3]= -np.sin(phi1) * lambeI/const

    Binvmat[2,0]=  np.cos(phi1)
    Binvmat[2,1]=  np.sin(phi1)
    Binvmat[2,2]=  np.sin(phi1) * lamboI/const
    Binvmat[2,3]= -np.cos(phi1) * lamboI/const

    Binvmat[3,0]= -np.sin(phi1)
    Binvmat[3,1]=  np.cos(phi1)
    Binvmat[3,2]=  np.cos(phi1) * lambeI/const
    Binvmat[3,3]=  np.sin(phi1) * lambeI/const

    print 'Binvmat'
    print lamboI, lambeI, const
    print lamboI/const, lambeI/const
    print Binvmat
#    return Binvmat
    return 0.5 * Binvmat

def multibire_Atil(m, lamboIII, lambeIII, const):
    Amat = np.zeros((4,2),dtype=complex)
    Amat = np.matrix(Amat)

    Amat[0,0] = m[0,0] + const/lamboIII*m[0,3]
    Amat[0,1] = m[0,1] - const/lambeIII*m[0,2]

    Amat[1,0] = m[1,0] + const/lamboIII*m[1,3]
    Amat[1,1] = m[1,1] - const/lambeIII*m[1,2]

    Amat[2,0] = m[2,0] + const/lamboIII*m[2,3]
    Amat[2,1] = m[2,1] - const/lambeIII*m[2,2]

    Amat[3,0] = m[3,0] + const/lamboIII*m[3,3]
    Amat[3,1] = m[3,1] - const/lambeIII*m[3,2]

    # Amat[0,0]= m[0,0] + const/lamboIII*m[3,0]
    # Amat[1,0]= m[1,0] - const/lambeIII*m[2,0]

    # Amat[0,1]= m[0,1] + const/lamboIII*m[3,1]
    # Amat[1,1]= m[1,1] - const/lambeIII*m[2,1]

    # Amat[0,2]= m[0,2] + const/lamboIII*m[3,2]
    # Amat[1,2]= m[1,2] - const/lambeIII*m[2,2]

    # Amat[0,3]= m[0,3] + const/lamboIII*m[3,3]
    # Amat[1,3]= m[1,3] - const/lambeIII*m[2,3]

    print 'm'
    print m
    print 'Amat'
    print Amat

    return Amat

def multibire_MIIinv(lamboII, lambeII, phi2, const, d):

    MIIinvmat = np.zeros((4,4),dtype=complex)
    MIIinvmat = np.matrix(MIIinvmat)

    freq = 1./(const*perm0)

    k = 2.*pi*freq/c

    n_o = c/(freq*lamboII)
    n_e = c/(freq*lambeII)

    ho = n_o*d
    he = n_e*d

    delo = k*ho
    dele = k*he

    MIIinvmat[0,0] =                     np.cos(phi2) * complex( np.cos( delo), np.sin( delo))
    MIIinvmat[1,0] =                     np.sin(phi2) * complex( np.cos( delo), np.sin( delo))
    MIIinvmat[2,0] = - lamboII / const * np.sin(phi2) * complex( np.cos( delo), np.sin( delo))
    MIIinvmat[3,0] =   lamboII / const * np.cos(phi2) * complex( np.cos( delo), np.sin( delo))

    MIIinvmat[0,1] =                   - np.sin(phi2) * complex( np.cos( dele), np.sin( dele))
    MIIinvmat[1,1] =                     np.cos(phi2) * complex( np.cos( dele), np.sin( dele))
    MIIinvmat[2,1] = - lambeII / const * np.cos(phi2) * complex( np.cos( dele), np.sin( dele))
    MIIinvmat[3,1] = - lambeII / const * np.sin(phi2) * complex( np.cos( dele), np.sin( dele))

    MIIinvmat[0,2] =                     np.cos(phi2) * complex( np.cos(-delo), np.sin(-delo))
    MIIinvmat[1,2] =                     np.sin(phi2) * complex( np.cos(-delo), np.sin(-delo))
    MIIinvmat[2,2] =   lamboII / const * np.sin(phi2) * complex( np.cos(-delo), np.sin(-delo))
    MIIinvmat[3,2] = - lamboII / const * np.cos(phi2) * complex( np.cos(-delo), np.sin(-delo))

    MIIinvmat[0,3] =                   - np.sin(phi2) * complex( np.cos(-dele), np.sin(-dele))
    MIIinvmat[1,3] =                     np.cos(phi2) * complex( np.cos(-dele), np.sin(-dele))
    MIIinvmat[2,3] =   lambeII / const * np.cos(phi2) * complex( np.cos(-dele), np.sin(-dele))
    MIIinvmat[3,3] =   lambeII / const * np.sin(phi2) * complex( np.cos(-dele), np.sin(-dele))

    return 0.5 * MIIinvmat

def multibire_MI(lamboII, lambeII, const):

    MImat = np.zeros((4,4))
    MImat = np.matrix(MImat)

    MImat[0,0] = 1.
    MImat[0,1] = 0.
    MImat[0,2] = 1.
    MImat[0,3] = 0.

    MImat[1,0] = 0.
    MImat[1,1] = 1.
    MImat[1,2] = 0.
    MImat[1,3] = 1.

    MImat[2,0] = 0.
    MImat[2,1] = - const/lambeII
    MImat[2,2] = 0.
    MImat[2,3] =   const/lambeII

    MImat[3,0] =   const/lamboII
    MImat[3,1] = 0.
    MImat[3,2] = - const/lamboII
    MImat[3,3] = 0.

    return MImat


def multibire_singleHWP_m(lamboII, lambeII, phi2, const, d):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[0], const, d) #; 1st plate
#    print m
#    sys.exit()
    return m

def multibire_singleHWP_ARcoated_m(lamboII, lambeII, phi2, const, d_HWP, lambAR, d_AR):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[0], const, d_AR) \
    * multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[1], const, d_HWP) \
    * multibire_MI( lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[2], const, d_AR) 
    return m

def multibire_three_stackAHWP_m(lamboII, lambeII, phi2, const, d):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[0], const, d) \
 	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[1], const, d) \
	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[2], const, d) 	# ; 3rd plate
    return m

def multibire_three_stackAHWP_ARcoated_m(lamboII, lambeII, phi2, const, d_HWP, lambAR, d_AR):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[0], const, d_AR) \
    * multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[1], const, d_HWP) \
 	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[2], const, d_HWP) \
	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[3], const, d_HWP) \
	* multibire_MI(lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[4], const, d_AR)
    return m

def multibire_five_stackAHWP_m(lamboII, lambeII, phi2, const, d):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[0], const, d) \
	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[1], const, d) \
	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[2], const, d) \
	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[3], const, d) \
	*  multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[4], const, d) 
    return m

def multibire_five_stackAHWP_ARcoated_m(lamboII, lambeII, phi2, const, d_HWP, lambAR, d_AR):
    m = np.zeros((4,4),dtype=complex)
    m = np.matrix(m)
    m = multibire_MI(lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[0], const, d_AR) \
    * multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[1], const, d_HWP) \
	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[2], const, d_HWP) \
	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[3], const, d_HWP) \
	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[4], const, d_HWP) \
	* multibire_MI(lamboII, lambeII, const) * multibire_MIIinv(lamboII, lambeII, phi2[5], const, d_HWP) \
	* multibire_MI(lambAR, lambAR, const) * multibire_MIIinv( lambAR,  lambAR, phi2[6], const, d_AR)
    return m


def multibire_Iin_r_t(E, alpha, r_t, noI, neI, rel_ang):

    Eino = E * np.cos(alpha)
    Eine = E * np.sin(alpha)

    co_roo = r_t[1]
    co_roe = r_t[2]
    co_reo = r_t[3]
    co_ree = r_t[4]

    co_too = r_t[5]
    co_toe = r_t[6]
    co_teo = r_t[7]
    co_tee = r_t[8]

    Ero = co_roo * Eino + co_roe * Eine
    Ere = co_reo * Eino + co_ree * Eine
    Eto = co_too * Eino + co_toe * Eine
    Ete = co_teo * Eino + co_tee * Eine

    Etop = np.cos(rel_ang) * Eto - np.sin(rel_ang) * Ete
    Etep = np.sin(rel_ang) * Eto + np.cos(rel_ang) * Ete

    Et_detecor = np.cos(-alpha) * Etop - np.sin(-alpha) * Etep

    Iino = 0.5 * noI/(c*perm0) * np.abs(Eino)**2.
    Iine = 0.5 * neI/(c*perm0) * np.abs(Eine)**2.

    Iro = 0.5 * noI/(c*perm0) * np.abs(Ero)**2.
    Ire = 0.5 * neI/(c*perm0) * np.abs(Ere)**2.

    Ito = 0.5 * noI/(c*perm0) * np.abs(Etop)**2.
    Ite = 0.5 * neI/(c*perm0) * np.abs(Etep)**2.

    It_detector = 0.5 * neI/(c*perm0) * np.abs(Et_detecor)**2.

    num_freq = len(r_t[0])

    output = np.zeros((7, num_freq))

    output[0] = Iino
    output[1] = Iine
    output[2] = Iro
    output[3] = Ire
    output[4] = Ito
    output[5] = Ite
    output[6] = It_detector

    return output

def multibire_Iout_HWPangle(output_r_t, noI, neI, relative_angle):

    E = 1.
    relative_angle = 0./180.*pi

    num_alpha = 360
    Iin = np.zeros(num_alpha)
    Iin_total = np.zeros(num_alpha)
    It_detec = np.zeros(num_alpha)
    It_detec_total = np.zeros(num_alpha)
    alpha_index = np.zeros(num_alpha)

    for i in range(0, num_alpha): 
    	alpha = i/180.*pi
    	intensity = multibire_Iin_r_t(E, alpha, output_r_t, noI, neI, relative_angle)
    	alpha_index[i] = alpha

    	Iin_total[i] = total(intensity[0]+intensity[1])
    	It_detec_total[i] = np.sum(intensity[6])

    output_Irho = np.zeros(2,num_alpha)
    output_Irho[0] = alpha_index
    output_Irho[1] = It_detec_total/Iin_total

    return output_Irho

#;-------------------------------------------------------------------------------------------------------------------
#;-------------------------------------------------------------------------------------------------------------------
#; This function calculates the polarization efficiency as a function of the HWP angle with N stack AHWP
#;
def Mueller_AHWP_Nplates_trans1_Irhoout(num_plate, n_o, n_e, HWP_thick, num_rho, hwp_anglemax, offset_plate, num_freq, freq_i, freq_f, SVin):

#;-- define array
    SVout = np.zeros((4,num_freq, num_rho))
    output = np.zeros((2,num_freq, num_rho))

#;-- freqency range
    freq = freq_i + np.arange(num_freq)/(np.float(num_freq)-1.) * (freq_f - freq_i) #; [Hz]

#;-- print output offset angles
#    print 'offset angles =   ', offset_plate[*]/pi*180.;k+1, ': ', offset_plate2/pi*180., offset_plate3/pi*180., offset_plate4/pi*180.

#;-- define grid muller matrix
    gt_xx = complex(1.,0.)
    gt_yy = complex(0.,0.)
    mmatrix_grid = mueller_transmission(gt_xx,gt_yy,gt_yy,gt_yy)

#;-- freq loop
    for j in range(0, num_freq):
#        ;-- cal retardance for n_o and n_e at each freq
        ret_o = retard_n(n_o, HWP_thick, freq[j])
        ret_e = retard_n(n_e, HWP_thick, freq[j])
        delretard = np.abs(ret_o - ret_e)
        mm = np.zeros((num_rho,4,4), dtype=complex)
#        ;-- hwp angle loop
        for i in range(0, num_rho):
            rho = np.arange(num_rho)/ np.float(num_rho) * hwp_anglemax
#            ;-- unit 4x4 matrix
            unitmm = unitMullermatrix(1.)
            m_matrix = unitmm
#            ;-- multiply wave plate loop
            for k in range(0,num_plate):
                m_RinvHR = cal_Mullermatrix_RotinvHWPRot(rho[i], offset_plate[k], delretard)
                m_matrix = m_RinvHR * m_matrix
            mm[i,:,:] = mmatrix_grid * m_matrix[:,:]

        SVout[0,j,:] = SVin[0]*mm[:,0,0] + SVin[1]*mm[:,0,1] + SVin[2]*mm[:,0,2] + SVin[3]*mm[:,0,3]
        SVout[1,j,:] = SVin[0]*mm[:,1,0] + SVin[1]*mm[:,1,1] + SVin[2]*mm[:,1,2] + SVin[3]*mm[:,1,3]
        SVout[2,j,:] = SVin[0]*mm[:,2,0] + SVin[1]*mm[:,2,1] + SVin[2]*mm[:,2,2] + SVin[3]*mm[:,2,3]
        SVout[3,j,:] = SVin[0]*mm[:,3,0] + SVin[1]*mm[:,3,1] + SVin[2]*mm[:,3,2] + SVin[3]*mm[:,3,3]

        output[0,j,:] = rho[:]
        output[1,j,:] = SVout[0,j,:]

    return output

#;==========================================================================================
#; output: coefficient of reflection and transmission (oo, oe, eo, ee) at given frequency
def multi_ref_birefringent_singlefreq(n_oI, n_eI, n_oII, n_eII, n_oIII, n_eIII, phi, freq, d_HWP, n_AR, d_AR, config_choice):

    const = 1. / (freq*perm0)

    lamboI = c/(freq*n_oI) #; o-wavelength in I region
    lambeI = c/(freq*n_eI) #; e-wavelength in I region

    lamboII = c/(freq*n_oII) #; o-wavelength in II region
    lambeII = c/(freq*n_eII) #; e-wavelength in II region

    lamboIII = c/(freq*n_oIII) #; o-wavelength in III region
    lambeIII = c/(freq*n_eIII) #; e-wavelength in III region

    lamb_AR = c/(freq*n_AR) #; e-wavelength in III region

    phi1 = phi[0] #; relative angle between the first layer(often vacuum) and the second layer(AR or HWP)
    phi2 = phi[1:]

    m = np.zeros((4,4),dtype=complex)
    a = np.zeros((4,2),dtype=complex)
    m = np.matrix(m)
    a = np.matrix(a)

    if config_choice == 1:
        print 'config_choice = 1: single HWP'
        m = multibire_singleHWP_m( lamboII, lambeII, phi2, const, d_HWP)

    if config_choice == 2:
        m = multibire_singleHWP_ARcoated_m( lamboII, lambeII, phi2, const, d_HWP, lamb_AR, d_AR)

    if config_choice == 3:
        m = multibire_three_stackAHWP_m( lamboII, lambeII, phi2, const, d_HWP)

    if config_choice == 4:
        m = multibire_three_stackAHWP_ARcoated_m( lamboII, lambeII, phi2, const, d_HWP, lamb_AR, d_AR)

    if config_choice == 5:
        m = MULTIBIRE_FIVE_STACKAHWP_M( lamboII, lambeII, phi2, const, d_HWP)

    if config_choice == 6:
        m = MULTIBIRE_FIVE_STACKAHWP_ARCOATED_M( lamboII, lambeII, phi2, const, d_HWP, lamb_AR, d_AR)

    ## BE SURE THAT THIS MODIFICATION IS OK. (4,4), (2,4)
    a = multibire_Binv(phi1, lamboI, lambeI, const) * multibire_Atil( m, lamboIII, lambeIII, const)

    func_output = np.zeros((8),dtype=complex)

    func_output[0] = multibire_roo(a)
    func_output[1] = multibire_roe(a)
    func_output[2] = multibire_reo(a)
    func_output[3] = multibire_ree(a)

    func_output[4] = multibire_too(a)
    func_output[5] = multibire_toe(a)
    func_output[6] = multibire_teo(a)
    func_output[7] = multibire_tee(a)

#    print a
#    print func_output[0]
#    sys.exit()

    return func_output


#;=======================================================================================================
#; output: coefficient of reflection and transmission (oo, oe, eo, ee)
#;         as a function of freq (between freq_i and freq_f)
#;
#; * coefficients of reflection and transmission are calculated from the boundary condition in Maxwell's eqs
#;   (multiple reflection) in the multi-layer of birefringent material
#;
def multi_ref_birefringent_freqcoverage(nsa_o, nsa_e, d_HWP, phi, freq_i, freq_f, num_freq, n_AR, d_AR):

    num_plate = len(phi)
    #;print, num_plate
    if num_plate == 2: config_choice = 1
    if num_plate == 4: config_choice = 2 #; changed 9-11-07 for single HWP + single ARC cal for SPIDER
    if num_plate == 6: config_choice = 5

#;;-------------------------------------------------------------------------------------
    print 'choose from below'
    print '1. air - single HWP - air'
    print '2. air - single layer AR coat - HWP - single layer AR coat - air'
    print '3. air - 3 stack AHWP - air'
    print '4. air - single layer AR coat - 3 stack AHWP - single layer AR coat- air'
    print '5. air - 5 stack AHWP - air'
    print '6. air - single layer AR coat - 5 stack AHWP - single layer AR coat- air'

#;read, 'pick number', config_choice
#;;--------------------------------------------------------------------------------------
    freq_index = np.zeros(num_freq)
#    freq_resol = (freq_f-freq_i)/np.float(num_freq-1.)
    freq = freq_i + np.arange(num_freq)/(np.float(num_freq)-1.) * (freq_f - freq_i) #; [Hz]

    output_roo = np.zeros((num_freq),dtype=complex)
    output_roe = np.zeros((num_freq),dtype=complex)
    output_reo = np.zeros((num_freq),dtype=complex)
    output_ree = np.zeros((num_freq),dtype=complex)

    output_too = np.zeros((num_freq),dtype=complex)
    output_toe = np.zeros((num_freq),dtype=complex)
    output_teo = np.zeros((num_freq),dtype=complex)
    output_tee = np.zeros((num_freq),dtype=complex)

    single_freq_output = np.zeros((8),dtype=complex)
    for i in range(0,num_freq):
#        freq = freq_i + i * freq_resol
        single_freq_output = multi_ref_birefringent_singlefreq( 1., 1., nsa_o, nsa_e, 1., 1., \
                                                                phi, freq[i], d_HWP, n_AR, d_AR, \
                                                                config_choice)

        output_roo[i] = single_freq_output[0]
        output_roe[i] = single_freq_output[1]
        output_reo[i] = single_freq_output[2]
        output_ree[i] = single_freq_output[3]

        output_too[i] = single_freq_output[4]
        output_toe[i] = single_freq_output[5]
        output_teo[i] = single_freq_output[6]
        output_tee[i] = single_freq_output[7]

#        print ''
#        print np.abs(output_roo)
#        print np.abs(output_roe)
#        print np.abs(output_reo)
#        print np.abs(output_ree)
#        sys.exit()

    multi_freq_output = np.zeros((9, num_freq),dtype=complex)
    multi_freq_output[0] = freq

    multi_freq_output[1] = output_roo
    multi_freq_output[2] = output_roe
    multi_freq_output[3] = output_reo
    multi_freq_output[4] = output_ree

    multi_freq_output[5] = output_too
    multi_freq_output[6] = output_toe
    multi_freq_output[7] = output_teo
    multi_freq_output[8] = output_tee

    return multi_freq_output


def Mueller_Irho_reffreq(n_o, n_e, d_HWP, phi, n_AR, d_AR, \
                        num_freq, freq_i, freq_f, \
                        num_rho, hwp_anglemax, offset_plate_trans1, SVin):
#;------------------------------
#;-- define output variables
    SVout = np.zeros((4, num_freq, num_rho))
#;------------------------------

#;------------------------------
#; call function to calculate coefficient of transmission and reflection
#;    roo, roe, reo, ree, too, toe, teo, tee
    output_r_t = multi_ref_birefringent_freqcoverage(n_o, n_e, d_HWP, phi, freq_i, freq_f, num_freq, n_AR, d_AR)

    freq_array = output_r_t[0]
    roo = output_r_t[1]
    roe = output_r_t[2]
    reo = output_r_t[3]
    ree = output_r_t[4]

    too = output_r_t[5]
    toe = output_r_t[6]
    teo = output_r_t[7]
    tee = output_r_t[8]
#;-------------------------------
#; for loop
#; j: frequency
#; i: rho
    fit_result = np.zeros((2,num_freq))

#;-- define grid muller matrix, transmission axis along x
#    txx=1.
#    txy=tyx=tyy=0.
#    mmatrix_grid = mueller_transmission(txx,txy,tyx,tyy)
    gt_xx = complex(1.,0.)
    gt_yy = complex(0.,0.)
    mmatrix_grid = mueller_transmission(gt_xx,gt_yy,gt_yy,gt_yy)

#;-- loop on frequency
    for j in range(0, num_freq):
        freq = freq_array[j]

#   ;-- form a transmission Jones matrix
        mmatrix_ahwp = mueller_transmission(too[j], toe[j], teo[j], tee[j])

#    ;-- loop on HWP angle, rho
        mm = np.zeros((num_rho,4,4), dtype=complex)
        rho = np.arange(num_rho)/ np.float(num_rho) * hwp_anglemax 
        for i in range(0, num_rho):
            rho_i = rho[i]
#         ;-- define roation matrix
            matrix_rot = mueller_rotation(rho_i, 0.)
            matrix_rot_inv = mueller_rotation(-rho_i, 0.)
            mm[i] = mmatrix_grid * matrix_rot_inv * mmatrix_ahwp * matrix_rot
#            mm[i,:,:] = mmatrix_grid[:,:] * matrix_rot_inv[:,:] * mmatrix_ahwp[:,:] * matrix_rot[:,:]
#            print rho_j, rho_j*radeg
#            print matrix_rot_inv, matrix_rot
#            print mmatrix_grid, matrix_rot_inv, mmatrix_ahwp, matrix_rot
#            mm[i] = mmatrix_grid * matrix_rot_inv * mmatrix_ahwp * matrix_rot
#            print mm[i,:,:]
#        sys.exit()
#        SVout[0,j,:] = SVin[0]*mm[:,0,0] + SVin[1]*mm[:,1,0] + SVin[2]*mm[:,2,0] + SVin[3]*mm[:,3,0]
        SVout[0,j,:] = SVin[0]*mm[:,0,0] + SVin[1]*mm[:,1,0] + SVin[2]*mm[:,2,0] + SVin[3]*mm[:,3,0]
#        print SVin[0],mm[:,0,0], SVin[1],mm[:,1,0], SVin[2],mm[:,2,0], SVin[3],mm[:,3,0]
#        print SVout[0,j,:]
#        SVout[1,j,:] = SVin[0]*mm[:,0,1] + SVin[1]*mm[:,1,1] + SVin[2]*mm[:,2,1] + SVin[3]*mm[:,3,1]
#        SVout[2,j,:] = SVin[0]*mm[:,0,2] + SVin[1]*mm[:,1,2] + SVin[2]*mm[:,2,2] + SVin[3]*mm[:,3,2]
#        SVout[3,j,:] = SVin[0]*mm[:,0,3] + SVin[1]*mm[:,1,3] + SVin[2]*mm[:,2,3] + SVin[3]*mm[:,3,3]

    #   ;----------------------------------------------
    #   ; plot intensity - rho curve at given frequency
        fit_result[:,j] = lib_o.IVAtrans2PolEffPhase_fit(rho[:],SVout[0,j,:])
    

    py.figure()
    for j in range(0, num_freq):
        py.plot(rho*radeg, SVout[0,j,:])
#;-- the end of the freq for loop
    py.show()    
    sys.exit()
#;---------------------------------

#    print SVout[0,0,:]
#    print len(SVout[0,0,:])
#    py.plot(rho*radeg,SVout[0,0,:])
#    py.show()
#    sys.exit()
#;---------------------------------
#;-- sum over frequency
    I_detector = np.zeros((num_rho))
    for i in range(0, num_rho):
        I_detector[i]=np.sum(SVout[0,:,i])

    I_detector_norm = I_detector/np.float(num_freq)
#oplot, rho*360./(2.*!pi), I_detector_norm, line=0, thick=2, color = red, psym=0
#;-- the end of the summing over frequency
#;-----------------------------------------


#;-- print the parameters used in this simulation
    print ''
    print '-- print the parameters used in this simulation'
    print 'freq range [GHz] and resolution', freq_i*1.e-9, freq_f*1.e-9,':', num_freq
    print 'no, ne= ', n_o, n_e
    print 'HWP offset angle at each boundary phi [degree]= ', phi/pi*180
    print 'HWP thickness [mm]= ', d_HWP*1.e3
    print 'input Stokes vector = ', SVin
    print 'HWP offset angle used for trans 1 cal  [degree]= ', offset_plate_trans1/pi*180
    print 'AR coating, index and thickness [mm] if ARcoat is used', n_AR, d_AR*1.e3
    print 'resolution of the HWP angle', num_rho
    print ''

    #;--------------------------------------------------------------------------------
    #;-- calculate I, Q, U from Isum-rho curve and print
    fit_result_sumIout = lib_o.IVAtrans2PolEffPhase_fit(I_detector_norm[:], rho[:])
#    print rho*radeg,I_detector_norm
#    py.plot(rho*radeg,I_detector_norm)
#    py.show()
#    sys.exit()
    print ""
    print "maxwell boundary condition for birefringent layers"
    print "fit from summed Iout curve as a function of the HWP angle"
    print "Poleff", fit_result_sumIout[0]
    print "Phase", fit_result_sumIout[1]*radeg

    fileoutput = np.zeros((2,num_rho))
    fileoutput[0,:] = rho[:]
    fileoutput[1,:] = I_detector_norm[:]

    py.plot(fileoutput[0,:]*radeg,fileoutput[1,:])
#   ;-- generate an output file contains Isum out vs rho over the band
#   ;-- plot Roo, Roe, Ree as a function of the frequency

    py.figure()
#    py.plot(np.array(freq_array)*1.e-9, np.abs(roo)**2.)
#    py.plot(np.array(freq_array)*1.e-9, np.abs(ree)**2.)
#    py.plot(np.array(freq_array)*1.e-9, np.abs(roe)**2.)
#    py.plot(np.array(freq_array)*1.e-9, np.abs(reo)**2.)

    py.plot(np.array(freq_array)*1.e-9, np.abs(too)**2.)
    py.plot(np.array(freq_array)*1.e-9, np.abs(tee)**2.)
    py.plot(np.array(freq_array)*1.e-9, np.abs(toe)**2.)
    py.plot(np.array(freq_array)*1.e-9, np.abs(teo)**2.)

    py.xlim([freq_i*1.e-9, freq_f*1.e-9])
#    py.ylim([0.,1.1])
    py.xlabel('Frequency [GHz]')
    py.ylabel('Reflectance')
    py.title('Reflectance')
    py.ylim([0,1.1])

    print np.array(np.abs(freq_array))*1.e-9, 
#    print np.abs(roo)**2.
#    print np.abs(roe)**2.
#    print np.abs(reo)**2.
#    print np.abs(ree)**2.
#    print ''
    print np.abs(too)**2.
    print np.abs(toe)**2.
    print np.abs(teo)**2.
    print np.abs(tee)**2.
#    py.show()
#    sys.exit()

#   ;****************************************************************************************
#   ;-- below here is independent calculation from above
#   ;   calcualte the I-rho curve over frequency WITHOUT considering the transmission and reflection
#   ;   from any wave plates. therefore, no AR coating is considered in here.
#
#   ;-- calculate the I-rho for frequency range between freq_i and freq_f
    num_plate = len(offset_plate_trans1)
#   ;-- generate I vs rho
    Ioutput_trans1 = Mueller_AHWP_Nplates_trans1_Irhoout(num_plate, n_o, n_e, d_HWP, num_rho, \
                                                        hwp_anglemax, offset_plate_trans1, \
                                                        num_freq, freq_i, freq_f, SVin)
    freq_trans1 = freq_i + np.arange(num_freq)/(np.float(num_freq)-1.) * (freq_f - freq_i) #; [Hz]

#   ;-- extract I, Q, U from above I vs rho
#   ;Ioutput_trans1 = Mueller_AHWP_Nplates_trans1_IQUout(num_plate, n_o,
#   ;n_e, d_HWP, num_rho, $
#   ;hwp_anglemax, offset_plate_trans1, $ 
#   ;  num_freq, freq_i, freq_f, SVin)
#   ;-- rename outputs
    IQUoutput_trans1 = np.zeros((2,num_freq))

    for j in range(0, num_freq):
        IQUoutput_trans1[:,j] = lib_o.IVAtrans2PolEffPhase_fit(Ioutput_trans1[1,j,:], Ioutput_trans1[0,j,:])

    Poleff_trans1 = IQUoutput_trans1[0,:]
    phase4_trans1 = IQUoutput_trans1[1,:]

#   ;-- print out the extracted I, Q, U of Isum vs rho curve for transmission 1 case
    print ""
    print "transmission 1 is assumed"
    print "fit from summed Iout curve as a function of the HWP angle (of transmission 1 curve)"
    print "Poleff ", IQUoutput_trans1[0]
    print "Phase ", IQUoutput_trans1[1]
    print ';-----------------------------------------------------------------------------------------------------------'

#   ;-- plot polarization efficiency calculated from I, Q, U for both non trans 1 and trans 1 cases
    py.figure()
    py.ylim([0.,1.])
    py.xlabel('Frequency [GHz]')
    py.ylabel('Polarization efficiency')
    py.title('Polarization efficiency')
    py.plot(np.abs(freq_array)*1.e-9, fit_result[0])
    py.plot(np.abs(freq_trans1)*1.e-9, IQUoutput_trans1[0])
    print fit_result[0]
    print IQUoutput_trans1[0]

#   ;-- plot output phases calculated from Q, U for both non trans 1 and trans 1 cases
    py.figure()
    py.title('Output Phase')
    py.ylim([-180,180])
    py.ylabel('Output phase [degree]')
    py.plot(np.abs(freq_array)*1.e-9, fit_result[1]/pi*180.)
    py.plot(np.abs(freq_trans1)*1.e-9, IQUoutput_trans1[1]/pi*180.)

    print fit_result[1]/pi*180.
    print IQUoutput_trans1[1]/pi*180.
    print freq_array*1.e-9

#   ;-- trans 1: plot I-rho curve for given frequency and Isum-rho
    py.figure()
    py.title('I-rho curve from trans 1 code')
    py.xlim([0.,360.])
    py.ylim([0.,1.])
    py.ylabel('Iout')
    py.xlabel('HWP angle [degree]')
    py.plot(Ioutput_trans1[0,0,:]/pi*180., Ioutput_trans1[1,0,:])
    for j in range(1, num_freq):
        py.plot(Ioutput_trans1[0,j,:]/pi*180., Ioutput_trans1[1,j,:])

#   ; sum over frequency
    I_detector_trans1 = np.zeros(num_rho)
    for i in range(0, num_rho):
        I_detector_trans1[i] = np.sum(Ioutput_trans1[1,:,i])

    I_detector_norm_trans1 = I_detector_trans1/np.float(num_freq)
    py.plot(Ioutput_trans1[0,0,:]/pi*180., I_detector_norm_trans1)

    return fileoutput, 

#    py.show()
