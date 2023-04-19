import numpy as np
import pylab as py
import lib_Clmanip as libcl
import sys

'''
lib_FGtemplate.py
located at /Users/tomotake_matsumura/work/codes/custom_lib/
2015-5-4, written by T. Matsumura
'''

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the basic parameters
pi = np.pi
k = 1.3806488e-23
h = 6.62606957e-34
c = 299792458.
T_cmb = 2.725

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the foreground models

def dust_color_corr(nu):
    '''
    FUNCTION dust_color_corr,nu
    ; This procedure returns the color correction factor:
    ;
    ; dust_color_corr = x^beta*(e^x-1)^2/e^x/(e^{x*Tcmb/Td}-1)
    ;
    ; for b=1.59 and Td=19.6 K, with respect to 353 GHz.
    ; 
    ; E.Komatsu, December 24, 2014
    '''
    beta=1.59
    Td=19.6
    x=353./56.78
    f353=(np.exp(x)-1.)**2/np.exp(x)*x**(-1.+beta)/(np.exp(x*2.725/Td)-1.)
    x=nu/56.78
    fnu=(np.exp(x)-1.)**2/np.exp(x)*x**(-1.+beta)/(np.exp(x*2.725/Td)-1.)/f353
    return fnu

def dust_power_bb(nu,ell):
    '''
    FUNCTION dust_power_bb,nu,ell
    ; This procedure returns the BB power spectrum of dust [l(l+1)C_l/(2pi) in uK^2]
    ; Ref: "LR42" in Table 1 of Planck Intermediate Results XXX (arXiv:1409.5738)
    ; E.Komatsu, December 24, 2014
    '''
    Dl=dust_color_corr(nu)**2.*78.6*0.53*(ell/80.)**(-0.46)
    return Dl

def dust_power_ee(nu,ell):
    '''
    FUNCTION dust_power_ee,nu,ell
    ; This procedure returns the EE power spectrum of dust [l(l+1)C_l/(2pi) in uK^2]
    ; Ref: "LR42" in Table 1 of Planck Intermediate Results XXX (arXiv:1409.5738)
    ; E.Komatsu, December 24, 2014
    '''
    Dl=dust_color_corr(nu)**2.*78.6*(ell/80.)**(-0.34)
    return Dl

def gen_Cl_Creminelli(ell_in,nu,option_unit=''):
    '''
    NOTE that the parameters are from 
    Creminelli et al. with fsky=53%
    The output spectrum is C_l in uK^2 (NOT INCLUDE l(l+1)/(2pi)) without antenna option
    '''
    nu_s = 65.e9
    A_s = 2.1e-5  
    alpha_s = -2.6
    beta_s = -2.9
    ell_s = 80.

    nu_d = 353.e9
    A_d = 0.065   
    alpha_d = -2.42
    beta_d = 1.59
    T_d = 19.6
    ell_d = 80.

    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

    x_s = ((h/k)*nu_s)/T_cmb
    W_cmb_s = x_s**2 * np.exp(x_s)/(np.exp(x_s)-1.)**2
    W_s = W_cmb_s/W_cmb * (nu/nu_s)**beta_s
    S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s

    x_d = ((h/k)*nu_d)/T_cmb
    W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
    W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
    D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    if option_unit=='antenna':
        print '++++++++++++++++++++'
        print 'func gen_SDl_Creminelli()'
        print 'antenna temperature'

#        W_s = (nu/nu_s)**beta_s
#        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s
#        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
#        D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d
        S_l = S_l*W_cmb**2
        D_l = D_l*W_cmb**2

    return ell_in, S_l, D_l

def gen_Cl_Creminelli_parin(ell_in,nu,par,option_unit=''):
    '''
    NOTE that the parameters are from 
    Creminelli et al. with fsky=53%
    The output spectrum is C_l in uK^2 (NOT INCLUDE l(l+1)/(2pi)) without antenna option
    '''

    nu_s = par['nu_s']
    A_s = par['A_s']
    alpha_s = par['alpha_s']
    beta_s = par['beta_s']
    ell_s = par['ell_s']

    nu_d = par['nu_d']
    A_d = par['A_d']
    alpha_d = par['alpha_d']
    beta_d = par['beta_d']
    T_d = par['T_d']
    ell_d = par['ell_d']

    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

    x_s = ((h/k)*nu_s)/T_cmb
    W_cmb_s = x_s**2 * np.exp(x_s)/(np.exp(x_s)-1.)**2
    W_s = W_cmb_s/W_cmb * (nu/nu_s)**beta_s
    S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s

    x_d = ((h/k)*nu_d)/T_cmb
    W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
    W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
    D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    if option_unit=='antenna':
        print '++++++++++++++++++++'
        print 'func gen_SDl_Creminelli()'
        print 'antenna temperature'

#        W_s = (nu/nu_s)**beta_s
#        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s
#        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
#        D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d
        S_l = S_l*W_cmb**2
        D_l = D_l*W_cmb**2

    return ell_in, S_l, D_l

def gen_Cl_Creminellibased_2dustcomp(ell_in,nu,option_unit=''):
    '''
    NOTE that the parameters are from 
    Creminelli et al. with fsky=53%,
    output: 
        Creminelli's synch.
        two dust models calibrating off from the Creminelli's model + 0.17 uncertainty in beta
        break down of the two dust models
        original creminelli's model
    '''
    nu_s = 65.e9
    A_s = 2.1e-5  
    alpha_s = -2.6
    beta_s = -2.9
    ell_s = 80.

    nu_d1 = 353.e9
    A_d1 = 0.065 
    alpha_d1 = -2.42
    beta_d1 = 1.63
    T_d1 = 9.75
    ell_d1 = 80.

    nu_d2 = 353.e9
    A_d2 = 0.065 
    alpha_d2 = -2.42
    beta_d2 = 2.82
    T_d2 = 15.70
    ell_d2 = 80.

    f = 0.17692287
    s = 1.01121803

    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

    x_s = ((h/k)*nu_s)/T_cmb
    W_cmb_s = x_s**2 * np.exp(x_s)/(np.exp(x_s)-1.)**2
    W_s = W_cmb_s/W_cmb * (nu/nu_s)**beta_s
    S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s

    nu_d = 353.e9
    A_d = 0.065  
    alpha_d = -2.42
    beta_d = 1.59
    T_d = 19.6
    ell_d = 80.

    x_d1 = ((h/k)*nu_d1)/T_cmb
    x_d2 = ((h/k)*nu_d2)/T_cmb
    W_cmb_d1 = x_d1**2 * np.exp(x_d1)/(np.exp(x_d1)-1.)**2
    W_cmb_d2 = x_d2**2 * np.exp(x_d2)/(np.exp(x_d2)-1.)**2
    W_d1 = W_cmb_d1/W_cmb * (nu/nu_d1)**(1.+beta_d1) * (np.exp(((h/k)*nu_d1)/T_d1)-1.)/(np.exp(((h/k)*nu)/T_d1)-1.)
    W_d2 = W_cmb_d2/W_cmb * (nu/nu_d2)**(1.+beta_d2) * (np.exp(((h/k)*nu_d2)/T_d2)-1.)/(np.exp(((h/k)*nu)/T_d2)-1.)
    Dl1 = W_d1**2 * A_d1 * (ell_in/ell_d1)**alpha_d1
    Dl2 = W_d2**2 * A_d2 * (ell_in/ell_d2)**alpha_d2
    D_l = (1.-f)*Dl1 + f*s* Dl2

    x_d = ((h/k)*nu_d)/T_cmb
    W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
    W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
    Dl_single = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    if option_unit=='antenna':
        print '++++++++++++++++++++'
        print 'func gen_SDl_Creminelli()'
        print 'antenna temperature'

#        W_s = (nu/nu_s)**beta_s
#        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s

#        W_d1 = (nu/nu_d1)**(1.+beta_d1) * (np.exp(((h/k)*nu_d1)/T_d1)-1.)/(np.exp(((h/k)*nu)/T_d1)-1.)
#        W_d2 = (nu/nu_d2)**(1.+beta_d2) * (np.exp(((h/k)*nu_d2)/T_d2)-1.)/(np.exp(((h/k)*nu)/T_d2)-1.)
#        Dl1 = W_d1**2 * A_d1 * (ell_in/ell_d1)**alpha_d1
#        Dl2 = W_d2**2 * A_d2 * (ell_in/ell_d2)**alpha_d2
#        D_l = (1.-f)*Dl1 + f*s*Dl2
#        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
#        Dl_single = W_d**2 * A_d * (ell_in/ell_d)**alpha_d
        S_l = S_l*W_cmb**2
        Dl1 = Dl1*W_cmb**2
        Dl2 = Dl2*W_cmb**2
        D_l = (1.-f)*Dl1 + f*s*Dl2
        Dl_single = Dl_single*W_cmb**2

    return ell_in, S_l, D_l, (1.-f)*Dl1, f*s* Dl2, Dl_single


def gen_SDl_Creminelli(ell_in,nu,option=''):
    '''
    NOTE that the parameters are from 
    Creminelli et al. with fsky=53%
    '''
    nu_s = 65.e9
    A_s = 2.1e-5
    alpha_s = -2.6
    beta_s = -2.9
    ell_s = 80.

    nu_d = 353.e9
    A_d = 0.065
    alpha_d = -2.42
    beta_d = 1.59
    T_d = 19.6
    ell_d = 80.

    if option=='thermo':
        x = ((h/k)*nu)/T_cmb
        W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2
        
        x_s = ((h/k)*nu_s)/T_cmb
        W_cmb_s = x_s**2 * np.exp(x_s)/(np.exp(x_s)-1.)**2
        W_s = W_cmb_s/W_cmb * (nu/nu_s)**beta_s
        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s
        
        x_d = ((h/k)*nu_d)/T_cmb
        W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
        W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
        D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d
        return  ell_in, S_l, D_l

    if option=='antenna':
        W_s = (nu/nu_s)**beta_s
        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
        D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d
        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s
        return ell_in, S_l, D_l

def gen_Cl_dustcomp(ell_in,nu, \
                    nu_d, A_d, alpha_d, beta_d, T_d, ell_d, \
                    option_unit=''):
    '''

    '''

    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

    x_d = ((h/k)*nu_d)/T_cmb
    W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
    W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
    Cl_d = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    if option_unit=='antenna':
        print '++++++++++++++++++++'
        print 'func gen_Cl_dustcomp()'
        print 'antenna temperature'

        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
        Cl_d = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    return ell_in, Cl_d

def gen_Cl_Meisner(ell_in,nu,option_unit=''):
    '''
    NOTE that the parameters are from 
    Creminelli et al. with fsky=53%
    '''
    nu_s = 65.e9
    A_s = 2.1e-5  
    alpha_s = -2.6
    beta_s = -2.9
    ell_s = 80.

    nu_d = 353.e9
    A_d = 0.065   
    alpha_d = -2.42
    beta_d = 1.59
    T_d = 19.6
    ell_d = 80.

    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

    x_s = ((h/k)*nu_s)/T_cmb
    W_cmb_s = x_s**2 * np.exp(x_s)/(np.exp(x_s)-1.)**2
    W_s = W_cmb_s/W_cmb * (nu/nu_s)**beta_s
    S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s

    x_d = ((h/k)*nu_d)/T_cmb
    W_cmb_d = x_d**2 * np.exp(x_d)/(np.exp(x_d)-1.)**2
    W_d = W_cmb_d/W_cmb * (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
    D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    if option_unit=='antenna':
        print '++++++++++++++++++++'
        print 'func gen_SDl_Creminelli()'
        print 'antenna temperature'

        S_l = S_l * W_cmb**2
        D_l = D_l * W_cmb**2
#        W_s = (nu/nu_s)**beta_s
#        S_l = W_s**2 * A_s * (ell_in/ell_s)**alpha_s
#        W_d = (nu/nu_d)**(1.+beta_d) * (np.exp(((h/k)*nu_d)/T_d)-1.)/(np.exp(((h/k)*nu)/T_d)-1.)
#        D_l = W_d**2 * A_d * (ell_in/ell_d)**alpha_d

    return ell_in, S_l, D_l
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the basic functions

def antenna2thermo_toDl(Dl,nu):
    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2
    return (Dl/W_cmb**2)

def thermo2antenna_toDl(Dl,nu):
    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2
    return (Dl*W_cmb**2)

def antenna2thermo(T_a,nu):
    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2
    return (T_a/W_cmb)

def thermo2antenna(T_c,nu):
    x = ((h/k)*nu)/T_cmb
    W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2
    return (T_c*W_cmb)

def extract_beta_from_Dl(Dl_d_nu1,Dl_d_nu2,nu1,nu2):
    return 0.5*np.log(Dl_d_nu1/Dl_d_nu2)/np.log(nu1/nu2)

def templatealpha2indexbeta(alpha,nu1,nu2):
    return np.log(alpha)/(np.log(nu1/nu2))

def err_templatealpha2indexbeta(alpha,del_alpha,nu1,nu2):
    return np.sqrt(1./(alpha*np.log(nu1/nu2))**2)*del_alpha

def plot_Cls(ell, Dl_BBr1, Dl_BBlensing, r_input,Dl_nu1,Dl_nu2,lmax):
    py.plot(ell,Dl_BBr1*r_input,label='$C_l^{p}$, $r_{in}='+str(r_input)+'$')
    py.plot(ell,Dl_BBlensing,label='$C_l^{L}$')
    py.plot(ell,Dl_nu1)
    py.plot(ell,Dl_nu2)
    py.loglog()
    py.xlim([2,lmax])
    py.xlabel('$l$')
    py.ylabel('$l(l+1)/2\pi C_l$ [$\mu$K$^2$]')

def Cl2Dl(ell,Cl):
    prefactor = (ell*(ell+1.)/(2.*pi))
    return Cl*prefactor

def Dl2Cl(ell,Dl):
    prefactor = (ell*(ell+1.)/(2.*pi))
    return Dl/prefactor

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the function to compare between template alpha and FG power law index
def CaleffPLI(nu1, nu2, F1, F2, beta):
    # CaleffPLI: compute the expected template alpha assuming 
    #           that the sky signal is single foreground component
    alpha_A = np.sqrt(np.mean(F1/F2))
    alpha_B = (nu1/nu2)**(beta)
    return alpha_A, alpha_B

# Defining the function to compute the ideal alpha value for 2-band 1-component
def Cal_idealtemplatefactor_2band1comp(F1, F2): 
    alpha = (F1/F2)
    return alpha

# Defining the function to compute the ideal alpha value for 3-band 2-component
def Cal_idealtemplatefactor_3band2comp(S1, S2, S3, \
                                        D1, D2, D3):
    alpha1 = (S2*D3-D2*S3)/(S1*D3-D1*S3)
    alpha3 = (S2*D1-D2*S1)/(S3*D1-D3*S1)
    return alpha1, alpha3

# Defining the function to compute the ideal alpha value for 6-band 2-component
def Cal_idealtemplatefactor_6band2comp(S1, S2, S3, S4, S5, S6, \
                                        D1, D2, D3, D4, D5, D6):
    alpha12 = ((S3+S4)*(D6-D5) - (D3+D4)*(S6-S5))/((S1-S2)*(D6-D5) - (D1-D2)*(S6-S5))
    alpha65 = ((S3+S4)*(D1-D2) - (D3+D4)*(S1-S2))/((S6-S5)*(D1-D2) - (D6-D5)*(S1-S2))
    return alpha12, alpha65

def Cal_Clrms(lmax,ell,C_l,sigma_b_in):
    ind = np.where( (ell>=2) & (ell<=lmax) )
    C_l_rms2 = (1./(4.*pi))*np.sum( (2.*ell[ind[0]]+1.) *C_l[ind[0]] * np.exp(-ell[ind[0]]**2*sigma_b_in**2))
    return C_l_rms2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the Foreground plotting functions

def Plot_RMS_Spectrum_incl_FG(r_in,
                              uKarcmin, F_l, FWHM, fsky,
                              dir_cl, filename_prim, filename_lens,
                              dirout, filename_out,
                              option_Lensing, option_unit, option_model):

#   readin CMB Cl fits file
    read_obj = libcl.read_cambdata()
    Cl_lens = read_obj.read_cl_classdata_in_cambformat(dir_cl+'/'+filename_lens)
    Cl_prim  = read_obj.read_cl_classdata_in_cambformat(dir_cl+'/'+filename_prim)

    ell_L = np.array(np.int_(Cl_lens['ell']))
    EEin_L = Cl_lens['EE']
    BBin_L = Cl_lens['BB']

    ell_P = np.array(np.int_(Cl_prim['ell']))
    EEin_P = Cl_prim['EE']
    BBin_P = Cl_prim['BB']

    prefact_g = (ell_L*(ell_L+1.)/(2.*pi))

    C_l_r1 = (BBin_P/prefact_g) 
    C_l_lensing = (BBin_L/prefact_g)
    C_l_ee = (EEin_L/prefact_g)
    C_l = C_l_r1*r_in + C_l_lensing

#   generate the noise spectrum
    ell_in = ell_P
    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')

    if option_Lensing == 'Lensing':
        Cl_hat = r_in*C_l_r1 + C_l_lensing + N_l + F_l/ell_in**2.
    if option_Lensing == 'noLensing':
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1 + N_l + F_l/ell_in**2.

#   compute the RMS 
    sigma_b_in = FWHM/60./180.*pi/np.sqrt(8.*np.log(2.))
    C_l_cmb = r_in*C_l_r1 + C_l_lensing
    C_l_rms2_tmp = (1./(4.*pi))*np.sum( (2.*ell_P+1.) *C_l_cmb * np.exp(-ell_P**2*sigma_b_in**2))

    num_nu = 100
    nu_in = np.arange(num_nu)/float(num_nu)*1000.e9 + 10.e9
    Clee_rms2 = np.zeros(num_nu)
    Clbb_rms2 = np.zeros(num_nu)
    S_l_rms2 = np.zeros(num_nu)
    D_l_rms2 = np.zeros(num_nu)
    for i in range(num_nu):
#        if option_model == 'Creminelli'
        ell_fg, S_l, D_l =  gen_Cl_Creminelli(ell_in,nu_in[i],option_unit=option_unit)
#        if option_model == 'Meisner'
#            ell_fg, S_l, D_l =  gen_Cl_Meisner(ell_in,nu_in[i],option_unit=option_unit)

        x = ((h/k)*nu_in[i])/T_cmb
        W_cmb = x**2 * np.exp(x)/(np.exp(x)-1.)**2

        if option_unit=="antenna":
            Clee_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) * 0.53 *C_l_ee * np.exp(-ell_P**2*sigma_b_in**2)) * W_cmb**2
            Clbb_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) * 0.53 *C_l * np.exp(-ell_P**2*sigma_b_in**2)) * W_cmb**2
        if option_unit=="thermo":
            Clee_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) * 0.53 *C_l_ee * np.exp(-ell_P**2*sigma_b_in**2)) 
            Clbb_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) * 0.53 *C_l * np.exp(-ell_P**2*sigma_b_in**2)) 
        S_l_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) *S_l * np.exp(-ell_P**2*sigma_b_in**2))
        D_l_rms2[i] = (1./(4.*pi))*np.sum( (2.*ell_P+1.) *D_l * np.exp(-ell_P**2*sigma_b_in**2))

#   plot the results!
    py.figure()
    py.subplot(111)
    py.plot(nu_in*1e-9, np.sqrt(Clee_rms2),'g',linewidth=8)
    py.plot(nu_in*1e-9, np.sqrt(Clbb_rms2),'b',linewidth=8)
    py.plot(nu_in*1e-9, np.sqrt(S_l_rms2),'c--',linewidth=6)
    py.plot(nu_in*1e-9, np.sqrt(D_l_rms2),'r--',linewidth=6)
    py.plot(nu_in*1e-9, np.sqrt(S_l_rms2*1e-4),'c',linewidth=8)
    py.plot(nu_in*1e-9, np.sqrt(D_l_rms2*1e-4),'r',linewidth=8)
    py.xlim([10,1000])
    py.ylim([1e-2,1e3])
    py.loglog()
    py.xlabel('Frequency [GHz]', fontsize = 17)
    if option_unit=='thermo': py.ylabel('RMS [$\mu$K$_{CMB}^2$]', fontsize = 17)
    if option_unit=='antenna': py.ylabel('RMS [$\mu$K$_{antenna}^2$]', fontsize = 17)
    py.legend(loc='best',prop={'size':5})
    py.savefig(dirout+'/'+filename_out)
    py.savefig(dirout+'/'+filename_out+'.eps')
    py.clf()

    py.figure()
    py.subplot(111)
#    py.plot(nu_in*1e-9, np.sqrt(Clee_rms2),'g',linewidth=8)
    py.plot(nu_in*1e-9, np.sqrt(Clbb_rms2),'b',linewidth=8)
    py.plot(nu_in*1e-9, np.sqrt(S_l_rms2)/np.sqrt(Clbb_rms2),'c--',linewidth=6)
    py.plot(nu_in*1e-9, np.sqrt(D_l_rms2)/np.sqrt(Clbb_rms2),'r--',linewidth=6)
    py.xlim([10,1000])
    py.ylim([1e-2,1e3])
    py.loglog()
    py.xlabel('Frequency [GHz]', fontsize = 17)
    py.ylabel('RMS ratio w.r.t. $C_l^{BB}$', fontsize = 17)
    py.legend(loc='best',prop={'size':5})
    py.savefig(dirout+'/'+filename_out+'_ratio')
    py.savefig(dirout+'/'+filename_out+'_ratio.eps')
    py.clf()

    return nu_in, Clee_rms2, Clbb_rms2, S_l_rms2, D_l_rms2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the a_lm foreground removal related functions
#

def Band2Comp2_FG(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2, \
        Dl_lensing_nu1,Dl_lensing_nu2, \
        Dl_s1,Dl_s2, \
        Dl_d1,Dl_d2, \
        nu1,nu2, \
        uKarcmin1,uKarcmin2,\
        FWHM1,FWHM2, \
        lmax, num_iter,
        option_unit):

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl_r1_nu1/prefact
    Cl_r1_nu2 = Dl_r1_nu2/prefact
    Cl_lensing_nu1 = (Dl_lensing_nu1/prefact)
    Cl_lensing_nu2 = (Dl_lensing_nu2/prefact)
    Cl_nu1 = Cl_r1_nu1 *r_in + Cl_lensing_nu1
    Cl_nu2 = Cl_r1_nu2 *r_in + Cl_lensing_nu2

#++++++++++++++++++++++++++++++++++++
#
    Cl_s1 = (Dl_s1/prefact)
    Cl_s2 = (Dl_s2/prefact)

#++++++++++++++++++++++++++++++++++++
#
    Cl_d1 = (Dl_d1/prefact)
    Cl_d2 = (Dl_d2/prefact)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        Cl_r1 = Cl_r1_nu1
        Cl_lensing = Cl_lensing_nu1
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu1,nu1)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu1,nu1)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha_mean = np.zeros(lmax-1)
    alpha_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)
    alpha_in = np.zeros(lmax-1)

    for i_ell in range(0,lmax-1):
        alpha_arr = np.zeros(num_iter)
        alpha_arr_in = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            foreg_rand = np.random.normal(0,1.,num_m)
            slm1 = np.sqrt(Cl_s1[i_ell])*foreg_rand
            slm2 = np.sqrt(Cl_s2[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm1 = np.sqrt(Cl_d1[i_ell])*foreg_rand
            dlm2 = np.sqrt(Cl_d2[i_ell])*foreg_rand
                
            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + slm1 + dlm1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + slm2 + dlm2 + nlm2

            alpha_input = np.array([0.,0.]) # dummy values
            if ((slm1[0] == 0.) & (slm1[1] == 0.)):
                alpha_input = Cal_idealtemplatefactor_2band1comp(dlm1, dlm2)
            if ((dlm1[0] == 0.) & (dlm1[1] == 0.)):
                alpha_input = Cal_idealtemplatefactor_2band1comp(slm1, slm2)

            if option_unit == 'thermo':
                alpha_th = np.sum(mlm1*mlm2)/np.sum(mlm2**2)
#                alpha_th = 0.227320512296*antenna2thermo(1.,nu1)/antenna2thermo(1.,nu2)
                del_mlm = mlm1-alpha_th*mlm2
                r = (  1./(1.-alpha_th)**2 * ( np.sum(del_mlm**2) / float(num_m-1) \
                                                - Nl_1[i_ell] - alpha_th**2.*Nl_2[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha_th = np.sum((A1*mlm1)*(A2*mlm2))/np.sum((A2*mlm2)**2)
                del_mlm = A1*mlm1-A2*alpha_th*mlm2
                r = (  1./(1.-alpha_th)**2 * ( np.sum(del_mlm**2) / float(num_m-1) \
                                                - A1**2*Nl_1[i_ell] - A2**2*alpha_th**2.*Nl_2[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            alpha_arr[i_iter] = alpha_th
            r_arr[i_iter] = r
            alpha_arr_in[i_iter] = np.mean(alpha_input)

        ell_out[i_ell] = ell[i_ell]
        alpha_mean[i_ell] = np.mean(alpha_arr)
        alpha_std[i_ell] = np.std(alpha_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)
        alpha_in[i_ell] = np.mean(alpha_arr_in)

    return ell_out, alpha_mean, alpha_std, r_mean, r_std, alpha_in

def Band2Comp2_FG_AHWP(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2, \
        Dl_lensing_nu1,Dl_lensing_nu2, \
        Dl_EE_nu1,Dl_EE_nu2, \
        Dl_EE_s1,Dl_EE_s2, \
        Dl_BB_s1,Dl_BB_s2, \
        Dl_EE_d1,Dl_EE_d2, \
        Dl_BB_d1,Dl_BB_d2, \
        alpha_c1, alpha_s1, alpha_d1, \
        alpha_c2, alpha_s2, alpha_d2, \
        nu1,nu2, \
        uKarcmin1,uKarcmin2,\
        FWHM1,FWHM2, \
        lmax, num_iter,
        option_unit):

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl_r1_nu1/prefact
    Cl_r1_nu2 = Dl_r1_nu2/prefact
    Cl_lensing_nu1 = (Dl_lensing_nu1/prefact)
    Cl_lensing_nu2 = (Dl_lensing_nu2/prefact)
    Cl_BB_nu1 = Cl_r1_nu1 *r_in + Cl_lensing_nu1
    Cl_BB_nu2 = Cl_r1_nu2 *r_in + Cl_lensing_nu2

    Cl_EE_nu1 = Dl_EE_nu1/prefact
    Cl_EE_nu2 = Dl_EE_nu2/prefact

#++++++++++++++++++++++++++++++++++++
#
    Cl_EE_s1 = (Dl_EE_s1/prefact)
    Cl_EE_s2 = (Dl_EE_s2/prefact)

    Cl_EE_d1 = (Dl_EE_d1/prefact)
    Cl_EE_d2 = (Dl_EE_d2/prefact)

    Cl_BB_s1 = (Dl_BB_s1/prefact)
    Cl_BB_s2 = (Dl_BB_s2/prefact)

    Cl_BB_d1 = (Dl_BB_d1/prefact)
    Cl_BB_d2 = (Dl_BB_d2/prefact)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        Cl_r1 = Cl_r1_nu1
        Cl_lensing = Cl_lensing_nu1
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu1,nu1)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu1,nu1)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha_mean = np.zeros(lmax-1)
    alpha_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)
    alpha_in = np.zeros(lmax-1)

    for i_ell in range(0,lmax-1):
        alpha_arr = np.zeros(num_iter)
        alpha_arr_in = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            foreg_rand = np.random.normal(0,1.,num_m)
            slm_ee_1 = np.sqrt(Cl_EE_s1[i_ell])*foreg_rand
            slm_ee_2 = np.sqrt(Cl_EE_s2[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            slm_bb_1 = np.sqrt(Cl_BB_s1[i_ell])*foreg_rand
            slm_bb_2 = np.sqrt(Cl_BB_s2[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm_ee_1 = np.sqrt(Cl_EE_d1[i_ell])*foreg_rand
            dlm_ee_2 = np.sqrt(Cl_EE_d2[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm_bb_1 = np.sqrt(Cl_BB_d1[i_ell])*foreg_rand
            dlm_bb_2 = np.sqrt(Cl_BB_d2[i_ell])*foreg_rand
                
            nlm_ee_1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm_ee_2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)

            nlm_bb_1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm_bb_2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_ee_nu1 = np.sqrt(Cl_EE_nu1[i_ell])*cmbl_rand
            alm_ee_nu2 = np.sqrt(Cl_EE_nu2[i_ell])*cmbl_rand

            mlm_ee_1 = alm_ee_nu1 + slm_ee_1 + dlm_ee_1 + nlm_ee_1
            mlm_ee_2 = alm_ee_nu2 + slm_ee_2 + dlm_ee_2 + nlm_ee_2

            mlm_bb_1 = alm_prim_nu1 + alm_lens_nu1 + slm_bb_1 + dlm_bb_1 + nlm_bb_1
            mlm_bb_2 = alm_prim_nu2 + alm_lens_nu2 + slm_bb_2 + dlm_bb_2 + nlm_bb_2

            mlm1 = alm_ee_nu1*np.sin(2.*alpha_c1) + slm_ee_1*np.sin(2.*alpha_s1) + dlm_ee_1*np.sin(2.*alpha_d1) \
                + alm_prim_nu1*np.cos(2.*alpha_c1) + alm_lens_nu1*np.cos(2.*alpha_c1) + slm_bb_1*np.cos(2.*alpha_s1) + dlm_bb_1*np.cos(2.*alpha_d1) + nlm_bb_1
            mlm2 = alm_ee_nu2*np.sin(2.*alpha_c2) + slm_ee_2*np.sin(2.*alpha_s2) + dlm_ee_2*np.sin(2.*alpha_d2) \
                + alm_prim_nu2*np.cos(2.*alpha_c2) + alm_lens_nu2*np.cos(2.*alpha_c2) + slm_bb_2*np.cos(2.*alpha_s2) + dlm_bb_2*np.cos(2.*alpha_d2) + nlm_bb_2

            alpha_input = np.array([0.,0.]) # dummy values
            if ((slm_ee_1[0] == 0.) & (slm_ee_1[1] == 0.)):
                alpha_input = Cal_idealtemplatefactor_2band1comp(dlm_ee_1, dlm_ee_2)
            if ((dlm_ee_1[0] == 0.) & (dlm_ee_1[1] == 0.)):
                alpha_input = Cal_idealtemplatefactor_2band1comp(slm_ee_1, slm_ee_2)

            if option_unit == 'thermo':
                alpha_th = np.sum(mlm1*mlm2)/np.sum(mlm2**2)
#                alpha_th = 0.227320512296*antenna2thermo(1.,nu1)/antenna2thermo(1.,nu2)
                del_mlm = mlm1-alpha_th*mlm2
                r = (  1./(1.-alpha_th)**2 * ( np.sum(del_mlm**2) / float(num_m-1) \
                                                - Nl_1[i_ell] - alpha_th**2.*Nl_2[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha_th = np.sum((A1*mlm1)*(A2*mlm2))/np.sum((A2*mlm2)**2)
                del_mlm = A1*mlm1-A2*alpha_th*mlm2
                r = (  1./(1.-alpha_th)**2 * ( np.sum(del_mlm**2) / float(num_m-1) \
                                                - A1**2*Nl_1[i_ell] - A2**2*alpha_th**2.*Nl_2[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            alpha_arr[i_iter] = alpha_th
            r_arr[i_iter] = r
            alpha_arr_in[i_iter] = np.mean(alpha_input)

        ell_out[i_ell] = ell[i_ell]
        alpha_mean[i_ell] = np.mean(alpha_arr)
        alpha_std[i_ell] = np.std(alpha_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)
        alpha_in[i_ell] = np.mean(alpha_arr_in)

    return ell_out, alpha_mean, alpha_std, r_mean, r_std, alpha_in

def Band3Comp2_FG(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3, \
        Dl_s1,Dl_s2,Dl_s3, \
        Dl_d1,Dl_d2,Dl_d3, \
        nu1,nu2,nu3, \
        uKarcmin1,uKarcmin2,uKarcmin3,\
        FWHM1,FWHM2,FWHM3, \
        lmax, num_iter,
        option_unit):
    # Note: nu2 is the CMB channel, nu1 is for synch and nu3 is for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl_r1_nu1/prefact
    Cl_r1_nu2 = Dl_r1_nu2/prefact
    Cl_r1_nu3 = Dl_r1_nu3/prefact
    Cl_lensing_nu1 = (Dl_lensing_nu1/prefact)
    Cl_lensing_nu2 = (Dl_lensing_nu2/prefact)
    Cl_lensing_nu3 = (Dl_lensing_nu3/prefact)
    Cl_nu1 = Cl_r1_nu1 *r_in + Cl_lensing_nu1
    Cl_nu2 = Cl_r1_nu2 *r_in + Cl_lensing_nu2
    Cl_nu3 = Cl_r1_nu3 *r_in + Cl_lensing_nu3

#++++++++++++++++++++++++++++++++++++
#
    Cl_s1 = (Dl_s1/prefact)
    Cl_s2 = (Dl_s2/prefact)
    Cl_s3 = (Dl_s3/prefact)

#++++++++++++++++++++++++++++++++++++
#
    Cl_d1 = (Dl_d1/prefact)
    Cl_d2 = (Dl_d2/prefact)
    Cl_d3 = (Dl_d3/prefact)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        Cl_r1 = Cl_r1_nu1
        Cl_lensing = Cl_lensing_nu1
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu2,nu2)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu2,nu2)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha1_mean = np.zeros(lmax-1)
    alpha3_mean = np.zeros(lmax-1)
    alpha1_std = np.zeros(lmax-1)
    alpha3_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)
    alpha_in = np.zeros((2,lmax-1))

    for i_ell in range(0,lmax-1):
        alpha1_arr = np.zeros(num_iter)
        alpha3_arr = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        alpha1_arr_in = np.zeros(num_iter)
        alpha3_arr_in = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            foreg_rand = np.random.normal(0,1.,num_m)
            slm1 = np.sqrt(Cl_s1[i_ell])*foreg_rand
            slm2 = np.sqrt(Cl_s2[i_ell])*foreg_rand
            slm3 = np.sqrt(Cl_s3[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm1 = np.sqrt(Cl_d1[i_ell])*foreg_rand
            dlm2 = np.sqrt(Cl_d2[i_ell])*foreg_rand
            dlm3 = np.sqrt(Cl_d3[i_ell])*foreg_rand
                
            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand
            alm_prim_nu3 = np.sqrt(Cl_r1_nu3[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand
            alm_lens_nu3 = np.sqrt(Cl_lensing_nu3[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + slm1 + dlm1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + slm2 + dlm2 + nlm2
            mlm3 = alm_prim_nu3 + alm_lens_nu3 + slm3 + dlm3 + nlm3

            alpha1_in, alpha3_in = Cal_idealtemplatefactor_3band2comp(slm1, slm2, slm3, dlm1, dlm2, dlm3)

            if option_unit == 'thermo':
                alpha1 = (np.sum(mlm1*mlm2)*np.sum(mlm3*mlm3)-np.sum(mlm2*mlm3)*np.sum(mlm1*mlm3)) \
                    /(np.sum(mlm1*mlm1)*np.sum(mlm3*mlm3)-np.sum(mlm1*mlm3)*np.sum(mlm1*mlm3))
                alpha3 = (np.sum(mlm2*mlm1)*np.sum(mlm1*mlm3)-np.sum(mlm2*mlm3)*np.sum(mlm1*mlm1)) \
                    /(np.sum(mlm1*mlm3)*np.sum(mlm1*mlm3)-np.sum(mlm1*mlm1)*np.sum(mlm3*mlm3))
                del_mlm = mlm2-alpha1*mlm1-alpha3*mlm3
                r = (  1./(1.-alpha1-alpha3)**2 * ( np.sum(del_mlm**2) / float(num_m-2) \
                                                - Nl_2[i_ell] - alpha1**2.*Nl_1[i_ell] - alpha3**2.*Nl_3[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha1 = (np.sum(A1*mlm1*A2*mlm2)*np.sum(A3*mlm3*A3*mlm3)-np.sum(A2*mlm2*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3)) \
                    /(np.sum(A1*mlm1*A1*mlm1)*np.sum(A3*mlm3*A3*mlm3)-np.sum(A1*mlm1*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3))
                alpha3 = (np.sum(A2*mlm2*A1*mlm1)*np.sum(A1*mlm1*A3*mlm3)-np.sum(A2*mlm2*A3*mlm3)*np.sum(A1*mlm1*A1*mlm1)) \
                    /(np.sum(A1*mlm1*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3)-np.sum(A1*mlm1*A1*mlm1)*np.sum(A3*mlm3*A3*mlm3))
                del_mlm = A2*mlm2-A1*alpha1*mlm1-A3*alpha3*mlm3
                r = (  1./(1.-alpha1-alpha3)**2 * ( np.sum(del_mlm**2) / float(num_m-2) \
                                                - A2**2*Nl_2[i_ell] - A1**2*alpha1**2.*Nl_1[i_ell] - A3**2*alpha3**2.*Nl_3[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            alpha1_arr[i_iter] = alpha1
            alpha3_arr[i_iter] = alpha3
            r_arr[i_iter] = r
            alpha1_arr_in[i_iter] = np.mean(alpha1_in)
            alpha3_arr_in[i_iter] = np.mean(alpha3_in)

        ell_out[i_ell] = ell[i_ell]
        alpha1_mean[i_ell] = np.mean(alpha1_arr)
        alpha3_mean[i_ell] = np.mean(alpha3_arr)
        alpha1_std[i_ell] = np.std(alpha1_arr)
        alpha3_std[i_ell] = np.std(alpha3_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)
        alpha_in[0,i_ell] = np.mean(alpha1_arr_in)
        alpha_in[1,i_ell] = np.mean(alpha3_arr_in)

    return ell_out, alpha1_mean, alpha1_std, alpha3_mean, alpha3_std, r_mean, r_std, alpha_in


def Band3Comp2_noFG(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3, \
        nu1,nu2,nu3, \
        uKarcmin1,uKarcmin2,uKarcmin3,\
        FWHM1,FWHM2,FWHM3, \
        lmax, num_iter,
        option_unit):
    # Note: nu2 is the CMB channel, nu1 is for synch and nu3 is for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl_r1_nu1/prefact
    Cl_r1_nu2 = Dl_r1_nu2/prefact
    Cl_r1_nu3 = Dl_r1_nu3/prefact
    Cl_lensing_nu1 = (Dl_lensing_nu1/prefact)
    Cl_lensing_nu2 = (Dl_lensing_nu2/prefact)
    Cl_lensing_nu3 = (Dl_lensing_nu3/prefact)
    Cl_nu1 = Cl_r1_nu1 *r_in + Cl_lensing_nu1
    Cl_nu2 = Cl_r1_nu2 *r_in + Cl_lensing_nu2
    Cl_nu3 = Cl_r1_nu3 *r_in + Cl_lensing_nu3

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        Cl_r1 = Cl_r1_nu1
        Cl_lensing = Cl_lensing_nu1
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu2,nu2)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu2,nu2)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha1_mean = np.zeros(lmax-1)
    alpha3_mean = np.zeros(lmax-1)
    alpha1_std = np.zeros(lmax-1)
    alpha3_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)
    alpha_in = np.zeros((2,lmax-1))

    for i_ell in range(0,lmax-1):
        alpha1_arr = np.zeros(num_iter)
        alpha3_arr = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        alpha1_arr_in = np.zeros(num_iter)
        alpha3_arr_in = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1
                
            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand
            alm_prim_nu3 = np.sqrt(Cl_r1_nu3[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand
            alm_lens_nu3 = np.sqrt(Cl_lensing_nu3[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + nlm2
            mlm3 = alm_prim_nu3 + alm_lens_nu3 + nlm3

            if option_unit == 'thermo':
                alpha1 = (np.sum(mlm1*mlm2)*np.sum(mlm3*mlm3)-np.sum(mlm2*mlm3)*np.sum(mlm1*mlm3)) \
                    /(np.sum(mlm1*mlm1)*np.sum(mlm3*mlm3)-np.sum(mlm1*mlm3)*np.sum(mlm1*mlm3))
                alpha3 = (np.sum(mlm2*mlm1)*np.sum(mlm1*mlm3)-np.sum(mlm2*mlm3)*np.sum(mlm1*mlm1)) \
                    /(np.sum(mlm1*mlm3)*np.sum(mlm1*mlm3)-np.sum(mlm1*mlm1)*np.sum(mlm3*mlm3))
                del_mlm = mlm2-alpha1*mlm1-alpha3*mlm3
                r = (  1./(1.-alpha1-alpha3)**2 * ( np.sum(del_mlm**2) / float(num_m-2) \
                                                - Nl_2[i_ell] - alpha1**2.*Nl_1[i_ell] - alpha3**2.*Nl_3[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha1 = (np.sum(A1*mlm1*A2*mlm2)*np.sum(A3*mlm3*A3*mlm3)-np.sum(A2*mlm2*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3)) \
                    /(np.sum(A1*mlm1*A1*mlm1)*np.sum(A3*mlm3*A3*mlm3)-np.sum(A1*mlm1*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3))
                alpha3 = (np.sum(A2*mlm2*A1*mlm1)*np.sum(A1*mlm1*A3*mlm3)-np.sum(A2*mlm2*A3*mlm3)*np.sum(A1*mlm1*A1*mlm1)) \
                    /(np.sum(A1*mlm1*A3*mlm3)*np.sum(A1*mlm1*A3*mlm3)-np.sum(A1*mlm1*A1*mlm1)*np.sum(A3*mlm3*A3*mlm3))
                del_mlm = A2*mlm2-A1*alpha1*mlm1-A3*alpha3*mlm3
                r = (  1./(1.-alpha1-alpha3)**2 * ( np.sum(del_mlm**2) / float(num_m-2) \
                                                - A2**2*Nl_2[i_ell] - A1**2*alpha1**2.*Nl_1[i_ell] - A3**2*alpha3**2.*Nl_3[i_ell]) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            alpha1_arr[i_iter] = alpha1
            alpha3_arr[i_iter] = alpha3
            r_arr[i_iter] = r

        ell_out[i_ell] = ell[i_ell]
        alpha1_mean[i_ell] = np.mean(alpha1_arr)
        alpha3_mean[i_ell] = np.mean(alpha3_arr)
        alpha1_std[i_ell] = np.std(alpha1_arr)
        alpha3_std[i_ell] = np.std(alpha3_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)

    return ell_out, alpha1_mean, alpha1_std, alpha3_mean, alpha3_std, r_mean, r_std


def Band6Comp2_FG(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3,Dl_r1_nu4,Dl_r1_nu5,Dl_r1_nu6, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3,Dl_lensing_nu4,Dl_lensing_nu5,Dl_lensing_nu6, \
        Dl_s1,Dl_s2,Dl_s3,Dl_s4,Dl_s5,Dl_s6, \
        Dl_d1,Dl_d2,Dl_d3,Dl_d4,Dl_d5,Dl_d6, \
        nu1,nu2,nu3,nu4,nu5,nu6, \
        uKarcmin1,uKarcmin2,uKarcmin3,uKarcmin4,uKarcmin5,uKarcmin6,\
        FWHM1,FWHM2,FWHM3,FWHM4,FWHM5,FWHM6, \
        lmax, num_iter,
        option_unit):
    # Note: nu3,4 are the CMB channel, nu1,2 are for synch and nu5,6 are for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl2Cl(ell,Dl_r1_nu1)
    Cl_r1_nu2 = Dl2Cl(ell,Dl_r1_nu2)
    Cl_r1_nu3 = Dl2Cl(ell,Dl_r1_nu3)
    Cl_r1_nu4 = Dl2Cl(ell,Dl_r1_nu4)
    Cl_r1_nu5 = Dl2Cl(ell,Dl_r1_nu5)
    Cl_r1_nu6 = Dl2Cl(ell,Dl_r1_nu6)
    Cl_lensing_nu1 = Dl2Cl(ell,Dl_lensing_nu1)
    Cl_lensing_nu2 = Dl2Cl(ell,Dl_lensing_nu2)
    Cl_lensing_nu3 = Dl2Cl(ell,Dl_lensing_nu3)
    Cl_lensing_nu4 = Dl2Cl(ell,Dl_lensing_nu4)
    Cl_lensing_nu5 = Dl2Cl(ell,Dl_lensing_nu5)
    Cl_lensing_nu6 = Dl2Cl(ell,Dl_lensing_nu6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_s1 = Dl2Cl(ell,Dl_s1)
    Cl_s2 = Dl2Cl(ell,Dl_s2)
    Cl_s3 = Dl2Cl(ell,Dl_s3)
    Cl_s4 = Dl2Cl(ell,Dl_s4)
    Cl_s5 = Dl2Cl(ell,Dl_s5)
    Cl_s6 = Dl2Cl(ell,Dl_s6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_d1 = Dl2Cl(ell,Dl_d1)
    Cl_d2 = Dl2Cl(ell,Dl_d2)
    Cl_d3 = Dl2Cl(ell,Dl_d3)
    Cl_d4 = Dl2Cl(ell,Dl_d4)
    Cl_d5 = Dl2Cl(ell,Dl_d5)
    Cl_d6 = Dl2Cl(ell,Dl_d6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)
 
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin4
    gen_Nl.FWHM = FWHM4
    gen_Nl.sigma_b()
    Nl_4 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin5
    gen_Nl.FWHM = FWHM5
    gen_Nl.sigma_b()
    Nl_5 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin6
    gen_Nl.FWHM = FWHM6
    gen_Nl.sigma_b()
    Nl_6 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        A4 = 1.
        A5 = 1.
        A6 = 1.
        Cl_r1 = Cl_r1_nu3
        Cl_lensing = Cl_lensing_nu3 
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        A4 = antenna2thermo(1.,nu4)
        A5 = antenna2thermo(1.,nu5)
        A6 = antenna2thermo(1.,nu6)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu3,nu3)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu3,nu3)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha12_mean = np.zeros(lmax-1)
    alpha65_mean = np.zeros(lmax-1)
    alpha12_std = np.zeros(lmax-1)
    alpha65_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)

    alpha_in = np.zeros((2,lmax-1))
 
    for i_ell in range(0,lmax-1):
        alpha12_arr = np.zeros(num_iter)
        alpha65_arr = np.zeros(num_iter)
        alpha12in_arr = np.zeros(num_iter)
        alpha65in_arr = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            foreg_rand = np.random.normal(0,1.,num_m)
            slm1 = np.sqrt(Cl_s1[i_ell])*foreg_rand
            slm2 = np.sqrt(Cl_s2[i_ell])*foreg_rand
            slm3 = np.sqrt(Cl_s3[i_ell])*foreg_rand
            slm4 = np.sqrt(Cl_s4[i_ell])*foreg_rand
            slm5 = np.sqrt(Cl_s5[i_ell])*foreg_rand
            slm6 = np.sqrt(Cl_s6[i_ell])*foreg_rand

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm1 = np.sqrt(Cl_d1[i_ell])*foreg_rand
            dlm2 = np.sqrt(Cl_d2[i_ell])*foreg_rand
            dlm3 = np.sqrt(Cl_d3[i_ell])*foreg_rand
            dlm4 = np.sqrt(Cl_d4[i_ell])*foreg_rand
            dlm5 = np.sqrt(Cl_d5[i_ell])*foreg_rand
            dlm6 = np.sqrt(Cl_d6[i_ell])*foreg_rand
                
            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)
            nlm4 = np.sqrt(Nl_4[i_ell])*np.random.normal(0,1.,num_m)
            nlm5 = np.sqrt(Nl_5[i_ell])*np.random.normal(0,1.,num_m)
            nlm6 = np.sqrt(Nl_6[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand
            alm_prim_nu3 = np.sqrt(Cl_r1_nu3[i_ell]*r_in)*cmbp_rand
            alm_prim_nu4 = np.sqrt(Cl_r1_nu4[i_ell]*r_in)*cmbp_rand
            alm_prim_nu5 = np.sqrt(Cl_r1_nu5[i_ell]*r_in)*cmbp_rand
            alm_prim_nu6 = np.sqrt(Cl_r1_nu6[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand
            alm_lens_nu3 = np.sqrt(Cl_lensing_nu3[i_ell])*cmbl_rand
            alm_lens_nu4 = np.sqrt(Cl_lensing_nu4[i_ell])*cmbl_rand
            alm_lens_nu5 = np.sqrt(Cl_lensing_nu5[i_ell])*cmbl_rand
            alm_lens_nu6 = np.sqrt(Cl_lensing_nu6[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + slm1 + dlm1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + slm2 + dlm2 + nlm2
            mlm3 = alm_prim_nu3 + alm_lens_nu3 + slm3 + dlm3 + nlm3
            mlm4 = alm_prim_nu4 + alm_lens_nu4 + slm4 + dlm4 + nlm4
            mlm5 = alm_prim_nu5 + alm_lens_nu5 + slm5 + dlm5 + nlm5
            mlm6 = alm_prim_nu6 + alm_lens_nu6 + slm6 + dlm6 + nlm6

            alpha12_in, alpha65_in = Cal_idealtemplatefactor_6band2comp( \
                                                        slm1,slm2,slm3,slm4,slm5,slm6, \
                                                        dlm1,dlm2,dlm3,dlm4,dlm5,dlm6)

            if option_unit == 'thermo':
                alpha12 = ( np.sum((mlm1-mlm2)*(mlm3+mlm4))*np.sum((mlm6-mlm5)**2) \
                            -np.sum((mlm6-mlm5)*(mlm3+mlm4))*np.sum((mlm1-mlm2)*(mlm6-mlm5))) \
                        / ( np.sum((mlm1-mlm2)**2)*np.sum((mlm6-mlm5)**2) \
                            -(np.sum((mlm1-mlm2)*(mlm6-mlm5)))**2 )
                alpha65 = ( np.sum((mlm1-mlm2)*(mlm3+mlm4))*np.sum((mlm1-mlm2)*(mlm6-mlm5)) \
                            -np.sum((mlm6-mlm5)*(mlm3+mlm4))*np.sum((mlm1-mlm2)**2)) \
                        / ( (np.sum((mlm1-mlm2)*(mlm6-mlm5)))**2 \
                            -np.sum((mlm6-mlm5)**2)*np.sum((mlm1-mlm2)**2))
                del_mlm = mlm3+mlm4-alpha12*(mlm1-mlm2)-alpha65*(mlm6-mlm5)
                r = (  (1./4. * np.sum(np.abs(del_mlm)**2) / float(num_m-2) \
                                                - 1./4.*(Nl_3[i_ell] + Nl_4[i_ell] \
                                                        + alpha12**2.*(Nl_1[i_ell]+Nl_2[i_ell]) \
                                                        + alpha65**2.*(Nl_6[i_ell]+Nl_5[i_ell]) )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha12 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))) \
                        / ( np.sum((A1*mlm1-A2*mlm2)**2)*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -(np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)))**2 )
                alpha65 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)**2)) \
                        / ( np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)**2)*np.sum((A1*mlm1-A2*mlm2)**2))
                del_mlm = A3*mlm3+A4*mlm4-alpha12*(A1*mlm1-A2*mlm2)-alpha65*(A6*mlm6-A5*mlm5)
                r = (  (1./4. * np.sum(del_mlm**2) / float(num_m-2) \
                                                - 1./4.*(A3**2*Nl_3[i_ell] + A4**2*Nl_4[i_ell] \
                                                        + alpha12**2.*(A1**2*Nl_1[i_ell]+A2**2*Nl_2[i_ell]) \
                                                        + alpha65**2.*(A6**2*Nl_6[i_ell]+A5**2*Nl_5[i_ell]) )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]

            alpha12_arr[i_iter] = alpha12
            alpha65_arr[i_iter] = alpha65
            alpha12in_arr[i_iter] = np.mean(alpha12_in)
            alpha65in_arr[i_iter] = np.mean(alpha65_in)
            r_arr[i_iter] = r

        ell_out[i_ell] = ell[i_ell]
        alpha12_mean[i_ell] = np.mean(alpha12_arr)
        alpha65_mean[i_ell] = np.mean(alpha65_arr)
        alpha12_std[i_ell] = np.std(alpha12_arr)
        alpha65_std[i_ell] = np.std(alpha65_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)

        alpha_in[0,i_ell] = np.mean(alpha12in_arr)
        alpha_in[1,i_ell] = np.mean(alpha65in_arr)

    return ell_out, alpha12_mean, alpha12_std, alpha65_mean, alpha65_std, r_mean, r_std, alpha_in

# the following is the case when the synch and dust are 100 % correlated.
def Band6Comp2_FGcorr(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3,Dl_r1_nu4,Dl_r1_nu5,Dl_r1_nu6, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3,Dl_lensing_nu4,Dl_lensing_nu5,Dl_lensing_nu6, \
        Dl_s1,Dl_s2,Dl_s3,Dl_s4,Dl_s5,Dl_s6, \
        Dl_d1,Dl_d2,Dl_d3,Dl_d4,Dl_d5,Dl_d6, \
        nu1,nu2,nu3,nu4,nu5,nu6, \
        uKarcmin1,uKarcmin2,uKarcmin3,uKarcmin4,uKarcmin5,uKarcmin6,\
        FWHM1,FWHM2,FWHM3,FWHM4,FWHM5,FWHM6, \
        lmax, num_iter,
        option_unit):
    # Note: nu3,4 are the CMB channel, nu1,2 are for synch and nu5,6 are for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl2Cl(ell,Dl_r1_nu1)
    Cl_r1_nu2 = Dl2Cl(ell,Dl_r1_nu2)
    Cl_r1_nu3 = Dl2Cl(ell,Dl_r1_nu3)
    Cl_r1_nu4 = Dl2Cl(ell,Dl_r1_nu4)
    Cl_r1_nu5 = Dl2Cl(ell,Dl_r1_nu5)
    Cl_r1_nu6 = Dl2Cl(ell,Dl_r1_nu6)
    Cl_lensing_nu1 = Dl2Cl(ell,Dl_lensing_nu1)
    Cl_lensing_nu2 = Dl2Cl(ell,Dl_lensing_nu2)
    Cl_lensing_nu3 = Dl2Cl(ell,Dl_lensing_nu3)
    Cl_lensing_nu4 = Dl2Cl(ell,Dl_lensing_nu4)
    Cl_lensing_nu5 = Dl2Cl(ell,Dl_lensing_nu5)
    Cl_lensing_nu6 = Dl2Cl(ell,Dl_lensing_nu6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_s1 = Dl2Cl(ell,Dl_s1)
    Cl_s2 = Dl2Cl(ell,Dl_s2)
    Cl_s3 = Dl2Cl(ell,Dl_s3)
    Cl_s4 = Dl2Cl(ell,Dl_s4)
    Cl_s5 = Dl2Cl(ell,Dl_s5)
    Cl_s6 = Dl2Cl(ell,Dl_s6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_d1 = Dl2Cl(ell,Dl_d1)
    Cl_d2 = Dl2Cl(ell,Dl_d2)
    Cl_d3 = Dl2Cl(ell,Dl_d3)
    Cl_d4 = Dl2Cl(ell,Dl_d4)
    Cl_d5 = Dl2Cl(ell,Dl_d5)
    Cl_d6 = Dl2Cl(ell,Dl_d6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)
 
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin4
    gen_Nl.FWHM = FWHM4
    gen_Nl.sigma_b()
    Nl_4 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin5
    gen_Nl.FWHM = FWHM5
    gen_Nl.sigma_b()
    Nl_5 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin6
    gen_Nl.FWHM = FWHM6
    gen_Nl.sigma_b()
    Nl_6 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        A4 = 1.
        A5 = 1.
        A6 = 1.
        Cl_r1 = Cl_r1_nu3
        Cl_lensing = Cl_lensing_nu3 
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        A4 = antenna2thermo(1.,nu4)
        A5 = antenna2thermo(1.,nu5)
        A6 = antenna2thermo(1.,nu6)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu3,nu3)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu3,nu3)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha12_mean = np.zeros(lmax-1)
    alpha65_mean = np.zeros(lmax-1)
    alpha12_std = np.zeros(lmax-1)
    alpha65_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)

    alpha_in = np.zeros((2,lmax-1))
 
    for i_ell in range(0,lmax-1):
        alpha12_arr = np.zeros(num_iter)
        alpha65_arr = np.zeros(num_iter)
        alpha12in_arr = np.zeros(num_iter)
        alpha65in_arr = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            foreg_rand = np.random.normal(0,1.,num_m)
            slm1 = np.sqrt(Cl_s1[i_ell])*foreg_rand
            slm2 = np.sqrt(Cl_s2[i_ell])*foreg_rand
            slm3 = np.sqrt(Cl_s3[i_ell])*foreg_rand
            slm4 = np.sqrt(Cl_s4[i_ell])*foreg_rand
            slm5 = np.sqrt(Cl_s5[i_ell])*foreg_rand
            slm6 = np.sqrt(Cl_s6[i_ell])*foreg_rand

#            foreg_rand = np.random.normal(0,1.,num_m)
            dlm1 = np.sqrt(Cl_d1[i_ell])*foreg_rand
            dlm2 = np.sqrt(Cl_d2[i_ell])*foreg_rand
            dlm3 = np.sqrt(Cl_d3[i_ell])*foreg_rand
            dlm4 = np.sqrt(Cl_d4[i_ell])*foreg_rand
            dlm5 = np.sqrt(Cl_d5[i_ell])*foreg_rand
            dlm6 = np.sqrt(Cl_d6[i_ell])*foreg_rand
                
            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)
            nlm4 = np.sqrt(Nl_4[i_ell])*np.random.normal(0,1.,num_m)
            nlm5 = np.sqrt(Nl_5[i_ell])*np.random.normal(0,1.,num_m)
            nlm6 = np.sqrt(Nl_6[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand
            alm_prim_nu3 = np.sqrt(Cl_r1_nu3[i_ell]*r_in)*cmbp_rand
            alm_prim_nu4 = np.sqrt(Cl_r1_nu4[i_ell]*r_in)*cmbp_rand
            alm_prim_nu5 = np.sqrt(Cl_r1_nu5[i_ell]*r_in)*cmbp_rand
            alm_prim_nu6 = np.sqrt(Cl_r1_nu6[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand
            alm_lens_nu3 = np.sqrt(Cl_lensing_nu3[i_ell])*cmbl_rand
            alm_lens_nu4 = np.sqrt(Cl_lensing_nu4[i_ell])*cmbl_rand
            alm_lens_nu5 = np.sqrt(Cl_lensing_nu5[i_ell])*cmbl_rand
            alm_lens_nu6 = np.sqrt(Cl_lensing_nu6[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + slm1 + dlm1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + slm2 + dlm2 + nlm2
            mlm3 = alm_prim_nu3 + alm_lens_nu3 + slm3 + dlm3 + nlm3
            mlm4 = alm_prim_nu4 + alm_lens_nu4 + slm4 + dlm4 + nlm4
            mlm5 = alm_prim_nu5 + alm_lens_nu5 + slm5 + dlm5 + nlm5
            mlm6 = alm_prim_nu6 + alm_lens_nu6 + slm6 + dlm6 + nlm6

            alpha12_in, alpha65_in = Cal_idealtemplatefactor_6band2comp( \
                                                        slm1,slm2,slm3,slm4,slm5,slm6, \
                                                        dlm1,dlm2,dlm3,dlm4,dlm5,dlm6)

            if option_unit == 'thermo':
                alpha12 = ( np.sum((mlm1-mlm2)*(mlm3+mlm4))*np.sum((mlm6-mlm5)**2) \
                            -np.sum((mlm6-mlm5)*(mlm3+mlm4))*np.sum((mlm1-mlm2)*(mlm6-mlm5))) \
                        / ( np.sum((mlm1-mlm2)**2)*np.sum((mlm6-mlm5)**2) \
                            -(np.sum((mlm1-mlm2)*(mlm6-mlm5)))**2 )
                alpha65 = ( np.sum((mlm1-mlm2)*(mlm3+mlm4))*np.sum((mlm1-mlm2)*(mlm6-mlm5)) \
                            -np.sum((mlm6-mlm5)*(mlm3+mlm4))*np.sum((mlm1-mlm2)**2)) \
                        / ( (np.sum((mlm1-mlm2)*(mlm6-mlm5)))**2 \
                            -np.sum((mlm6-mlm5)**2)*np.sum((mlm1-mlm2)**2))
                del_mlm = mlm3+mlm4-alpha12*(mlm1-mlm2)-alpha65*(mlm6-mlm5)
                r = (  (1./4. * np.sum(np.abs(del_mlm)**2) / float(num_m-2) \
                                                - 1./4.*(Nl_3[i_ell] + Nl_4[i_ell] \
                                                        + alpha12**2.*(Nl_1[i_ell]+Nl_2[i_ell]) \
                                                        + alpha65**2.*(Nl_6[i_ell]+Nl_5[i_ell]) )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha12 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))) \
                        / ( np.sum((A1*mlm1-A2*mlm2)**2)*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -(np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)))**2 )
                alpha65 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)**2)) \
                        / ( np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)**2)*np.sum((A1*mlm1-A2*mlm2)**2))
                del_mlm = A3*mlm3+A4*mlm4-alpha12*(A1*mlm1-A2*mlm2)-alpha65*(A6*mlm6-A5*mlm5)
                r = (  (1./4. * np.sum(del_mlm**2) / float(num_m-2) \
                                                - 1./4.*(A3**2*Nl_3[i_ell] + A4**2*Nl_4[i_ell] \
                                                        + alpha12**2.*(A1**2*Nl_1[i_ell]+A2**2*Nl_2[i_ell]) \
                                                        + alpha65**2.*(A6**2*Nl_6[i_ell]+A5**2*Nl_5[i_ell]) )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]

            alpha12_arr[i_iter] = alpha12
            alpha65_arr[i_iter] = alpha65
            alpha12in_arr[i_iter] = np.mean(alpha12_in)
            alpha65in_arr[i_iter] = np.mean(alpha65_in)
            r_arr[i_iter] = r

        ell_out[i_ell] = ell[i_ell]
        alpha12_mean[i_ell] = np.mean(alpha12_arr)
        alpha65_mean[i_ell] = np.mean(alpha65_arr)
        alpha12_std[i_ell] = np.std(alpha12_arr)
        alpha65_std[i_ell] = np.std(alpha65_arr)
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)

        alpha_in[0,i_ell] = np.mean(alpha12in_arr)
        alpha_in[1,i_ell] = np.mean(alpha65in_arr)

    return ell_out, alpha12_mean, alpha12_std, alpha65_mean, alpha65_std, r_mean, r_std, alpha_in

def Band6_noFG(ell,r_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3,Dl_r1_nu4,Dl_r1_nu5,Dl_r1_nu6, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3,Dl_lensing_nu4,Dl_lensing_nu5,Dl_lensing_nu6, \
        nu1,nu2,nu3,nu4,nu5,nu6, \
        uKarcmin1,uKarcmin2,uKarcmin3,uKarcmin4,uKarcmin5,uKarcmin6,\
        FWHM1,FWHM2,FWHM3,FWHM4,FWHM5,FWHM6, \
        lmax, num_iter,
        option_unit):
    # Note: nu3,4 are the CMB channel, nu1,2 are for synch and nu5,6 are for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl2Cl(ell,Dl_r1_nu1)
    Cl_r1_nu2 = Dl2Cl(ell,Dl_r1_nu2)
    Cl_r1_nu3 = Dl2Cl(ell,Dl_r1_nu3)
    Cl_r1_nu4 = Dl2Cl(ell,Dl_r1_nu4)
    Cl_r1_nu5 = Dl2Cl(ell,Dl_r1_nu5)
    Cl_r1_nu6 = Dl2Cl(ell,Dl_r1_nu6)
    Cl_lensing_nu1 = Dl2Cl(ell,Dl_lensing_nu1)
    Cl_lensing_nu2 = Dl2Cl(ell,Dl_lensing_nu2)
    Cl_lensing_nu3 = Dl2Cl(ell,Dl_lensing_nu3)
    Cl_lensing_nu4 = Dl2Cl(ell,Dl_lensing_nu4)
    Cl_lensing_nu5 = Dl2Cl(ell,Dl_lensing_nu5)
    Cl_lensing_nu6 = Dl2Cl(ell,Dl_lensing_nu6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)
 
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin4
    gen_Nl.FWHM = FWHM4
    gen_Nl.sigma_b()
    Nl_4 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin5
    gen_Nl.FWHM = FWHM5
    gen_Nl.sigma_b()
    Nl_5 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin6
    gen_Nl.FWHM = FWHM6
    gen_Nl.sigma_b()
    Nl_6 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        A4 = 1.
        A5 = 1.
        A6 = 1.
        Cl_r1 = Cl_r1_nu3
        Cl_lensing = Cl_lensing_nu3 
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        A4 = antenna2thermo(1.,nu4)
        A5 = antenna2thermo(1.,nu5)
        A6 = antenna2thermo(1.,nu6)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu3,nu3)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu3,nu3)

#++++++++++++++++++++++++++++++++++++
#
    ell_out = np.zeros(lmax-1)
    alpha12_mean = np.zeros(lmax-1)
    alpha65_mean = np.zeros(lmax-1)
    alpha12_std = np.zeros(lmax-1)
    alpha65_std = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)

    alpha_in = np.zeros((2,lmax-1))
 
    for i_ell in range(0,lmax-1):
        alpha12_arr = np.zeros(num_iter)
        alpha65_arr = np.zeros(num_iter)
        alpha12in_arr = np.zeros(num_iter)
        alpha65in_arr = np.zeros(num_iter)
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = 2*ell[i_ell]+1

            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)
            nlm4 = np.sqrt(Nl_4[i_ell])*np.random.normal(0,1.,num_m)
            nlm5 = np.sqrt(Nl_5[i_ell])*np.random.normal(0,1.,num_m)
            nlm6 = np.sqrt(Nl_6[i_ell])*np.random.normal(0,1.,num_m)

            cmbp_rand = np.random.normal(0,1.,num_m)
            alm_prim_nu1 = np.sqrt(Cl_r1_nu1[i_ell]*r_in)*cmbp_rand
            alm_prim_nu2 = np.sqrt(Cl_r1_nu2[i_ell]*r_in)*cmbp_rand
            alm_prim_nu3 = np.sqrt(Cl_r1_nu3[i_ell]*r_in)*cmbp_rand
            alm_prim_nu4 = np.sqrt(Cl_r1_nu4[i_ell]*r_in)*cmbp_rand
            alm_prim_nu5 = np.sqrt(Cl_r1_nu5[i_ell]*r_in)*cmbp_rand
            alm_prim_nu6 = np.sqrt(Cl_r1_nu6[i_ell]*r_in)*cmbp_rand

            cmbl_rand = np.random.normal(0,1.,num_m)
            alm_lens_nu1 = np.sqrt(Cl_lensing_nu1[i_ell])*cmbl_rand
            alm_lens_nu2 = np.sqrt(Cl_lensing_nu2[i_ell])*cmbl_rand
            alm_lens_nu3 = np.sqrt(Cl_lensing_nu3[i_ell])*cmbl_rand
            alm_lens_nu4 = np.sqrt(Cl_lensing_nu4[i_ell])*cmbl_rand
            alm_lens_nu5 = np.sqrt(Cl_lensing_nu5[i_ell])*cmbl_rand
            alm_lens_nu6 = np.sqrt(Cl_lensing_nu6[i_ell])*cmbl_rand

            mlm1 = alm_prim_nu1 + alm_lens_nu1 + nlm1
            mlm2 = alm_prim_nu2 + alm_lens_nu2 + nlm2
            mlm3 = alm_prim_nu3 + alm_lens_nu3 + nlm3
            mlm4 = alm_prim_nu4 + alm_lens_nu4 + nlm4
            mlm5 = alm_prim_nu5 + alm_lens_nu5 + nlm5
            mlm6 = alm_prim_nu6 + alm_lens_nu6 + nlm6

            if option_unit == 'thermo':
                del_mlm = mlm3+mlm4
                r = (  (1./4. * np.sum(np.abs(del_mlm)**2) / float(num_m) \
                                                - 1./4.*(Nl_3[i_ell] + Nl_4[i_ell] )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]
            if option_unit == 'antenna':
                alpha12 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))) \
                        / ( np.sum((A1*mlm1-A2*mlm2)**2)*np.sum((A6*mlm6-A5*mlm5)**2) \
                            -(np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)))**2 )
                alpha65 = ( np.sum((A1*mlm1-A2*mlm2)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)*(A3*mlm3+A4*mlm4))*np.sum((A1*mlm1-A2*mlm2)**2)) \
                        / ( np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5))*np.sum((A1*mlm1-A2*mlm2)*(A6*mlm6-A5*mlm5)) \
                            -np.sum((A6*mlm6-A5*mlm5)**2)*np.sum((A1*mlm1-A2*mlm2)**2))
                del_mlm = A3*mlm3+A4*mlm4
                r = (  (1./4. * np.sum(del_mlm**2) / float(num_m) \
                                                - 1./4.*(A3**2*Nl_3[i_ell] + A4**2*Nl_4[i_ell] )) - Cl_lensing[i_ell]) / Cl_r1[i_ell]

            r_arr[i_iter] = r

        ell_out[i_ell] = ell[i_ell]
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)

    return ell_out, r_mean, r_std


def Plot_Band6Comp2_likelihood(ell,r_in,r_var_in,alpha12_in,alpha65_in,\
        Dl_r1_nu1,Dl_r1_nu2,Dl_r1_nu3,Dl_r1_nu4,Dl_r1_nu5,Dl_r1_nu6, \
        Dl_lensing_nu1,Dl_lensing_nu2,Dl_lensing_nu3,Dl_lensing_nu4,Dl_lensing_nu5,Dl_lensing_nu6, \
        Dl_s1,Dl_s2,Dl_s3,Dl_s4,Dl_s5,Dl_s6, \
        Dl_d1,Dl_d2,Dl_d3,Dl_d4,Dl_d5,Dl_d6, \
        nu1,nu2,nu3,nu4,nu5,nu6, \
        uKarcmin1,uKarcmin2,uKarcmin3,uKarcmin4,uKarcmin5,uKarcmin6,\
        FWHM1,FWHM2,FWHM3,FWHM4,FWHM5,FWHM6, \
        lmax,option_unit,option_variable):
    # Note: nu3,4 are the CMB channel, nu1,2 are for synch and nu5,6 are for dust

#++++++++++++++++++++++++++++++++++++
#
    num_ell = len(ell)
    prefact = (ell*(ell+1.)/(2.*pi))

#++++++++++++++++++++++++++++++++++++
#
    Cl_r1_nu1 = Dl2Cl(ell,Dl_r1_nu1)
    Cl_r1_nu2 = Dl2Cl(ell,Dl_r1_nu2)
    Cl_r1_nu3 = Dl2Cl(ell,Dl_r1_nu3)
    Cl_r1_nu4 = Dl2Cl(ell,Dl_r1_nu4)
    Cl_r1_nu5 = Dl2Cl(ell,Dl_r1_nu5)
    Cl_r1_nu6 = Dl2Cl(ell,Dl_r1_nu6)
    Cl_lensing_nu1 = Dl2Cl(ell,Dl_lensing_nu1)
    Cl_lensing_nu2 = Dl2Cl(ell,Dl_lensing_nu2)
    Cl_lensing_nu3 = Dl2Cl(ell,Dl_lensing_nu3)
    Cl_lensing_nu4 = Dl2Cl(ell,Dl_lensing_nu4)
    Cl_lensing_nu5 = Dl2Cl(ell,Dl_lensing_nu5)
    Cl_lensing_nu6 = Dl2Cl(ell,Dl_lensing_nu6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_s1 = Dl2Cl(ell,Dl_s1)
    Cl_s2 = Dl2Cl(ell,Dl_s2)
    Cl_s3 = Dl2Cl(ell,Dl_s3)
    Cl_s4 = Dl2Cl(ell,Dl_s4)
    Cl_s5 = Dl2Cl(ell,Dl_s5)
    Cl_s6 = Dl2Cl(ell,Dl_s6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_d1 = Dl2Cl(ell,Dl_d1)
    Cl_d2 = Dl2Cl(ell,Dl_d2)
    Cl_d3 = Dl2Cl(ell,Dl_d3)
    Cl_d4 = Dl2Cl(ell,Dl_d4)
    Cl_d5 = Dl2Cl(ell,Dl_d5)
    Cl_d6 = Dl2Cl(ell,Dl_d6)

#++++++++++++++++++++++++++++++++++++
#
    Cl_zero = np.zeros(num_ell)
 
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin1
    gen_Nl.FWHM = FWHM1
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')
            
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin2
    gen_Nl.FWHM = FWHM2
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin3
    gen_Nl.FWHM = FWHM3
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin4
    gen_Nl.FWHM = FWHM4
    gen_Nl.sigma_b()
    Nl_4 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin5
    gen_Nl.FWHM = FWHM5
    gen_Nl.sigma_b()
    Nl_5 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = 1.
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin6
    gen_Nl.FWHM = FWHM6
    gen_Nl.sigma_b()
    Nl_6 = gen_Nl.gen_KnoxdNl('noCV')

#++++++++++++++++++++++++++++++++++++
#
    if option_unit == 'thermo':
        A1 = 1.
        A2 = 1.
        A3 = 1.
        A4 = 1.
        A5 = 1.
        A6 = 1.
        Cl_r1 = Cl_r1_nu3
        Cl_lensing = Cl_lensing_nu3 
    if option_unit == 'antenna':
        A1 = antenna2thermo(1.,nu1)
        A2 = antenna2thermo(1.,nu2)
        A3 = antenna2thermo(1.,nu3)
        A4 = antenna2thermo(1.,nu4)
        A5 = antenna2thermo(1.,nu5)
        A6 = antenna2thermo(1.,nu6)
        Cl_r1 = antenna2thermo_toDl(Cl_r1_nu3,nu3)
        Cl_lensing = antenna2thermo_toDl(Cl_lensing_nu3,nu3)

#++++++++++++++++++++++++++++++++++++
#
    L = np.zeros(lmax-1)
    Cl_til = np.zeros(lmax-1)
    Cl_hat = np.zeros(lmax-1)
    alpha_in = np.zeros((2,lmax-1))
    L_mult = 1.

    for i_ell in range(0,lmax-1):

        num_m = 2*ell[i_ell]+1

        alpha12_in_ideal, alpha65_in_ideal = Cal_idealtemplatefactor_6band2comp( \
                                                    np.sqrt(Cl_s1[i_ell]), \
                                                    np.sqrt(Cl_s2[i_ell]), \
                                                    np.sqrt(Cl_s3[i_ell]), \
                                                    np.sqrt(Cl_s4[i_ell]), \
                                                    np.sqrt(Cl_s5[i_ell]), \
                                                    np.sqrt(Cl_s6[i_ell]), \
                                                    np.sqrt(Cl_d1[i_ell]), \
                                                    np.sqrt(Cl_d2[i_ell]), \
                                                    np.sqrt(Cl_d3[i_ell]), \
                                                    np.sqrt(Cl_d4[i_ell]), \
                                                    np.sqrt(Cl_d5[i_ell]), \
                                                    np.sqrt(Cl_d6[i_ell]))

        if (option_variable == 'r'): 
            alpha12_in = alpha12_in_ideal
            alpha65_in = alpha65_in_ideal

        if (option_variable == 'alpha12'): 
            r_var_in = r_in
            alpha65_in = alpha65_in_ideal
        if (option_variable == 'alpha65'): 
            r_var_in = r_in
            alpha12_in = alpha12_in_ideal

        Cl_hat[i_ell] = 4.*( Cl_r1[i_ell]*r_var_in + Cl_lensing[i_ell] ) \
                + Nl_3[i_ell] + Nl_4[i_ell] \
                + np.mean(alpha12_in)**2*( Nl_1[i_ell] + Nl_2[i_ell] ) \
                + np.mean(alpha65_in)**2*( Nl_6[i_ell] + Nl_5[i_ell] ) \
                + (np.sqrt(Cl_s3[i_ell]) + np.sqrt(Cl_s4[i_ell]) \
                - np.mean(alpha12_in)*( np.sqrt(Cl_s1[i_ell])-np.sqrt(Cl_s2[i_ell]) ) \
                - np.mean(alpha65_in)*( np.sqrt(Cl_s6[i_ell])-np.sqrt(Cl_s5[i_ell]) ) )**2\
                + (np.sqrt(Cl_d3[i_ell]) + np.sqrt(Cl_d4[i_ell]) \
                - np.mean(alpha12_in)*( np.sqrt(Cl_d1[i_ell])-np.sqrt(Cl_d2[i_ell]) ) \
                - np.mean(alpha65_in)*( np.sqrt(Cl_d6[i_ell])-np.sqrt(Cl_d5[i_ell]) ) )**2 

        Cl_til[i_ell] = 4.*( Cl_r1[i_ell]*r_in + Cl_lensing[i_ell] ) \
                + Nl_3[i_ell] + Nl_4[i_ell] \
                + np.mean(alpha12_in)**2*(Nl_1[i_ell]+Nl_2[i_ell]) \
                + np.mean(alpha65_in)**2*(Nl_6[i_ell]+Nl_5[i_ell] )

        L[i_ell] = 1./np.sqrt(Cl_hat[i_ell]) * np.exp(- 0.5*Cl_til[i_ell]/Cl_hat[i_ell] )
        L_mult *= L[i_ell]

    return L_mult

def Band6Comp2_indexin( beta_s_in, beta_d_in,
                    ell,Dl_r1,Dl_lensing,r_in, \
                    fsky_det,
                    noise_par, 
                    lmax, num_iter):

    prefact_g = (ell*(ell+1.)/2./pi)
    Cl_r1 = (Dl_r1/prefact_g) 
    Cl_lensing = (Dl_lensing/prefact_g)
    Cl = Cl_r1*r_in + Cl_lensing
    
    tmp, Cl_s1, Cl_d1 = gen_Cl_Creminelli(ell,noise_par['nu1'])
    tmp, Cl_s2, Cl_d2 = gen_Cl_Creminelli(ell,noise_par['nu2'])
    tmp, Cl_s3, Cl_d3 = gen_Cl_Creminelli(ell,noise_par['nu3'])
    tmp, Cl_s4, Cl_d4 = gen_Cl_Creminelli(ell,noise_par['nu4'])
    tmp, Cl_s5, Cl_d5 = gen_Cl_Creminelli(ell,noise_par['nu5'])
    tmp, Cl_s6, Cl_d6 = gen_Cl_Creminelli(ell,noise_par['nu6'])
 

    print ''
    print ''
    print 'in Band6Comp2_indexin'
    print beta_s_in, beta_d_in
    beta_s_in = extract_beta_from_Dl(Cl_s1,Cl_s2,noise_par['nu1'],noise_par['nu2'])
    beta_d_in = extract_beta_from_Dl(Cl_d6,Cl_d5,noise_par['nu6'],noise_par['nu5'])
    print ''
    beta_s = np.mean(beta_s_in)
    beta_d = np.mean(beta_d_in)
    print '$\\beta_s$, $\\beta_d$', beta_s, beta_d
#    beta_s = beta_s_in*2.*0.5  # Cl ~ nu^(2*beta) -> alm space
#    beta_d = beta_d_in*2.*0.5
    A = ( (noise_par['nu3']**beta_s + noise_par['nu4']**beta_s) \
              *(noise_par['nu6']**beta_d - noise_par['nu5']**beta_d) \
              - (noise_par['nu3']**beta_d + noise_par['nu4']**beta_d) \
              *(noise_par['nu6']**beta_s - noise_par['nu5']**beta_s) )  \
              / ( (noise_par['nu1']**beta_s - noise_par['nu2']**beta_s) \
                      *(noise_par['nu6']**beta_d - noise_par['nu5']**beta_d) \
                      - (noise_par['nu1']**beta_d - noise_par['nu2']**beta_d) \
                      *(noise_par['nu6']**beta_s - noise_par['nu5']**beta_s) )
    
    B = ( (noise_par['nu3']**beta_s + noise_par['nu4']**beta_s) \
              *(noise_par['nu1']**beta_d - noise_par['nu2']**beta_d) \
              - (noise_par['nu3']**beta_d + noise_par['nu4']**beta_d) \
              *(noise_par['nu1']**beta_s - noise_par['nu2']**beta_s) )  \
              / ( (noise_par['nu6']**beta_s - noise_par['nu5']**beta_s) \
                      *(noise_par['nu1']**beta_d - noise_par['nu2']**beta_d) \
                      - (noise_par['nu6']**beta_d - noise_par['nu5']**beta_d) \
                      *(noise_par['nu1']**beta_s - noise_par['nu2']**beta_s) )

    Cl_zero = np.zeros(len(ell))

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin1']
    gen_Nl.FWHM = noise_par['FWHM1']
    gen_Nl.sigma_b()
    Nl_1 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin2']
    gen_Nl.FWHM = noise_par['FWHM2']
    gen_Nl.sigma_b()
    Nl_2 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin3']
    gen_Nl.FWHM = noise_par['FWHM3']
    gen_Nl.sigma_b()
    Nl_3 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin4']
    gen_Nl.FWHM = noise_par['FWHM4']
    gen_Nl.sigma_b()
    Nl_4 = gen_Nl.gen_KnoxdNl('noCV')

    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin5']
    gen_Nl.FWHM = noise_par['FWHM5']
    gen_Nl.sigma_b()
    Nl_5 = gen_Nl.gen_KnoxdNl('noCV')  
    
    gen_Nl = libcl.gen_Nl(ell)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = Cl_zero
    gen_Nl.fsky = fsky_det
    gen_Nl.prefact_option(False)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = noise_par['uKarcmin6']
    gen_Nl.FWHM = noise_par['FWHM6']
    gen_Nl.sigma_b()
    Nl_6 = gen_Nl.gen_KnoxdNl('noCV')    

    ell_out = np.zeros(lmax-1)
    r_mean = np.zeros(lmax-1)
    r_std = np.zeros(lmax-1)
    for i_ell in range(0,lmax-1):
        r_arr = np.zeros(num_iter)
        for i_iter in range(0,num_iter):
            num_m = (2*ell[i_ell]+1)

            alm_prim = np.sqrt(Cl_r1[i_ell]*r_in)*np.random.normal(0,1.,num_m)
            alm_lens = np.sqrt(Cl_lensing[i_ell])*np.random.normal(0,1.,num_m)

            nlm1 = np.sqrt(Nl_1[i_ell])*np.random.normal(0,1.,num_m)
            nlm2 = np.sqrt(Nl_2[i_ell])*np.random.normal(0,1.,num_m)
            nlm3 = np.sqrt(Nl_3[i_ell])*np.random.normal(0,1.,num_m)
            nlm4 = np.sqrt(Nl_4[i_ell])*np.random.normal(0,1.,num_m)
            nlm5 = np.sqrt(Nl_5[i_ell])*np.random.normal(0,1.,num_m)
            nlm6 = np.sqrt(Nl_6[i_ell])*np.random.normal(0,1.,num_m)

            foreg_rand = np.random.normal(0,1.,num_m)
            dlm1 = np.sqrt(Cl_d1[i_ell])*foreg_rand
            dlm2 = np.sqrt(Cl_d2[i_ell])*foreg_rand
            dlm3 = np.sqrt(Cl_d3[i_ell])*foreg_rand
            dlm4 = np.sqrt(Cl_d4[i_ell])*foreg_rand
            dlm5 = np.sqrt(Cl_d5[i_ell])*foreg_rand
            dlm6 = np.sqrt(Cl_d6[i_ell])*foreg_rand
            
            foreg_rand = np.random.normal(0,1.,num_m)
            slm1 = np.sqrt(Cl_s1[i_ell])*foreg_rand
            slm2 = np.sqrt(Cl_s2[i_ell])*foreg_rand
            slm3 = np.sqrt(Cl_s3[i_ell])*foreg_rand
            slm4 = np.sqrt(Cl_s4[i_ell])*foreg_rand
            slm5 = np.sqrt(Cl_s5[i_ell])*foreg_rand
            slm6 = np.sqrt(Cl_s6[i_ell])*foreg_rand

            A_ = ( (slm3 + slm4)*(dlm6 - dlm5) \
              - (dlm3 + dlm4)*(slm6 - slm5) )  \
              / ( (slm1 - slm2)*(dlm6 - dlm5) \
              - (dlm1 - dlm2)*(slm6 - slm5) )

            B_ = ( (slm3 + slm4)*(dlm1 - dlm2) \
              - (dlm3 + dlm4)*(slm1 - slm2) )  \
              / ( (slm6 - slm5)*(dlm1 - dlm2) \
              - (dlm6 - dlm5)*(slm1 - slm2) )
            
            A_ = np.mean(A_)
            B_ = np.mean(B_)
            print 'estimated A, B->', np.mean(A), np.mean(B)
            print 'theoretical A_, B_->', A_, B_
            print 'ratio, A/A_, B/B_->', A_/np.mean(A), B_/np.mean(B)
            print ''
#            sys.exit()

            mlm1 = alm_prim + alm_lens + nlm1 + dlm1 + slm1
            mlm2 = alm_prim + alm_lens + nlm2 + dlm2 + slm2
            mlm3 = alm_prim + alm_lens + nlm3 + dlm3 + slm3
            mlm4 = alm_prim + alm_lens + nlm4 + dlm4 + slm4
            mlm5 = alm_prim + alm_lens + nlm5 + dlm5 + slm5
            mlm6 = alm_prim + alm_lens + nlm6 + dlm6 + slm6

#            print 'A, B ->', A, B
#            print 2*(alm_prim+alm_lens)
#            print slm3+slm4 - A*(slm1-slm2) - B*(slm6-slm5)
#            print dlm3+dlm4 - A*(dlm1-dlm2) - B*(dlm6-dlm5)
#            print ''
#            sys.exit()

            del_mlm65 = mlm6-mlm5
            del_mlm12 = mlm1-mlm2
            
            r = ( np.sum(( mlm3 + mlm4 \
                               - A*del_mlm12 - B*del_mlm65 )**2) / float(num_m) \
                      - Nl_3[i_ell] - Nl_4[i_ell] \
                      - A**2.*(Nl_2[i_ell]+Nl_1[i_ell]) \
                      - B**2.*(Nl_5[i_ell]+Nl_6[i_ell]) \
                      - 2.*Cl_lensing[i_ell]) / (2.*Cl_r1[i_ell])
            
            r_arr[i_iter] = r
            
#        if i_ell == 100: 
#            if option_arrayout == True: return ell_p, r_arr
        ell_out[i_ell] = ell[i_ell]
        r_mean[i_ell] = np.mean(r_arr)
        r_std[i_ell] = np.std(r_arr)

    return ell_out, r_mean, r_std

