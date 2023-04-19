import numpy as np
import lib_Clmanip as libcl
import global_par as g

option_debug = False

###################################################
def Cal_CMB_likelifood_lmax(r_max_arr, l_max, r_in,
                            uKarcmin, FWHM, fsky,
                            option_Lensing):
    L_mult = 1.
    for j in range(2,l_max+1):
        ell_in = np.array([j])
        L = Cal_CMB_likelifood_perell(ell_in, r_max_arr, r_in, \
                                          uKarcmin, FWHM, fsky, \
                                          option_Lensing)
        L_mult *= L
    return L_mult/max(L_mult)

def Cal_CMB_likelifood_perell(ell_in, r_var, r_in,
                              uKarcmin, FWHM, fsky,
                              option_Lensing):
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ind = np.where(ell_in == ell_P)

    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l[ind[0]]
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')
    
    if option_Lensing == 'Lensing':
        Cl_hat = r_in*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l
        Cl = r_var*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)
    if option_Lensing == 'noLensing':
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1[ind[0]] + N_l
        Cl = r_var*C_l_r1[ind[0]] + N_l
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)

    return L_wL


def Cal_CMB_likelifood_lmax_incl1oL(r_max_arr, l_max, r_in,
                                    uKarcmin, F_l, FWHM, fsky,
                                    option_Lensing):
    L_mult = 1.
    for j in range(2,l_max+1):
        ell_in = np.array([j])
        L = Cal_CMB_likelifood_perell_incl1oL(ell_in, r_max_arr, r_in, \
                                                  uKarcmin, F_l, FWHM, fsky, \
                                                  option_Lensing)
        L_mult *= L
    return L_mult/max(L_mult)

def Cal_CMB_likelifood_perell_incl1oL(ell_in, r_var, r_in,
                                      uKarcmin, F_l, FWHM, fsky,
                                      option_Lensing):
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ind = np.where(ell_in == ell_P)

    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l[ind[0]]
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')

    if option_Lensing == 'Lensing':
        Cl_hat = r_in*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l + F_l/ell_in**2.
        Cl = r_var*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l + F_l/ell_in**2.
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)
    if option_Lensing == 'noLensing':
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1[ind[0]] + N_l + F_l/ell_in**2.
        Cl = r_var*C_l_r1[ind[0]] + N_l + F_l/ell_in**2.
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)

    return L_wL


def Cal_CMB_likelifood_lmax_inclFGDl(r_max_arr, l_max, r_in,
                                    uKarcmin, FWHM, fsky,
                                    option_Lensing, nu_in, Dl_s, Dl_d):
    L_mult = 1.
    for j in range(2,l_max+1):
        ell_in = np.array([j])
        L = Cal_CMB_likelifood_perell_inclFGDl(ell_in, r_max_arr, r_in, \
                                                  uKarcmin, FWHM, fsky, \
                                                  option_Lensing, nu_in, Dl_s, Dl_d)
        L_mult *= L
    return L_mult/max(L_mult)

def Cal_CMB_likelifood_perell_inclFGDl(ell_in, r_var, r_in,
                                      uKarcmin, FWHM, fsky,
                                      option_Lensing, nu_in, Dl_s, Dl_d):
    prefact_g = ell_in * (ell_in + 1.) / (2.*pi)
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ind = np.where(ell_in == ell_P)

    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l[ind[0]]
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')

    if option_prefact==False: 
        Cl_s = Dl_s/prefact_g
        Cl_d = Dl_d/prefact_g

    if option_Lensing == 'Lensing':
        Cl_hat = r_in*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l + FG_remove_frac*(Cl_s + Cl_d)
        Cl = r_var*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l + FG_remove_frac*(Cl_s + Cl_d)
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)
    if option_Lensing == 'noLensing':
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1[ind[0]] + N_l + FG_remove_frac*(Cl_s + Cl_d)
        Cl = r_var*C_l_r1[ind[0]] + N_l + FG_remove_frac*(Cl_s + Cl_d)
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)

    return L_wL

def Cal_CMB_likelifood_lmax_inclDelAngle(r_max_arr, l_max, r_in,
                                          uKarcmin, del_alpha, FWHM, fsky,
                                          option_Lensing):
    L_mult = 1.
    for j in range(2,l_max+1):
        ell_in = np.array([j])
        L = Cal_CMB_likelifood_perell_inclDelAngle(ell_in, r_max_arr, r_in, \
                                                       uKarcmin, del_alpha, FWHM, fsky, \
                                                       option_Lensing)
        L_mult *= L
    return L_mult/max(L_mult)

def Cal_CMB_likelifood_perell_inclDelAngle(ell_in, r_var, r_in,
                                           uKarcmin, del_alpha, FWHM, fsky,
                                           option_Lensing):
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing
        C_l_ee = EEin_L/prefact_g

    ind = np.where(ell_in == np.array(ell_P))

    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l[ind[0]]
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')

    if option_Lensing == 'Lensing':
        Cl_fake = C_l_ee[ind[0]] * np.sin(2.*del_alpha)**2
        Cl_hat = r_in*C_l_r1[ind[0]] + C_l_lensing[ind[0]] + N_l + Cl_fake
        Cl = r_var*C_l_r1[ind[0]] + C_l_lensing[ind[0]]  + N_l #+ Cl_fake
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)
    if option_Lensing == 'noLensing':
        Cl_fake = C_l_ee[ind[0]] * np.sin(2.*del_alpha)**2
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1[ind[0]] + N_l # + Cl_fake
        Cl = r_var*C_l_r1[ind[0]] + N_l + Cl_fake
        L_wL = np.sqrt(Cl_hat**(2.*ell_in-1)/Cl**(2.*ell_in+1)) * np.exp(-(2.*ell_in+1.)*Cl_hat/Cl/2.)

    return L_wL


def Plot_CMB_likelifood_perell_incl1oL(r_in,
                                       uKarcmin,F_l,FWHM,fsky,
                                       option_Lensing,dirout,filename):
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ell_in = ell_P
    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
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

    py.figure()
    py.subplot(111)
    ell_in = ell_P
#    py.plot(ell_in, Cl_hat*prefact_g, label='$C_l^P+C_l^L$')
    py.plot(ell_in, N_l*prefact_g, label='$N_l$')
    py.plot(ell_in, F_l/ell_in**2.*prefact_g, label='$F_l/l^2$')
    py.plot(ell_in, (N_l+F_l/ell_in**2.)*prefact_g, label='$N_l+F_l/l^2$')
    py.plot(ell_in, C_l_r1*r_in*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_lensing*prefact_g, label='$C_l^L$')
    py.plot(ell_in, C_l_r1*r_in*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_r1*2e-3*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_r1*2e-3*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')
    py.xlim([1,200])
    py.ylim([1e-8,1e-1])
    py.loglog()
    py.title(str(uKarcmin)+'$\mu$K.arcmin, $F_l$='+str(F_l))
    py.xlabel('$l$')
    py.ylabel('$l(l+1)/2\pi C_l$ [$\mu$K$^2$]')
    py.legend(loc='best',prop={'size':11})
    py.savefig(dirout+'/'+filename)
    py.clf()

def Plot_CMB_likelifood_perell_incl1oL_FG(r_in,
                                          uKarcmin,F_l,FWHM,fsky,
                                          option_Lensing,dirout,filename):
    if option_prefact==True: 
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ell_in = ell_P
    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
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

    py.figure()
    py.subplot(111)
    ell_in = ell_P
#    py.plot(ell_in, Cl_hat*prefact_g, label='$C_l^P+C_l^L$')
#    py.plot(ell_in, N_l*prefact_g, label='$N_l$')
#    py.plot(ell_in, F_l/ell_in**2.*prefact_g, label='$F_l/l^2$')
    py.plot(ell_in, (N_l+F_l/ell_in**2.)*prefact_g, 'm',label='$N_l+F_l/l^2$', linewidth=5)
#    py.plot(ell_in, C_l_r1*r_in*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
#    py.plot(ell_in, C_l_lensing*prefact_g, label='$C_l^L$')
#    py.plot(ell_in, C_l_r1*r_in*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')

    r_in = 1.e-3
    py.plot(ell_in, C_l_r1*r_in*prefact_g, 'b', label='$C_l^P, r=1\\times 10^{-3}$', linewidth=5)
    py.plot(ell_in, C_l_lensing*prefact_g, 'c', label='$C_l^L$', linewidth=5)
    py.plot(ell_in, C_l_r1*r_in*prefact_g+C_l_lensing*prefact_g, 'g', label='$C_l^L+C_l^P, r=1\\times 10^{-3}$', linewidth=5)
#    py.plot(ell_in, C_l_r1*2e-3*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
#    py.plot(ell_in, C_l_r1*2e-3*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')

    ell_fg, S_l_100, D_l_100 =  gen_SDl_Creminelli(ell_in,100.e9)
    ell_fg, S_l_140, D_l_140 =  gen_SDl_Creminelli(ell_in,140.e9)
#    print '-> ', S_l_100, D_l_100
    
#    py.plot(ell_fg, S_l_100*prefact_g, 'c-.', label='S_l (100GHz)')
#    py.plot(ell_fg, D_l_100*prefact_g, 'r-.', label='D_l (100GHz)')
#    py.plot(ell_fg, S_l_140*prefact_g, 'c--', label='S_l (140GHz)')
#    py.plot(ell_fg, D_l_140*prefact_g, 'r--', label='D_l (140GHz)')

#    py.plot(ell_fg, S_l_100*prefact_g*1.e-4, 'c-.', linewidth=4, label='S_l (100GHz) 1%')
    py.plot(ell_fg, D_l_100*prefact_g*1.e-4, 'r--', linewidth=5, label='D_l (100GHz) 1%')
#    py.plot(ell_fg, S_l_140*prefact_g*1.e-4, 'c--', linewidth=4, label='S_l (140GHz) 1%')
#    py.plot(ell_fg, D_l_140*prefact_g*1.e-4, 'r--', linewidth=4, label='D_l (140GHz) 1%')

    py.xlim([1.5,200])
    py.ylim([1e-6,1e-2])
    py.loglog()
#    py.title(str(uKarcmin)+'$\mu$K.arcmin, $F_l$='+str(F_l))
    py.xlabel('$l$',fontsize=17)
    py.ylabel('$l(l+1)/2\pi C_l$ [$\mu$K$^2$]',fontsize=17)
#    py.legend(loc='best',prop={'size':8})
    py.xticks( color = 'k', size = 17)
    py.yticks( color = 'k', size = 17)
    print str(uKarcmin)+'$\mu$K.arcmin, $F_l$='+str(F_l), filename
    py.savefig(dirout+'/'+filename)
    py.savefig(dirout+'/'+filename+'.eps')
    py.clf()
    print ''


def Plot_CMB_likelifood_perell_inclDelAngle_FG(r_in,
                                               uKarcmin,del_angle,FWHM,fsky,
                                               option_Lensing,dirout,filename):
    if option_prefact==True: 
        C_l_ee = EEin_L
        C_l_r1 = BBin_P
        C_l_lensing = BBin_L
        C_l = C_l_r1*r_in + C_l_lensing
    if option_prefact==False: 
        C_l_ee = EEin_L/prefact_g
        C_l_r1 = (BBin_P/prefact_g) 
        C_l_lensing = (BBin_L/prefact_g)
        C_l = C_l_r1*r_in + C_l_lensing

    ell_in = ell_P
    gen_Nl = libcl.gen_Nl(ell_in)
    prefact = gen_Nl.cal_prefact()
    gen_Nl.C_l = C_l
    gen_Nl.fsky = fsky
    gen_Nl.prefact_option(option_prefact)
    gen_Nl.modeloss_option(False)
    gen_Nl.uKarcmin = uKarcmin
    gen_Nl.FWHM = FWHM
    gen_Nl.sigma_b()
    N_l = gen_Nl.gen_KnoxdNl('noCV')

    if option_Lensing == 'Lensing':
        Cl_hat = r_in*C_l_r1 + C_l_lensing + N_l #+ F_l/ell_in**2.
    if option_Lensing == 'noLensing':
        C_l_lensing = np.zeros(len(l_in2))
        Cl_hat = r_in*C_l_r1 + N_l #+ F_l/ell_in**2.

    py.figure()
    py.subplot(111)
    ell_in = ell_P
#    py.plot(ell_in, Cl_hat*prefact_g, label='$C_l^P+C_l^L$')
    py.plot(ell_in, N_l*prefact_g, label='$N_l$')
#    py.plot(ell_in, F_l/ell_in**2.*prefact_g, label='$F_l/l^2$')
#    py.plot(ell_in, (N_l+F_l/ell_in**2.)*prefact_g, label='$N_l+F_l/l^2$')
    py.plot(ell_in, C_l_r1*r_in*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_lensing*prefact_g, label='$C_l^L$')
    py.plot(ell_in, C_l_r1*r_in*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_r1*2e-3*prefact_g, label='$C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_r1*2e-3*prefact_g+C_l_lensing*prefact_g, label='$C_l^L+C_l^P, r=1\\times 10^{-3}$')
    py.plot(ell_in, C_l_ee*prefact_g, 'm-', label='$C_l^{EE}$')
    py.plot(ell_in, C_l_ee*prefact_g*np.sin(2.*del_angle)**2, 'm--', label='$C_l^{EE}, \\alpha=$'+str(del_angle/pi*180.)+'degs')

    ell_fg, S_l_100, D_l_100 =  gen_SDl_Creminelli(ell_in,100.e9)
    ell_fg, S_l_140, D_l_140 =  gen_SDl_Creminelli(ell_in,140.e9)
    print '-> ', S_l_100, D_l_100
    
    py.plot(ell_fg, S_l_100*prefact_g, 'c-.', label='S_l (100GHz)')
    py.plot(ell_fg, D_l_100*prefact_g, 'r-.', label='D_l (100GHz)')
    py.plot(ell_fg, S_l_140*prefact_g, 'c--', label='S_l (140GHz)')
    py.plot(ell_fg, D_l_140*prefact_g, 'r--', label='D_l (140GHz)')

    py.plot(ell_fg, S_l_100*prefact_g*1.e-4, 'c-.', linewidth=4, label='S_l (100GHz) 1%')
    py.plot(ell_fg, D_l_100*prefact_g*1.e-4, 'r-.', linewidth=4, label='D_l (100GHz) 1%')
    py.plot(ell_fg, S_l_140*prefact_g*1.e-4, 'c--', linewidth=4, label='S_l (140GHz) 1%')
    py.plot(ell_fg, D_l_140*prefact_g*1.e-4, 'r--', linewidth=4, label='D_l (140GHz) 1%')

    py.xlim([1,200])
    py.ylim([1e-8,1e-1])
    py.loglog()
    py.title(str(uKarcmin)+'$\mu$K.arcmin, $\Delta\\alpha$='+str(del_angle/pi*180.)+'degs')
    py.xlabel('$l$')
    py.ylabel('$l(l+1)/2\pi C_l$ [$\mu$K$^2$]')
    py.legend(loc='best',prop={'size':8})
    py.savefig(dirout+'/'+filename)
    py.clf()

