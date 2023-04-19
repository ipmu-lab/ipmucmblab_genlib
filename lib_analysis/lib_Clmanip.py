import numpy as np
import pylab as py
from scipy.optimize import fmin
from numpy import random
import sys
from scipy.interpolate import splrep, splev
import healpy as h
import fileinput
import os.path

pi = np.pi
radeg = (180./pi)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class read_cambdata():
    def __init__(self):
        pass
    
    def read_cl_cambdata(self, filename):
        ell = []
        cl1 = []
        cl2 = []
        cl3 = []
        cl4 = []
        for line in fileinput.input(filename):
            ar = line.split()
            if len(ar)>1:
                ell.append(float(ar[0]))
                cl1.append(float(ar[1]))
                cl2.append(float(ar[2]))
                cl3.append(float(ar[3]))
                cl4.append(float(ar[4]))
        out={}
        out={'ell':np.array(ell),'TT':np.array(cl1),'EE':np.array(cl2),'BB':np.array(cl3),'TE':np.array(cl4)}
        self.ell = ell
        return out

    def read_cl_classdata_in_cambformat(self, filename):
        ell = []
        cl1 = []
        cl2 = []
        cl3 = []
        cl4 = []
#        import fileinput
        for line in fileinput.input(filename):
            ar = line.split()
            if ar[0] != '#':
                ell.append(float(ar[0]))
                cl1.append(float(ar[1]))
                cl2.append(float(ar[2]))
                cl3.append(float(ar[3]))
                cl4.append(float(ar[4]))
        out={}
        out={'ell':np.array(ell),'TT':np.array(cl1),'EE':np.array(cl2),'BB':np.array(cl3),'TE':np.array(cl4)}
        self.ell = ell
        return out

    def read_cl_classdata_in_plancklegacydataformat(self, filename):
        ell = []
        cl1 = []
        cl2 = []
        cl3 = []
        cl4 = []
        cl5 = []
#        import fileinput
        for line in fileinput.input(filename):
            ar = line.split()
            if ar[0] != '#':
                ell.append(float(ar[0]))
                cl1.append(float(ar[1]))
                cl2.append(float(ar[2]))
                cl3.append(float(ar[3]))
                cl4.append(float(ar[4]))
                cl5.append(float(ar[5]))
        out={}
        out={'ell':np.array(ell),'TT':np.array(cl1),'TE':np.array(cl2),'EE':np.array(cl3),'BB':np.array(cl4),'PP':np.array(cl5)}
        self.ell = ell
        return out

    def write_cl_cambdata(self, filename_out, ell, TT, EE, BB, TE):

        if os.path.exists(filename_out):
            print( filename_out, " exists already.")
        else:
            print( filename_out, " does not exist.")

        num = len(ell)
        f = open(filename_out,'w')
        for i in range(num):
            f.write('%d     %1.5e     %1.5e     %1.5e     %1.5e \n' % (ell[i], TT[i], EE[i], BB[i], TE[i]))
        f.close()

    def read_cldata(self, filename):
        ell = []
        cl1 = []
        cl2 = []
        cl3 = []
        cl4 = []
#        import fileinput
        for line in fileinput.input(filename):
            ar = line.split()
            if len(ar)>1:
                ell.append(float(ar[0]))
                cl1.append(float(ar[1]))
                cl2.append(float(ar[2]))
                cl3.append(float(ar[3]))
                cl4.append(float(ar[4]))
        print( "   END: reading "+filename)
        return np.array(ell), np.array(cl1), np.array(cl2), np.array(cl3), np.array(cl4)

    def read_two_cls(self, filename_prim, filename_lens):
        self.cl_prim = self.read_cl_cambdata(filename_prim)
        self.cl_lens = self.read_cl_cambdata(filename_lens)
        if len(self.cl_prim['ell']) != len(self.cl_lens['ell']):
            print( '[WARNING] The # of ell from two files are not the same!')
            print( '[WARNING] Match the length of the two Cls to the shorter array')
            if len(self.cl_prim['ell']) > len(self.cl_lens['ell']): 
                self.cl_prim = self.match_Cllength(self.cl_prim,self.cl_lens['ell'])
            if len(self.cl_prim['ell']) < len(self.cl_lens['ell']): 
                self.cl_lens = self.match_Cllength(self.cl_lens,self.cl_prim['ell'])
        return self.cl_prim, self.cl_lens

    def interpol_1D(self, x_in, y_in, x_out):
        tmp = splrep(x_in,y_in)
        y_out = splev(x_out,tmp)
        return y_out
    
    def match_Cllength(self,cls,x_out):
        x_in = cls['ell']
        num_in = len(x_in)
        num_out = len(x_out)
        if num_out > num_in:
            print( '[ERROR] The array to trancate is shorter than the reference.')
        y_dict_out = {}
        y_dict_out['ell'] = x_out
        name = ['TT','EE','BB','TE']
        for i_name in name:
            y_dict_out[i_name] = cls[i_name][0:num_out]
        return y_dict_out
        
    def Cl_interpol(self,cls,x_out):
        x_in = cls['ell']
        y_dict_out = {}
        y_dict_out['ell'] = x_out
        name = ['TT','EE','BB','TE']
        for i_name in name:
            y_in = cls[i_name]
            y_out = self.interpol_1D(x_in, y_in, x_out)
            y_dict_out[i_name] = y_out
        return y_dict_out

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class FG_Clmodel():
    def __init__(self, ell):
        self.ell = ell
#        self.prefactor = self.ell*(self.ell+1.)/(2.*pi)

    def gen_CClsynch(self,nu,params):
        self.nu = nu
        self.A_s = params['A_s'] 
        self.nu_s = params['nu_s'] 
        self.beta_s = params['beta_s'] 
        self.ell_in0 = params['ell_in0'] 
        self.m_s = params['m_s'] 
        self.CClsynch = self.A_s*(self.nu/self.nu_s)**(2.*self.beta_s)*(self.ell/self.ell_in0)**self.m_s
        return self.CClsynch

    def gen_CCldust(self,nu,params):
        self.nu = nu
        self.A_d = params['A_d'] 
        self.nu_d = params['nu_d'] 
        self.beta_d = params['beta_d'] 
        self.ell_in0 = params['ell_in0'] 
        self.m_d = params['m_d'] 
        self.CCldust = self.A_d*(self.nu/self.nu_d)**(2.*self.beta_d)*(self.ell/self.ell_in0)**self.m_d
        return self.CCldust

    def gen_CCldust_model(self,nu,params):
        self.nu = nu
        self.A_d = params['A_d'] 
        self.nu_d = params['nu_d'] 
        self.beta_d = params['beta_d'] 
        self.ell_in0 = params['ell_in0'] 
        self.m_d = params['m_d'] 
        self.CCldust = self.A_d*(self.nu)**(2.*self.beta_d)*(self.ell)**self.m_d
        return self.CCldust

    def gen_dust_2bands(self, ell, nu1, nu2, fg_par, model=False):
        # generatet the input foreground 
        #        self.fg_par = fg_par
        dust = FG_Clmodel(ell)
        if model == False:
            self.model_dust1 = dust.gen_CCldust(nu1,params=fg_par)
            self.model_dust2 = dust.gen_CCldust(nu2,params=fg_par)
        if model == True:
            self.model_dust1 = dust.gen_CCldust_model(nu1,params=fg_par)
            self.model_dust2 = dust.gen_CCldust_model(nu2,params=fg_par)

        return self.model_dust1, self.model_dust2

    def gen_dust_3bands(self, ell, nu1, nu2, nu3, fg_par, model=False):
        # generatet the input foreground 
        #        self.fg_par = fg_par
        dust = FG_Clmodel(ell)
        if model == False:
            self.model_dust1 = dust.gen_CCldust(nu1,params=fg_par)
            self.model_dust2 = dust.gen_CCldust(nu2,params=fg_par)
            self.model_dust3 = dust.gen_CCldust(nu3,params=fg_par)
        if model == True:
            self.model_dust1 = dust.gen_CCldust_model(nu1,params=fg_par)
            self.model_dust2 = dust.gen_CCldust_model(nu2,params=fg_par)
            self.model_dust3 = dust.gen_CCldust_model(nu3,params=fg_par)
        return self.model_dust1, self.model_dust2, self.model_dust3

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class gen_Nl():
    def __init__(self,ell):
        self.num_ell = len(ell)
        self.ell = ell
        self.fsky = 1.
        self.uKarcmin = 1.
        self.FWHM = 1.
        self.C_l = np.zeros(self.num_ell)

    def printout(self):
        print( '')
        print( '===INPUT TO gen_Nl class===')
        print( '   skyarea [degs^2], fsky', '{0:2.2f} {1:2.4f}'.format(self.fsky*(4.*pi*(180./pi)**2), self.fsky))
        print( '   uKarcmin', '{0:2.3e}'.format(self.uKarcmin))
        print( '   FWHM [arcmin]', '{0:2.2f}'.format(self.FWHM))
        print( '===END===')
        print( '')

    def gen_Cl(self):
        out = self.prefact * self.C_l
        return out

    def sigma_b(self):
        self.sigma_b = (self.FWHM/60.)*pi/180./np.sqrt(8.*np.log(2.))
        return self.sigma_b

    def cal_prefact(self):
        prefact = (self.ell*(self.ell+1.)/(2.*pi))
        return prefact

    def cal_deprefact(self):
        deprefact = 1./(self.ell*(self.ell+1.)/(2.*pi))
        return deprefact

    def modeloss_option(self,mode_err_option):
        if mode_err_option==True: self.mode_err_prefac = np.sqrt(2./((2.*self.ell+1.)*self.fsky))
        if mode_err_option==False: self.mode_err_prefac = 1.
        return self.mode_err_prefac

    def prefact_option(self,prefact_option):
        if prefact_option==True: self.prefact = self.ell*(self.ell+1.)/(2.*pi)
        if prefact_option==False: self.prefact = np.ones(self.num_ell)
        return self.prefact

    def PBTelescopePar_1of(self, fknee, scanspeed_ground, el):
        scanspeed_el = scanspeed_ground*np.cos(el/180.*pi)
        theta_el = scanspeed_el/fknee
        l_knee_el = 180./theta_el
#        print 'l_knee ', l_knee_el, 'at el=', el, '[degs]'
        return l_knee_el

#    def gen_KnoxdNl(self):
#        uKrad = self.uKarcmin * pi/(180.*60.)
#        N_l = uKrad**2.
#        out = self.prefact * self.mode_err_prefac * (self.C_l + N_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
#        return out

    def gen_KnoxdNl(self,option):
        uKrad = self.uKarcmin * pi/(180.*60.)
        N_l = uKrad**2.
        if option == 'noCV':
            out = self.prefact * self.mode_err_prefac * N_l * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
        if option == 'CV':
            out = self.prefact * self.mode_err_prefac * (self.C_l + N_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
        return out

    def gen_KnoxdNl_constbin(self,bin_const):
        uKrad = self.uKarcmin * pi/(180.*60.)
        N_l = uKrad**2.
        out = self.prefact * self.mode_err_prefac * (self.C_l + N_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
        out = out/np.sqrt(bin_const)
        return out

    def gen_KnoxdNl_add1of(self,ell_0,alpha):
        uKrad = self.uKarcmin * pi/(180.*60.)
        N_l = uKrad**2.*(1.+(ell_0/self.ell)**alpha)
        out = self.prefact * self.mode_err_prefac * (self.C_l + N_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
        return out

    def gen_KnoxdNl_add1of_constbin(self,ell_0,alpha,bin_const):
        uKrad = self.uKarcmin * pi/(180.*60.)
        N_l = uKrad**2.*(1.+(ell_0/self.ell)**alpha)
        out = self.prefact * self.mode_err_prefac * (self.C_l + N_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2.)
        out = out/np.sqrt(bin_const)
        return out

    def gen_whitenoise_data_from_alm(self):
        uKrad = self.uKarcmin * pi/(180.*60.)
        num_ell = len(self.ell)
        noise_l = np.zeros(num_ell)
        for l in range(0,num_ell):
#            seed = np.random.randint(0,1e4)
#            np.random.seed(seed)
            m_sum = 2*l+1
#            m_sum = int((2.*l+1.)*np.sqrt(self.fsky))
#            a_lm = np.sqrt(self.C_l[l]) * random.randn(m_sum)
            a_lm = np.sqrt(self.C_l[l]) * np.random.normal(0.,1.,m_sum)
#            np.random.seed(seed+1)
#            n_lm = uKrad * np.sqrt(np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * random.randn(m_sum)
            n_lm = uKrad * np.sqrt(np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * np.random.normal(0.,1.,m_sum)
            n_lm2 = (a_lm + n_lm)**2
            noise_l[l] = np.sum(n_lm2)/float(m_sum*np.sqrt(self.fsky))  #np.mean(n_lm2)
#            noise_l[l] = np.sum(n_lm2)/float(m_sum)  #np.mean(n_lm2)
        out = self.prefact * noise_l 
        return out

    def gen_whitenoise_add1of_data_from_alm(self,ell_0,alpha):
        uKrad = self.uKarcmin * pi/(180.*60.)
        num_ell = len(self.ell)
        noise_l = np.zeros(num_ell)
        for l in range(0,num_ell):
#            seed = np.random.randint(0,1e4)
#            np.random.seed(seed)
            m_sum = 2*l+1
#            m_sum = int((2.*l+1.)*self.fsky)
            a_lm = np.sqrt(self.C_l[l]) * np.random.normal(0.,1.,m_sum)
#            a_lm = np.sqrt(self.C_l[l]) * random.randn(m_sum)
#            np.random.seed(seed+1)
            n_lm = uKrad * np.sqrt((1.+(ell_0/self.ell[l])**alpha) * np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * random.randn(m_sum)
            n_lm2 = (a_lm + n_lm)**2
            noise_l[l] = np.sum(n_lm2)/float(m_sum*np.sqrt(self.fsky))  #np.mean(n_lm2)
#            noise_l[l] = np.sum(n_lm2)/float(m_sum)  #np.mean(n_lm2)
        out = self.prefact * noise_l 
        return out

    def gen_whitenoise_model_from_alm(self):
        uKrad = self.uKarcmin * pi/(180.*60.)
        num_ell = len(self.ell)
        noise_l = np.zeros(num_ell)
        for l in range(0,num_ell):
#            seed = np.random.randint(0,1e4)
#            np.random.seed(seed)
            m_sum = 2*l+1
#            m_sum = int((2.*l+1.)*np.sqrt(self.fsky))
#            a_lm = np.sqrt(self.C_l[l]) * random.randn(m_sum)
            a_lm = np.sqrt(self.C_l[l]) #* np.random.normal(0.,1.,m_sum)
#            np.random.seed(seed+1)
#            n_lm = uKrad * np.sqrt(np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * random.randn(m_sum)
            n_lm = uKrad * np.sqrt(np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * np.ones(m_sum)
            n_lm2 = (a_lm + n_lm)**2
            noise_l[l] = np.sum(n_lm2)/float(m_sum*np.sqrt(self.fsky))  #np.mean(n_lm2)
#            noise_l[l] = np.sum(n_lm2)/float(m_sum)  #np.mean(n_lm2)
        out = self.prefact * noise_l

#        uKrad = self.uKarcmin * pi/(180.*60.)
#        num_ell = len(self.ell)
#        noise_l = uKrad**2
#        out = self.prefact * (self.C_l + noise_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2)
        return out

    def gen_whitenoise_add1of_model_from_alm(self,ell_0,alpha):
        uKrad = self.uKarcmin * pi/(180.*60.)
        num_ell = len(self.ell)
        noise_l = np.zeros(num_ell)
        for l in range(0,num_ell):
#            seed = np.random.randint(0,1e4)
#            np.random.seed(seed)
            m_sum = 2*l+1
#            m_sum = int((2.*l+1.)*self.fsky)
            a_lm = np.sqrt(self.C_l[l]) * np.random.normal(0.,1.,m_sum)
#            a_lm = np.sqrt(self.C_l[l]) * random.randn(m_sum)
#            np.random.seed(seed+1)
            n_lm = uKrad * np.sqrt((1.+(ell_0/self.ell[l])**alpha) * np.exp(self.ell[l]*(self.ell[l]+1.)*self.sigma_b**2)) * np.ones(m_sum) #random.randn(m_sum)
            n_lm2 = (a_lm + n_lm)**2
            noise_l[l] = np.sum(n_lm2)/float(m_sum*np.sqrt(self.fsky))  #np.mean(n_lm2)
#            noise_l[l] = np.sum(n_lm2)/float(m_sum)  #np.mean(n_lm2)
        out = self.prefact * noise_l 

#        uKrad = self.uKarcmin * pi/(180.*60.)
#        num_ell = len(self.ell)
#        noise_l = uKrad**2 * (1.+(ell_0/self.ell)**alpha)
#        out = self.prefact * (self.C_l + noise_l) * np.exp(self.ell*(self.ell+1.)*self.sigma_b**2)
        return out

    def gen_Nl_spline(self, n_l):
        num = len(n_l)
        num_ave = 20
        num_new = int(num/num_ave)
        x_in = np.zeros(num_new-1)
        y_in = np.zeros(num_new-1)
        for i in range(0,num_new-1):
            #print i, i*num_ave, (i+1)*num_ave-1
            x_in[i] = np.mean(ell[i*num_ave:(i+1)*num_ave-1])
            y_in[i] = np.mean(n_l[i*num_ave:(i+1)*num_ave-1])
        tmp = splrep(x_in,y_in)
        y_out = splev(self.ell,tmp)
        return self.ell, y_out

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class uKarcmin():
    def __init__(self):
        self.NET = 1.
        self.Nbolo = 1.
        self.Tobs = 1.
        self.sky_deg2 = 1.
    
    def cal_arrayNET(self):
        array_sensitivity = self.NET/np.sqrt(self.Nbolo)
        return array_sensitivity

    def cal_fsky(self):
        self.fullsky_degs2 = (4.*pi / pi**2 * 180.**2) # degs^2
        self.fsky = (self.sky_degs2/self.fullsky_degs2)
        return self.fsky

    def cal_uKarcmin(self, unit=1.):
        rad2arcmin = (60.*180./pi)
        self.unit = unit
        self.uKarcmin = np.sqrt(4.*pi*self.fsky*2.*self.NET**2./self.Tobs/self.Nbolo)*rad2arcmin
        return self.uKarcmin*self.unit
        
    def printout(self):
        print( '')
        print( '===INPUT===')
        print( '   NET', self.NET)
        print( '   Nbolo', self.Nbolo)
        print( '   Tobs [sec]', self.Tobs)
        print( '   patch size [degs^2]', self.sky_degs2)
        print( '===CalResults===')
        print( '   skyarea {0:2.2f} [degs^2], fsky {1:2.5f}'.format(self.fsky*self.fullsky_degs2, self.fsky))
        print( '   uKarcmin', '{0:2.3e}'.format(self.uKarcmin*self.unit))
        print( '   array NET[uK rtsec]', '{0:2.2f}'.format(np.sqrt(2.*self.NET**2./self.Nbolo)))
        print( '===END===')
        print( '' )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def extract_uKarcmin_from_Nl(ell, n_l, FWHM):
    sigma_b = (FWHM/60.)*pi/180./np.sqrt(8.*np.log(2.))
    x = ell
    Nl = 2.*pi*n_l*np.exp(-x*(x+1.)*sigma_b**2.)
    fit = np.polyfit(x, Nl, 2)
    uKrad = np.sqrt(fit[0])
    uKarcmin = uKrad /pi*180.*60.
    return uKarcmin

def subtract_noise(Cl_in, N_l):
    Cl_out = Cl_in - N_l
    return Cl_out

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Cl_Util():
    def __init__(self):
        pass
        
    def make_combined_data(self, initial_r, b_prim, b_lens, foreground, noise):
        BB1 = initial_r*b_prim + b_lens + foreground + noise
        return BB1

    def make_combined_data_x2(self, initial_r, b_prim, b_lens, foreground1, foreground2, noise1, noise2):
        BB1 = self.make_combined_data(initial_r, b_prim, b_lens, foreground1, noise1)
        BB2 = self.make_combined_data(initial_r, b_prim, b_lens, foreground2, noise2)
        return BB1, BB2

    def make_combined_data_x3(self, initial_r, b_prim, b_lens, foreground1, foreground2, foreground3, noise1, noise2, noise3):
        BB1 = self.make_combined_data(initial_r, b_prim, b_lens, foreground1, noise1)
        BB2 = self.make_combined_data(initial_r, b_prim, b_lens, foreground2, noise2)
        BB3 = self.make_combined_data(initial_r, b_prim, b_lens, foreground3, noise3)
        return BB1, BB2, BB3

class gen_bintab():
    def __init__(self,ell):
        self.del_bin = 100
        self.blini = 51
        self.ell = ell
        self.num_ell = len(ell)

    def gen_bintab_const(self):
        blmin = int(self.blini - self.del_bin/2.)
        bin_end_tmp = 0
        num_of_bin = 1
        bin_arr = self.blini
        while(bin_end_tmp < max(self.ell)):
            bin_end_tmp = blmin+self.del_bin*num_of_bin
            bin_arr = np.hstack((bin_arr,int(bin_end_tmp + self.del_bin/2.)))
            num_of_bin += 1
        return bin_arr[:-2]

class binning():
    def __init__(self, bin_arr):
        self.bin_arr = bin_arr

    def binning_by_bintab(self, x, y, option=True):
        if option==False: 
            return x, y, np.zeros(len(x))
        if option: 
            num_bin = len(self.bin_arr)
            del_bin = self.bin_arr[1]-self.bin_arr[0]
            bin_arr_init = self.bin_arr - int(del_bin/2.)
#            print bin_arr_init, del_bin

            x_bin = 0
            y_bin = 0
            y_rms = 0
            for i in range(0,num_bin):
#                if bin_arr_init[i] != 0:
                if i != 0:
                    i_binarr = bin_arr_init[i]+np.arange(0,del_bin,dtype=int)
                    x_bin = np.hstack((x_bin,np.mean(x[i_binarr])))
                    y_bin = np.hstack((y_bin,np.mean(y[i_binarr])))
                    y_rms = np.hstack((y_rms,np.std(y[i_binarr])))
            return x_bin[1:], y_bin[1:], y_rms[1:]

def test_binning():
    ell = np.arange(1,1500)
    out = gen_bintab(ell)
    bin_arr = out.gen_bintab_const()
    print( bin_arr)

    y = 0.01*ell**2
    bin = binning(bin_arr)
    xx, yy = bin.binning_by_bintab(ell,y)
    print( xx, yy)
    py.plot(ell,y)
    py.plot(xx,yy,'.')
#    py.show()
    
def gen_dipole(nside,unit):
    # from WMAP 1st yaer Bennett et al
    l = 263.85 # +/-0.1 degs
    b = 48.25 # +/-0.04 degs
    amp_mK = 3.346 # +/-0.017 mK
    if unit == 'uK': amp_unit = amp_mK * 1.e3; print( unit)
    if unit == 'mK': amp_unit = amp_mK; print( unit)
    if unit == 'K': amp_unit = amp_mK * 1.e-3; print( unit)

    npix = h.nside2npix(nside)
    ipix = range(npix)
    theta, phi = h.pix2ang(nside,ipix)
    dipole = np.cos(theta)
    
    xyz = h.ang2vec(theta,phi)
    theta0 = pi/2.-b/radeg;  phi0 = l/radeg
    x = np.cos(phi0)*np.cos(theta0)*xyz[:,0] + np.sin(phi0)*np.cos(theta0)*xyz[:,1] - np.sin(theta0)*xyz[:,2]
    y = - np.sin(phi0)*xyz[:,0]+np.cos(phi0)*xyz[:,1]
    z = np.sin(theta0)*np.cos(phi0)*xyz[:,0] + np.sin(phi0)*np.sin(theta0)*xyz[:,1] + np.cos(theta0)*xyz[:,2]

    ipix = h.vec2pix(nside,x,y,z)

    dipole_out = amp_unit*dipole[ipix]
    return dipole_out
    
if __name__ == "__main__":
    
    test_binning()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
