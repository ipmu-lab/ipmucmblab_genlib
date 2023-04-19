import numpy as np
import pylab as py
import sys
import glob
import lib_planetcal as lib_p
print ''

c = 3.e11
pi = np.pi
radeg = (180./pi)
skysize = 4.*pi

def lib_trancatedGaussianBeamShape(nu_obs_GHz,Dapt_mm,edge_dB,png_filenameout=''):

#NET_rtsec=float(sys.argv[3]) 

    if ((edge_dB <= 0) & (edge_dB > -2)): map_width_rad = 6000.; res = 5. #[mm]
    if ((edge_dB <= -2) & (edge_dB > -10)): map_width_rad = 2000.; res = 10. # [mm]
    if ((edge_dB <= -10) & (edge_dB > -50)): map_width_rad = 1000.; res = 10. # [mm]
    if ((edge_dB <= -50) & (edge_dB > -100)): map_width_rad = 1000.; res = 10. # [mm]

    nu_obs = nu_obs_GHz*1.e9

    print ''
    print 'frequency [GHz]:', nu_obs_GHz
    print 'aperture diameter [mm]: ', Dapt_mm
    print 'edge_dB [dB]: ', edge_dB

    edge_level =  10.**(edge_dB/10.)  
    sigma_r = Dapt_mm/np.sqrt(-8.*np.log(edge_level))
    par_in = np.array([0.,0.,sigma_r, sigma_r, 0., 1.])

    elip = lib_p.ellipticalGaussian()
    elip.resol_rad = res
    elip.map_width_rad = map_width_rad
    elip.par = par_in
    X_el, Y_el, MAP_S_el = elip.gen_flatellip_map()

    num_bin = 100
    Bkk_S_el, kx, ky = lib_p.fft_2d(res,res,MAP_S_el)
    Bk_S_el_kr, Bk_S_el_mean, Bk_S_el_std, Bk_S_el_med = lib_p.cal_Bk(num_bin,kx,ky,Bkk_S_el)

    #ind_el = np.where(MAP_S_el/np.max(MAP_S_el) < edge_level)
    r_el = np.sqrt(X_el**2+ Y_el**2)
    ind_el = np.where(r_el > Dapt_mm*0.5)
    MAP_S_el_edge = np.copy(MAP_S_el)
    MAP_S_el_edge[ind_el] = 0.
    Bkk_S_el_edge, kx, ky = lib_p.fft_2d(res,res,MAP_S_el_edge)
    Bk_S_el_edge_kr, Bk_S_el_edge_mean, Bk_S_el_edge_std, Bk_S_el_edge_med = lib_p.cal_Bk(num_bin,kx,ky,Bkk_S_el_edge)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if png_filenameout != '':
        py.figure(0, figsize=(10,8))

        py.subplot(2,2,1)
        X = X_el
        Y = Y_el
        Z = MAP_S_el
        xmin, xmax, ymin, ymax = np.min(X), np.max(X), np.min(Y), np.max(Y)
        extent = xmin, xmax, ymin, ymax
        im1 = py.imshow(Z, extent=extent)
        py.colorbar()
        py.xlabel('$x$ [mm]')
        py.ylabel('$y$ [mm]')
        py.title('Map space beam')

        py.subplot(2,2,2)
        X = X_el
        Y = Y_el
        Z = MAP_S_el_edge
        xmin, xmax, ymin, ymax = np.min(X), np.max(X), np.min(Y), np.max(Y)
        extent = xmin, xmax, ymin, ymax
        im1 = py.imshow(Z, extent=extent)
        py.colorbar()
        py.xlabel('$x$ [mm]')
        py.ylabel('$y$ [mm]')
        py.title('Map space beam')

    #++++ 
    theta = np.arcsin(np.array(Bk_S_el_kr) * c /nu_obs) * radeg
    theta_edge = np.arcsin(np.array(Bk_S_el_edge_kr) * c /nu_obs) * radeg

    Bk_fit, par = lib_p.Bk1D_fit(theta, Bk_S_el_mean, Bk_S_el_std, 1.)
    Bk_fit_edge, par_edge = lib_p.Bk1D_fit(theta_edge, Bk_S_el_edge_mean, Bk_S_el_edge_std, 1.)

    theta_FHWM = np.sqrt(2./par[0]) * np.sqrt(8.*np.log(2)) * 60. # [arcmin]
    theta_FHWM_edge = np.sqrt(2./par_edge[0]) * np.sqrt(8.*np.log(2)) * 60. # [arcmin]  

    if png_filenameout != '':

        py.subplot(2,2,3)
        py.plot(theta,Bk_S_el_mean,'o')
        py.plot(theta,Bk_fit*max(Bk_S_el_mean),'-')
        py.plot(theta_edge,Bk_S_el_edge_mean,'o')
        py.plot(theta,Bk_fit_edge*max(Bk_S_el_edge_mean),'-')
        py.ylim([1e-10,max(Bk_S_el_mean)*10])
        py.semilogy()
        py.xlabel('$\\theta$ [deg]')
        py.ylabel('Beam')
        py.title('k space beam')

        py.subplot(2,2,4)
        py.plot(theta,Bk_S_el_mean,'o')
        py.plot(theta,Bk_fit*max(Bk_S_el_mean),'-')
        py.plot(theta_edge,Bk_S_el_edge_mean,'o')
        py.plot(theta,Bk_fit_edge*max(Bk_S_el_edge_mean),'-')
        py.ylim([1e-10,max(Bk_S_el_mean)*10])
        py.loglog()
        py.ylim([1e-10,1e7])
        py.title('$\\theta_{FWHM}$=%1.1f [degs]' % theta_FHWM)
        py.xlabel('$\\theta$ [deg]')

        py.savefig(png_filenameout)
        py.clf()

    return theta_FHWM, theta_FHWM_edge
#py.show()
#sys.exit()

