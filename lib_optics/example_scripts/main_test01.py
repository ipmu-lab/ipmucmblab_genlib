import numpy as np
import pylab as py  
import lib_optics as lib_o

'''
practice script to use 
	oblique_basic_multilayer_r_t_incloss( n, losstan, d, freq_i, freq_f, freq_int, angle_i, incpol):
in lib_optics

	incpol: 1 or -1 
    	1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident

2023-5-11 @ IPMU, by T. Matsumura

'''

pi = np.pi

n_alumina = 3.14
thickness_m = 5.e-3

# 多層にしたい場合は、自分で好きな層数を準備すれば、アレイの要素数分だけ勝手に総数を増やして計算してくれます。
# 屈折率 n と losstan は、空気ー材料ー空気で三層分。
# 厚み d は材料だけに対応するので要素は一つだけ。
# 例えば、空気ーARー母材ーARー空気にしたかったら、n と losstan は5層分準備。厚み d は三層分準備。
# n = np.array([1.,    n_ar1, n_ar2, ..., n_arn,    n_alumina,     n_arn, ...,n_ar2, n_ar1,    1.])

# 構造を決める -> (RCWA) -> transmittance
# 構造を決める -> area fraction vs z -> (EMT) -> effective index vs z -> transmittance
# 2nd-order EMTは -> SWG_frac2index(freq, n1, n2, f, pitch) in lib_optics

n = np.array([1.,n_alumina,1.])
losstan = np.array([0., 1.e-3, 0.])
d = np.array([thickness_m])

freq_i = 30.e9
freq_f = 200.e9
freq_int = 0.1e9
angle_i = 0./180.*pi #radian
incpol = 1 # out of the plane, s
#incpol = -1 # in the plane, p

output = lib_o.oblique_basic_multilayer_r_t_incloss( n, losstan, d, freq_i, freq_f, freq_int, angle_i, incpol)

freq_GHz = np.abs(output[0]) * 1.e-9
R = np.abs(output[1])**2
T = np.abs(output[2])**2

py.figure()

py.subplot(211)
py.plot(freq_GHz, T)
py.ylim([0.,1.1])
py.ylabel('Transmittance')
py.grid()
py.title('main_test01.py')

py.subplot(212)
py.plot(freq_GHz, R)
py.ylim([0.,1.1])
py.ylabel('Reflectance')
py.grid()
py.xlabel('Frequency [GHz]')

py.show()
