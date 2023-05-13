import numpy as np
import pylab as py  
import lib_optics as lib_o

'''
main_test02.py

practice script to use 
	oblique_basic_multilayer_r_t_incloss( n, losstan, d, freq_i, freq_f, freq_int, angle_i, incpol):
in lib_optics

	incpol: 1 or -1 
    	1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident

	use SWG_frac2index to model one side AR coding based on a pointy linear pyramid

2023-5-11 @ IPMU, by T. Matsumura

'''

pi = np.pi
c_m = 3.e8
n_vac = 1.
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

freq_max = 170.e9
lambda_m = c_m/freq_max
pitch = lambda_m/n_alumina
freq_sws = 100.e9

print("++++++++++++++++++++++++++++++++++++++")
print("")
print("freq_max in GHz", freq_max*1e9)
print("lambda_m in mm", freq_max*1000)
print("pitch in mm", pitch*1000)

total_arheight = 2.e-3
#d_ar = 0.2e-3
num_ar = 10
d_ar = total_arheight/float(num_ar)
line_frac = np.linspace(0,1,num_ar)

print("++++++++++++++++++++++++++++++++++++++")
print("")
print("freq_sws in GHz", freq_sws*1e9)
print("d_ar in mm", d_ar*1e3)
print("num_ar", num_ar)
print("line_frac", line_frac)

n_ar, w = lib_o.SWG_frac2index(freq_sws, n_vac, n_alumina, line_frac, pitch)

n = np.array([1.,n_alumina,1.])
n = np.hstack((np.array(1.), n_ar))
n = np.hstack((n, np.array([n_alumina,1.])))
#losstan = np.array([0., 1.e-3, 0.])
losstan = np.zeros(len(n))

#d = np.array([thickness_m])
d = np.ones(num_ar) * d_ar
d = np.hstack((d, thickness_m))
#d = np.array([thickness_m])

print("++++++++++++++++++++++++++++++++++++++")
print("")
print("array of n", n)
print("array of losstan", losstan)
print("array of d", d)

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
py.title('main_test02.py')

py.subplot(212)
py.plot(freq_GHz, R)
py.ylim([0.,1.1])
py.ylabel('Reflectance')
py.grid()
py.xlabel('Frequency [GHz]')

py.show()
