import numpy as np
import pylab as py
import lib_matprop as lib_mat
from scipy import integrate



material_name = 'help'
out = lib_mat.read_RunyanTable2(material_name)
print(out)

print('---')
material_name = 'Vespel SP-1'
#out = lib_mat.read_RunyanTable2(material_name)
#print(out['name'], out['a'], out['b'], out['c'], out['n'])
#
#a = out['a']
#b = out['b']
#c = out['c']
#n = out['n']
#
#def thermalconduct_Runyan(T):
#    return a*T**(b+c*T**n)
#
#num = 50
#T_low = np.linspace(0.3,4.2,num)
#int_out = np.zeros(num)
#for i in range(0,num):
#	integral_tmp = integrate.quad(thermalconduct_Runyan, T_low[i], 4.2)
#	int_out[i] = integral_tmp[0]


#for i in range(num-1,0,-1): print(T_low[i],int_out[i])
#py.plot(T_low,int_out)
#py.xlabel('Temperature [K]')
#py.ylabel('Integrated thermal conductance [mW/m]')
#py.grid()
#py.show()

num = 40

thermalcond1 = lib_mat.generate_integratedThermalConduct_Runyan('Vespel SP-1',num)
thermalcond2 = lib_mat.generate_integratedThermalConduct_Runyan('Vespel SP-22',num)
thermalcond3 = lib_mat.generate_integratedThermalConduct_Runyan('PEEK',num)
thermalcond4 = lib_mat.generate_integratedThermalConduct_Runyan('G-10/FR-4',num)

print('Vespel SP-1', 'Vespel SP-22', 'PEEK', 'G-10/FR-4', 'mW/m')
for i in range(num-1,0,-1):
	print(format(thermalcond1[0][i],'.3f'), format(thermalcond1[1][i],'.3f'), format(thermalcond2[1][i],'.3f'), format(thermalcond3[1][i],'.3f'), format(thermalcond4[1][i],'.3f'))

py.plot(thermalcond1[0],thermalcond1[1], label='Vespel SP-1')
py.plot(thermalcond2[0],thermalcond2[1], label='Vespel SP-22')
py.plot(thermalcond3[0],thermalcond3[1], label='PEEK')
py.plot(thermalcond4[0],thermalcond4[1], label='G-10/FR-4')
py.xlabel('Temperature [K]')
py.ylabel('Integrated thermal conductance [mW/m]')
py.legend(loc='best')
py.ylim([0,40])
py.grid()
py.show()
