import numpy as np
from scipy import integrate

def read_txt3f_NISTdata(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=4:
            ar = line.split()
#            arr1.append(ar[0])
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
        i+=1
    return np.array(arr2), np.array(arr3)

def read_txt4f_NISTdata(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=4:
            ar = line.split()
#            arr1.append(ar[0])
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
        i+=1
    return np.array(arr2), np.array(arr3), np.array(arr4)

def read_txt_NISTdata_cupper(filename):
    import fileinput
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    filelines = fileinput.input(filename)
    i=0
    for line in filelines:
        if i>=4:
            ar = line.split()
#            arr1.append(ar[0])
            arr2.append(float(ar[1]))
            arr3.append(float(ar[2]))
            arr4.append(float(ar[3]))
            arr5.append(float(ar[4]))
            arr6.append(float(ar[5]))
        i+=1
    return np.array(arr2), np.array(arr3), np.array(arr4), np.array(arr5), np.array(arr6)

def modeleq_10powerlog_8terms(x,par):
	output = 10**(par[0]+par[1]*(np.log10(x)) \
				 + par[2]*(np.log10(x))**2 \
				 + par[3]*(np.log10(x))**3 \
				 + par[4]*(np.log10(x))**4 \
				 + par[5]*(np.log10(x))**5 \
				 + par[6]*(np.log10(x))**6 \
				 + par[7]*(np.log10(x))**7 \
				 + par[8]*(np.log10(x))**8 )
	return output

def read_RunyanTable2(name):
    # https://doi.org/10.1016/j.cryogenics.2008.06.002
    # Cryogenics 48 (2008) 448–454
    # Contents lists available at ScienceDirect Cryogenics
    # journal homepage: www.elsevier.com/locate/cryogenics
    # Thermal conductivity of thermally-isolating polymeric and composite structural support materials between 0.3 and 4 K
    # M.C. Runyan a,*, W.C. Jones a,b
    # Table 2 Best fit values for the thermal conductivity of materials in the temperature range of 0.3 to 4.2 K
    #
    # Material alpha (mW/m K), beta, gamma (1/Kn), n
    # k(T) = alpha T^(beta+gamma T^n)
    # Vespel SP-1, Vespel SP-22, PEEK, PEEK CA30, PEEK GF30, Graphlite CF Rod, Avia Fiberglass Rod, Torlon 4301, G-10/FR-4, Macor, Poco Graphite AXM-5Q
    #2.23    1.44    3.88    3.37    4.14    8.39    10.3    7.77    12.8    4.00    1.54
    #1.92    2.11    2.41    3.20    3.07    2.12    2.28    5.46    2.41    2.55    3.36
    #-0.819  -0.521  -1.43   -2.19   -1.84   -1.05   -0.585  -4.13   -0.921  -0.140  -1.83
    #0.0589  -0.0163 0.0884  0.0640  0.0553  0.181   0.310   0.0682  0.222   0.809   -0.142
    if name=='help': return ('Vespel SP-1, Vespel SP-22, PEEK, PEEK CA30, PEEK GF30, Graphlite CF Rod, Avia Fiberglass Rod, Torlon 4301, G-10/FR-4, Macor, Poco Graphite AXM-5Q')
    if name=='Vespel SP-1': return {'name':'Vespel SP-1', 'a':2.23, 'b':1.92, 'c':-0.819, 'n':0.0589}
    if name=='Vespel SP-22': return {'name':'Vespel SP-22', 'a':1.44, 'b':2.11, 'c':-0.521, 'n':-0.0163}
    if name=='PEEK': return {'name':'PEEK', 'a':3.88, 'b':2.41, 'c':-1.43, 'n':0.0884}
    if name=='PEEK CA30': return {'name':'PEEK CA30', 'a':3.37, 'b':3.20, 'c':-2.19, 'n':0.0640}
    if name=='PEEK GF30': return {'name':'PEEK GF30',  'a':4.14, 'b':3.07, 'c':-1.84, 'n':0.0553}
    if name=='Graphite CF Rod': return {'name':'Graphite CF Rod',  'a':8.39, 'b':2.12, 'c':-1.05, 'n':0.181}
    if name=='Avia Fiberglass Rod': return {'name':'Avia Fiberglass Rod', 'a':10.3, 'b':2.28, 'c':-0.585, 'n':0.310}
    if name=='Torlon 4301': return {'name':'Torlon 4301',  'a':7.77, 'b':5.46, 'c':-4.13, 'n':0.0682}
    if name=='G-10/FR-4': return {'name':'G-10/FR-4', 'a':12.8, 'b':2.41, 'c':-0.921, 'n':0.222}
    if name=='Macor': return {'name':'Macor', 'a':4.00, 'b':2.55, 'c':-0.140, 'n':0.809}
    if name=='Poco Graphite AXM-5Q': return {'name':'Poco Graphite AXM-5Q', 'a':1.54, 'b':3.36, 'c':-1.83, 'n':-0.142}

def generate_integratedThermalConduct_Runyan(material_name,num):
    out = read_RunyanTable2(material_name)
    print(out['name'], out['a'], out['b'], out['c'], out['n'])

    a = out['a']
    b = out['b']
    c = out['c']
    n = out['n']

    def thermalconduct_Runyan(T):
        return a*T**(b+c*T**n)

#    T_low = np.linspace(0.3,4.2,num)
    T_low = np.logspace(np.log10(0.3),np.log10(4.2),num)
    int_out = np.zeros(num)
    for i in range(0,num):
        integral_tmp = integrate.quad(thermalconduct_Runyan, T_low[i], 4.2)
        int_out[i] = integral_tmp[0]

    return T_low, int_out




