import numpy as np

num_band = 15
num_band_lft = 12
num_band_hft = 3

band_c = np.array([40.0, 50.0, 60.0, 68.0, 78.0, 89.0, 100.0, 119.0, 140.0, 166.0, 195.0, 235.0, 280.0, 337.0, 402.0])
band_i = np.array([34.0, 42.5, 53.1, 60.2, 69.0, 78.8, 88.5, 101.2, 119.0, 141.1, 165.8, 199.8, 238.0, 286.5, 355.8])
band_f = np.array([46.0, 57.5, 66.9, 75.8, 87.0, 99.2, 111.5, 136.9, 161.0, 190.9, 224.3, 270.3, 322.0, 387.6, 448.2])
band_frac = np.array([0.30, 0.30, 0.23, 0.23, 0.23, 0.23, 0.23, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.23])

def band_average(xin,yin):
	out = np.zeros((3,num_band))
	for i in range(num_band):
		ind = np.where((xin >= band_i[i]) & (xin <= band_f[i]))
		out[0,i] = np.mean(band_c[i])
		out[1,i] = np.mean(yin[ind[0]])
		out[2,i] = np.std(yin[ind[0]])
		print( out[0:3,i] )
	return out


def litebird_config_Ver20180422(band_index):
	'''	Telescope
		Band ID
		Center Frequency [GHz]
		Frequency Band [GHz]
		Beam size [arcmin]
		Detector pixel size [mm]
		Total Number of Bolo.
		NET per Bolo. [uKrts]
		NET_array [uKrts]
		Polarization
		Sensitivity [uK']
	'''
	if band_index == 0: return ['Telescope', 'BandID', 'Center freq [GHz]', 'Frequency bandwidth [GHz]', 'Beam size[arcmin]', 'Detector pixel size[mm]', 'Total number Bolo', 'NET per Bolo. [uKrts]', 'NET_array [uKrts]', 'Polarization', 'Sensitivity [uK arcmin]']
	if band_index == 1: return ['LFT',	1,	40,	12,	69.2,	30,	42,	87.0,	17.3,	36.1]
	if band_index == 2: return ['LFT',	2,	50,	15,	56.9,	30,	56,	54.6,	9.4,	19.6]
	if band_index == 3: return ['LFT',	3,	60,	14,	49.0,	30,	42,	48.7,	9.7,	20.2]
	if band_index == 4: return ['LFT',	4,	68,	16,	40.8,	30,	56,	43.4,	5.4,	11.3]
	if band_index == 5: return ['LFT',	5,	78,	18,	36.1,	30,	42,	40.0,	4.9,	10.3]
	if band_index == 6: return ['LFT',	6,	89,	20,	32.3,	30,	56,	36.8,	4.0,	8.4]
	if band_index == 7: return ['LFT',	4,	68,	16,	40.8,	18,	114, 64.9,	5.4,	11.3]
	if band_index == 8: return ['LFT',	5,	78,	18,	36.1,	18,	114, 52.3,	4.9,	10.3]
	if band_index == 9: return ['LFT',	6,	89,	20,	32.3,	18,	114, 43.5,	4.0,	8.4]
	if band_index == 10: return ['LFT',	7,	100, 24, 27.7,	18,	114, 40.1,	4.8,	7.0]
	if band_index == 11: return ['LFT',	8,	119, 36, 23.7,	18,	114, 31.3,	3.8,	5.8]
	if band_index == 12: return ['LFT',	9,	140, 42, 20.7,	18,	114, 29.6,	3.6,	4.7]
	if band_index == 13: return ['HFT',	7,	100, 24, 37.0,	12,	222, 53.7,	4.6,	7.0]
	if band_index == 14: return ['HFT',	8,	119, 36, 31.6,	12,	148, 38.4,	4.1,	5.8]
	if band_index == 15: return ['HFT',	9,	140, 42, 27.6,	12,	222, 33.9,	2.9,	4.7]
	if band_index == 16: return ['HFT',	10,	166, 50, 24.2,	12,	148, 31.8,	3.4,	7.0]
	if band_index == 17: return ['HFT',	11,	195, 58, 21.7,	12,	222, 32.4,	2.8,	5.8]
	if band_index == 18: return ['HFT',	12,	235, 70, 19.6,	12,	148, 36.4,	3.8,	8.0]
	if band_index == 19: return ['HFT',	13,	280, 84, 13.2,	5.2, 338, 62.2,	4.4,	9.1]
	if band_index == 20: return ['HFT',	14,	337, 102, 11.2,	5.2, 338, 78.3,	5.5,	11.4]
	if band_index == 21: return ['HFT',	15,	402, 92, 9.7,	5.2, 338, 134.5, 9.4,	19.6]
