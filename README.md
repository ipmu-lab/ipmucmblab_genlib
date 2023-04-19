# ipmucmblab_genlib
This is a set of general libraries that we share within Kavli IPMU CMB group.

History:
	init: 2023-April-19, on the day of CD3 opening symposium, the first set of the libraries are from Tomo.

+++++++++++++++++++++++++++++
General rules:
	- When you add any script, please leave a log in readme in each lib holder
	- When you add any script, please prepare a corresponding test script with together with the instruction.
	- Last but not least, it might not be as obvious as hardware, but people spent time writing codes, and so please respect when you use and be careful when you make a change. 
	- The codes for the group people to use fully and accelerate the output. When you think you will be sharing the codes beyong the group, we should discuss every time. 

+++++++++++++++++++++++++++++

- lib_analysis
	- lib_anallysis.py
	- lib_Clmanip.py: Cl related scripts
	- lib_foreground.py: foreground spectra in ell space
	- lib_likelihood.py: likelihood code for r
- lib_cal
	- calibration related scripts
- lib_daq
	- lab data acquisition related code, e.g. labview
- lib_general
	- general library: lib_m.py: read/write, coordinate transformation, histogram, PSD cal., Gaussian fit, filter, binning, and more
	- lib_bb.py: Planck distribution, blackbody rad., ...
- lib_lb
	- litebird basic parameters. Be careful it might be obsolite. Please refer the latest IMO.
- lib_materialproperties
	- script and collection of the data to plot the material properties.
- lib_optics
	- multi-layer transmission cal., 2nd order EMT, HWP ...
- lib_pmu
	- lib_smb.py: fit function for spin down... 
- TestScripts
	- Test scripts to run the library for new users

