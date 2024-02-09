#! /usr/bin/python
#
"""
02 Feb 2024 LMY

This is the code used by Young et al (2022) ApJ 933, 90 for the work described in Appendix B.
(Note to me: descended from radex_grid5dv2_co.py)

PURPOSE: generate grids of CO isotopologue line intensities and line ratios for a variety of 
physical conditions.  Format the output to be used by sample_radex_output.py.

#### note here
You need to have radex installed so that you can call from the command line
There are now a plethora of python wrappers that will call it for you

First the code runs radex to predict the intensities of the 12CO 1-0, 2-1, and 3-2 transitions
for a 3D grid of 12CO column density, gas kinetic temperature, and H2 volume density nH2.
(The assumed line width is fixed, as that parameter is degenerate with the total column density.)
The code writes a bunch of 3D FITS files called things like logtemp_co.fits, ratio12co2110.fits, etc.
All the cubes have the same dimensions and they are structured as follows.
Each pixel corresponds to a set of physical conditions (column density, volume density, and temperature).
To use those cubes, you select a pixel [i, j, k].
logtemp_co[i, j, k] tells you the log of the gas kinetic temperature for that set of conditions.
trad_12co21[i, j, k] tells you the radiation temperature of the 12CO2-1 line for that set of conditions.
ratio12co3210_results[i, j, k] tells you the 12CO3-2/1-0 line ratio for that set of conditions.
And so on.

The code then makes similar calculations for 13CO and C18O, producing a
similar set of 3D fits files for them.  The ranges of 13CO and C18O column densities covered are
controlled by the fiducial 12CO/13CO and 12CO/C18O ratios given in the "setup" part of the code.
For example, if the grid parameters describe log N(H2) = 20.0 to 24.0 and we assume 12CO/H2 = 8.0e-5,
the 3D cubes cover N(12CO) = 8.0e15 to 8.0e19. At a fiducial value of 12CO/13CO = 40, the 3D 13CO
cubes cover N(13CO) = 2.0e14 to 2.0e18.

The results are stored as FITS files so that I could load multiple cubes into the casa viewer 
simultaneously, and have it report the line ratios and physical conditions at each position 
as I moved the cursor interactively through the cubes.

The code then builds a set of 5D cubes to be used by the error analysis in sample_radex_output.py.
The 5D cubes have these axes: N(H2), volume density, temperature, 12CO/13CO abundance ratio, 
and 13CO/C18O abundance ratio, in that order.  (The 12CO/H2 abundance ratio is fixed.)
The 5D cubes are built as follows.
Imagine choosing values for the volume density and temperature.  You take the 12CO1-0 line
intensity from the 3D cube at the pixel with N(12CO) = 8.0e15, for example, and the 13CO1-0 line intensity 
from the 3D cube at the pixel with N(13CO) = 8.0e14.  Then you can compute the 12CO1-0/13CO1-0 line
ratio for the case N(H2) = 1.0e20, N(12CO) = 8.0e15, and 12CO/13CO = 10.
In the error analysis step (the second python script), the MCMC samplers walk around in the 5D cubes
exploring line ratios as a function of those 5 physical parameters.

The workflow for using these two scripts will be
1. Edit radex_makecubes.py; check the grid boundaries and step sizes to verify the parameter ranges of interest.
   Also adjust the radex function call to match your system.
2. Run radex_makecubes.py.  The output FITS files may take up several GB on disk, depending on the size of the grids.
3. Edit sample_radex_output.py; adjust values of the measurements (the line ratios and 12CO brightness
    temperature) to match your data.
4. Run sample_radex_output.py and enjoy the plots.


TO DO: 

    
WISH LIST:
    - parallel processing.  might be able to have each core work on a different transition
    - some kind of progress meter during the radex calcs
    - improve headers of output fits files with more info on axis labels
    - better handling of the calculations with insanely high optical depths 
      (they usually arent in my parameter ranges of interest, but they are still klutzy)
      
      
"""
import os
import time
import numpy as np
import astropy.io.fits as fits

#%%
#  ---------  How to format and interpret the input and output files. --------------

def write_input_wrapper():
    # infile will store a record of all the parameter combinations used for every run of radex
    infile = open('radex.inp','w')
    for icd in range(ncd+1):
        for idens in range(ndens+1):
            for itemp in range(ntemp+1):
    
                # choose parameters
                dens = denstouse[idens]
                cd12co = cdtouse[icd]
                temp = tempstouse[itemp]
                
                # scale each molecule's column density through its abundance
                cdmol = cd12co * abun_rat
                
                # store parameters in infile
                write_input(infile,temp,dens,mole,cdmol,dv)
                if (itemp == ntemp and idens == ndens and icd == ncd):  # stop
                    infile.write('0\n')
                    infile.close()
                    print('Last set of params', mole, temp, dens, cdmol)
                else:
                    infile.write('1\n')  # keep going
                    

def write_input(infile,tkin,nh2,mole,cdmol,dv):
    # dumps parameters to infile to store them in the format radex wants
    infile.write(mole+'.dat\n')  # record where we're getting the Einstein As and the collision probabilities etc from
    infile.write('radex.out\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')  # frequencies of the transitions going into the ratio
    infile.write(str(tkin)+'\n')   # kinetic temperature
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')  # density (n, not log n)
    infile.write(str(tbg)+'\n')
    infile.write(str(cdmol)+'\n')  # column density
    infile.write(str(dv)+'\n')   #  line width


def read_radex_grid(outfile,ntemp,ndens,ncd,abun_rat):
    # for default setup, radex.out stores all the output calculations but not in a convenient way.
    # so this bit of code reads the inconvenient output file, extracts what I want, and preps it for writing to FITS files

    outfile  = open('radex.out')
    temp_results = np.zeros((ncd+1,ndens+1,ntemp+1))
    dens_results = np.zeros_like(temp_results)
    cd_results = np.zeros_like(temp_results)
    line1_results = np.zeros_like(temp_results)
    line2_results = np.zeros_like(temp_results)
    line3_results = np.zeros_like(temp_results)
    tau1 = np.zeros_like(temp_results)
    tau2 = np.zeros_like(temp_results)
    tau3 = np.zeros_like(temp_results)
    line1_trad = np.zeros_like(temp_results)
    line2_trad = np.zeros_like(temp_results)
    line3_trad = np.zeros_like(temp_results)
    
    count = 0

    for icd in range(ncd+1):
        for idens in range(ndens+1):
            for itemp in range(ntemp+1):

                # read results for this set of parameters
                line  = outfile.readline()
                words = line.split()
                # ignore lines until we find the one that reports the assumed kinetic temperature
                while (words[1] != "T(kin)"):
                    line  = outfile.readline()
                    words = line.split()
                temp  = float(words[-1])  # temp is the last entry in this line
                line  = outfile.readline()  # density is the immediate next line
                words = line.split()
                dens  = float(words[-1])  # density value is the last one in this line
                line  = outfile.readline()  
                words = line.split()
                # if (count==0):
                #     print(words)
                # ignore more lines until we find the one with the column density
                while (words[1] != "Column"):  
                    line  = outfile.readline()
                    words = line.split()
                #     if (count==0):
                #         print(words)
                cdmol = float(words[-1])
                line  = outfile.readline()  
                words = line.split()
                # ignore more lines until we find the one with the column headers
                while (words[-1] != "FLUX"):  
                    line  = outfile.readline()
                    words = line.split()
                line  = outfile.readline()  # skip row with units
                line  = outfile.readline() # this should have the results for the first transition
                words = line.split()
                ftmp  = float(words[4])  # item 4 is the frequency, which is used to identify the transition
                while ((ftmp < flow*(1-bw)) or (ftmp > flow/(1-bw))):
                    line  = outfile.readline()
                    words = line.split()
                    ftmp  = float(words[4])
                low   = float(words[-2])  # this is the lower transition; item -2 picks out the flux in K km/s
                trad_low = float(words[8]) # radiation temperature in K.  actually it seems to be just doing flux = trad * deltav * 1.065
                tau_low = float(words[7])
                line  = outfile.readline()  # this should be the next transition
                words = line.split()
                ftmp  = float(words[4])  # similarly, used to identify the transition
                while ((ftmp < fmid*(1-bw)) or (ftmp > fmid/(1-bw))):
                    line  = outfile.readline()
                    words = line.split()
                    ftmp  = float(words[4])
                mid   = float(words[-2])  # similarly, for the flux in K km/s for the middle transition
                trad_mid = float(words[8])
                tau_mid = float(words[7])
                line  = outfile.readline()  # this should be the next transition
                words = line.split()
                ftmp  = float(words[4])  # similarly, used to identify the transition
                while ((ftmp < fupp*(1-bw)) or (ftmp > fupp/(1-bw))):
                    line  = outfile.readline()
                    words = line.split()
                    ftmp  = float(words[4])
                upp   = float(words[-2])  # similarly, for the flux in K km/s for the upper transition
                trad_upp = float(words[8])
                tau_upp = float(words[7])
                
                # sanity check: should recover same parameters as we initially requested for this calculation
                if ((temp < 0.99*tempstouse[itemp]) or (temp > 1.01*tempstouse[itemp])):
                    print('Fail', temp, tempstouse[itemp])
                if ((dens < 0.99*denstouse[idens]) or (dens > 1.01*denstouse[idens])):
                    print('other fail')
                # could check cd here too but watch abundance ratio and also I suspect if there is an error the previous 2 checks will find it

                # here you can see how the output arrays are structured.
                # The axes are column density, volume density, and temperature in that order
                temp_results[icd,idens,itemp] = temp
                dens_results[icd,idens,itemp] = dens
                cd_results[icd,idens,itemp] = cdmol
                line1_results[icd,idens,itemp] = low
                line2_results[icd,idens,itemp] = mid
                line3_results[icd,idens,itemp] = upp
                line1_trad[icd,idens,itemp] = trad_low
                line2_trad[icd,idens,itemp] = trad_mid
                line3_trad[icd,idens,itemp] = trad_upp
                tau1[icd,idens,itemp] = tau_low
                tau2[icd,idens,itemp] = tau_mid
                tau3[icd,idens,itemp] = tau_upp
                
                count += 1
                
    outfile.close()
    return temp_results,dens_results,cd_results,line1_results,line2_results,line3_results,tau1,tau2,tau3,line1_trad,line2_trad,line3_trad


def cleanuparray(array):
    # the purpose here is to take care of the ridiculously large or small ratios caused by divide by 0.
    #  they make it difficult for the casaviewer to put contours on, so you can't see what's happening.
    # only run this one on the line ratio cubes, not any other parameters.
    array[(array > 1000.)] = 0.0 # just some sanity checks to protect against the divide by 0 problems
    array[(array < 1.e-5)] = 0.0 # the ratios I'm doing will be neither huge nor tiny, but change if necessary
    return array

#%%
# --------------------------  setup stuff ------------------------

#
# Grid boundaries
#
tmin = 3.0  # minimum kinetic temperature (K)
tmax = 80.0 # maximum kinetic temperature (K)
nmin = 50.0   # minimum H2 density (cm^-3)
nmax = 1e6   # maximum H2 density (cm^-3)
cdh2min = 1.e20 # these are the total H2 column density; convert to the molecule in question with abun_rat (below)
cdh2max = 1.e24
dolineartemp = False # default behavior is logarithmic steps in temp, but you can do linear

# Numerical parameters
#
ntemp = 20  # number of temperature steps (npoints is this +1)
ndens = 20  # number of density steps
ncd = 60     # number of column density steps. 
bw    = 0.01 # "bandwidth": free spectral range around line (only used within radex, as it IDs transition of interest)

#
# physical parameters
#
tbg   = 2.73 # background radiation temperature - usually this will be CMB
#
abun_12coh2 = 8.e-5 # copied from D Meier 2008 for 12co/H2
cd12comin = cdh2min * abun_12coh2
cd12comax = cdh2max * abun_12coh2
abun_12co13co = 40. # in this version, these two values are just sort of fiducial mid-range about which we explore.
abun_12coc18o = 320. 
fiducial_dv = 30.0 # km/s, scaling parameter again.  remember only the ratio of coldens/dv is really significant.

# here come the grids
if (dolineartemp):
    tempstouse = np.linspace(tmin, tmax, num=ntemp+1)
else:
    tempstouse = np.geomspace(tmin, tmax, num=ntemp+1)
denstouse = np.geomspace(nmin, nmax, num=ndens+1)
cdtouse = np.geomspace(cd12comin, cd12comax, num=ncd+1)
dv = fiducial_dv

# radex function call
# "radex.inp" is hardwired here in a few places, so best not change that (or change them all)
radexcall = '/Users/lyoung/software/Radex/bin/radex < radex.inp > /dev/null'
 

#%%
# ------------  Work starts here ---------------
# 12CO

start = time.time()
print('starting calculations for 12CO')
mole = '12co'
abun_rat = 1.0 # this is 12co
# Frequencies of transitions to study.  Radex uses these values to obtain Einstein A values etc.
flow = 115.3  # GHz
fmid = 230.5
fupp = 345.8

# store a record of all the parameter combinations used for every run of radex
write_input_wrapper()

# having written the input file in the correct format, do the calculations
os.system(radexcall)
print('reading output file')
temp_results,dens_results,cd12co_results,line_12co10,line_12co21,line_12co32,tau_12co10,tau_12co21,tau_12co32,trad_12co10,trad_12co21,trad_12co32 = read_radex_grid('radex.out',ntemp,ndens,ncd,abun_rat)
ratio12co2110_results = cleanuparray((line_12co21/line_12co10))
ratio12co3210_results = cleanuparray((line_12co32/line_12co10)) 
stop = time.time()
dure = stop - start
print("Run time = ",dure, "seconds")
os.system('mv radex.inp radex_12co.inp')
os.system('mv radex.out radex_12co.out')

# dump to fits files
print('writing 3D fits files')
arylist = [np.log10(temp_results), np.log10(dens_results), np.log10(cd12co_results), ratio12co2110_results, ratio12co3210_results, tau_12co10, tau_12co21, tau_12co32, trad_12co10, trad_12co21, trad_12co32]
namelist = ['logtemp_co', 'logdens_co', 'logcd12co', 'ratio12co2110', 'ratio12co3210', 'tau_12co10', 'tau_12co21', 'tau_12co32', 'trad_12co10', 'trad_12co21', 'trad_12co32']
for i in range(len(arylist)):
    hdu = fits.PrimaryHDU(arylist[i])
    hdu.writeto(namelist[i]+'.fits', overwrite=True)


#%%
# 13CO

start = time.time()
print('')
print('starting calculations for 13CO')
mole = '13co'
abun_rat = 1. / abun_12co13co  # the range of N(13CO) values covered will be N(12CO) * abun_rat
flow = 110.2
fmid = 220.4
fupp = 330.6

# store a record of all the parameter combinations used for every run of radex
write_input_wrapper()

# having written the input file in the correct format, do the calculations
os.system(radexcall)
print('reading output file')
temp_results,dens_results,cd13co_results,line_13co10,line_13co21,line_13co32,tau_13co10,tau_13co21,tau_13co32,trad_13co10,trad_13co21,trad_13co32 = read_radex_grid('radex.out',ntemp,ndens,ncd,abun_rat)
stop = time.time()
dure = stop - start
print("Run time = ",dure, "seconds")
os.system('mv radex.inp radex_13co.inp')
os.system('mv radex.out radex_13co.out')
ratio12co13co10_results = cleanuparray((line_12co10/line_13co10))
ratio12co13co21_results = cleanuparray((line_12co21/line_13co21))

# this set of 3d cubes is computed for the fiducial 13co abundance listed in the setup part above; you may not prefer that value
print('writing 3D fits files')
arylist = [ratio12co13co10_results, ratio12co13co21_results, tau_13co10, tau_13co21, tau_13co32, np.log10(cd13co_results), trad_13co10, trad_13co21, trad_13co32]
namelist = ['ratio12co13co10', 'ratio12co13co21', 'tau_13co10', 'tau_13co21', 'tau_13co32', 'logcd13co', 'trad_13co10', 'trad_13co21', 'trad_13co32']
for i in range(len(arylist)):
    hdu = fits.PrimaryHDU(arylist[i])
    hdu.writeto(namelist[i]+'.fits', overwrite=True)



#%%
# C18O

start = time.time()
print('')
print('starting calculations for C18O')
mole = 'c18o'
abun_rat = 1. / abun_12coc18o
flow = 109.8
fmid = 219.6
fupp = 329.3

# store a record of all the parameter combinations used for every run of radex
write_input_wrapper()

# having written the input file in the correct format, do the calculations
os.system(radexcall)
print('reading output file')
temp_results,dens_results,cdc18o_results,line_c18o10,line_c18o21,line_c18o32,tau_c18o10,tau_c18o21,tau_c18o32,trad_c18o10,trad_c18o21,trad_c18o32 = read_radex_grid('radex.out',ntemp,ndens,ncd,abun_rat)
stop = time.time()
dure = stop - start
print("Run time = ",dure, "seconds")
os.system('mv radex.inp radex_c18o.inp')
os.system('mv radex.out radex_c18o.out')
ratio12coc18o10_results = cleanuparray((line_12co10/line_c18o10))
ratio13coc18o10_results = cleanuparray((line_13co10/line_c18o10))

# this set of 3d cubes is computed for the fiducial c18o abundance listed in the setup part above; you may not prefer that value
print('writing 3D fits files')
arylist = [ratio12coc18o10_results, ratio13coc18o10_results, tau_c18o10, tau_c18o21, tau_c18o32, np.log10(cdc18o_results), trad_c18o10, trad_c18o21, trad_c18o32]
namelist = ['ratio12coc18o', 'ratio13coc18o10', 'tau_c18o10', 'tau_c18o21', 'tau_c18o32', 'logcdc18o', 'trad_c18o10', 'trad_c18o21', 'trad_c18o32']
for i in range(len(arylist)):
    hdu = fits.PrimaryHDU(arylist[i])
    hdu.writeto(namelist[i]+'.fits', overwrite=True)



#%%

# Now using 3d cubes to construct the 5d results datasets that are used in the uncertainty analysis.
# 5d cubes have axes: N(H2), volume density, temperature, 12CO/13CO abundance ratio, 13CO/C18O abundance ratio

# first figuring out what 12/13 ratios are accessible and how to compute them
stepdelta = np.median(np.diff(np.log10(cd12co_results[:,0,0])))  # by construction this is also the step in [12co/13co] ratios
indices = np.arange(-ncd,ncd+1,1)
logcando_1213 = np.log10(abun_12co13co) + indices*stepdelta
cando_1213 = 10.**logcando_1213
min1213 = 3.
max1213 = 400.
useme1213 = (cando_1213 > min1213) * (cando_1213 < max1213)
num1213s = np.sum(useme1213)
# same thing now for the 12/18 ratios
logcando_1318 = np.log10(abun_12coc18o/abun_12co13co) + indices*stepdelta
cando_1318 = 10.**logcando_1318
min1318 = 2.
max1318 = 30.
useme1318 = (cando_1318 > min1318) * (cando_1318 < max1318)
num1318s = np.sum(useme1318)


logdens_5d = np.zeros((ncd+1,ndens+1,ntemp+1,num1213s,num1318s))  # here's the axis order info
logtemp_5d = np.zeros_like(logdens_5d)
logcd12co_5d = np.zeros_like(logdens_5d)
logcd13co_5d = np.zeros_like(logdens_5d)
tau12co10_5d = np.zeros_like(logdens_5d)
tau13co10_5d = np.zeros_like(logdens_5d)
tauc18o10_5d = np.zeros_like(logdens_5d)
ratio12co2110_5d = np.zeros_like(logdens_5d)
ratio12co13co10_5d = np.zeros_like(logdens_5d)
ratio12co13co21_5d = np.zeros_like(logdens_5d)
ratio12coc18o10_5d = np.zeros_like(logdens_5d)
ratio13coc18o10_5d = np.zeros_like(logdens_5d)
logabun_12co13co_5d = np.zeros_like(logdens_5d)
logabun_13coc18o_5d = np.zeros_like(logdens_5d)
line12co10_5d = np.zeros_like(logdens_5d)  # these must be the raw fluxes (K km/s)  = trad * deltav * 1.065
line13co10_5d = np.zeros_like(logdens_5d)
linec18o10_5d = np.zeros_like(logdens_5d)
trad12co10_5d = np.zeros_like(logdens_5d)  # radiation temperature, K
trad13co10_5d = np.zeros_like(logdens_5d)
tradc18o10_5d = np.zeros_like(logdens_5d)

for l in range(num1213s):
    for n in range(num1318s):
        # loop over the abundance ratios
        # in this loop can do all the stuff that is the same for all 12co/13co and 13co/c18o abundance ratios
        # other stuff you may want at some point: tau of the other 12co lines, 12co32/10, etc
        logdens_5d[:,:,:,l,n] = np.log10(dens_results[:,:,:])
        logtemp_5d[:,:,:,l,n] = np.log10(temp_results[:,:,:])
        logcd12co_5d[:,:,:,l,n] = np.log10(cd12co_results[:,:,:])
        tau12co10_5d[:,:,:,l,n] = tau_12co10[:,:,:]
        ratio12co2110_5d[:,:,:,l,n] = ratio12co2110_results[:,:,:]
        line12co10_5d[:,:,:,l,n] = line_12co10[:,:,:] # K km/s
        trad12co10_5d[:,:,:,l,n] = trad_12co10[:,:,:] # K


 
for i in range(ncd+1):
    # now for the stuff that depends on the 13co abundance & 12co column density BUT NOT c18o abundance
    # matchup describes how an abundance ratio connects the i and l indices.
    # loop over the matchup indices, see if we want to keep them
    for n in range(num1318s):
        for l in range(num1213s):
            for matchup in range(ncd+1):
                testrat = cd12co_results[i,0,0]/cd13co_results[matchup,0,0]  # the abundance ratio corresponding to this set of cd12co and cd13co
                if (0.99*cando_1213[useme1213][l] < testrat < 1.01*cando_1213[useme1213][l]):
                    # you may have to fix this abundance ratio calculation by hand so the values are monotonic
                    # i.e. the interpolator (in the sampling script) will not want zeros in the abundance ratio 
                    # maybe use nearest-neighbor extrapolation at the ends of the ranges where I don't have
                    # "real" ratio calculations because of the overhang
                    logabun_12co13co_5d[i,:,:,l,n] = np.log10(cd12co_results[i,:,:]/cd13co_results[matchup,:,:])
                    ratio12co13co10_5d[i,:,:,l,n] = line_12co10[i,:,:]/line_13co10[matchup,:,:]
                    ratio12co13co21_5d[i,:,:,l,n] = line_12co21[i,:,:]/line_13co21[matchup,:,:]
                    tau13co10_5d[i,:,:,l,n] = tau_13co10[matchup,:,:]
                    line13co10_5d[i,:,:,l,n] = line_13co10[matchup,:,:]
                    trad13co10_5d[i,:,:,l,n] = trad_13co10[matchup,:,:]

logcd13co_5d = logcd12co_5d - logabun_12co13co_5d

# remaining items should be the ones related to c18o, which depends on both 12co/13co and 13co/c18o.    
for i in range(ncd+1):
    # matchup describes how an abundance ratio connects the i and n indices.
    # loop over the matchup indices, see if we want to keep them
    for l in range(num1213s):
        for n in range(num1318s):
            for matchup in range(ncd+1):
                testrat = 10.**(logcd13co_5d[i,0,0,l,n]) / cdc18o_results[matchup,0,0] # the 13co/c18o abundance ratio corresponding to this set of cd12co and cd13co and cdc18o
                if (0.99*cando_1318[useme1318][n] < testrat < 1.01*cando_1318[useme1318][n]):
                    # you may have to fix this abundance ratio calculation by hand so the values are monotonic
                    # i.e. the interpolator will not want zeros in the abundance ratio 
                    # maybe use nearest-neighbor extrapolation at the ends of the ranges where I don't have
                    # "real" ratio calculations because of the overhang
                    logabun_13coc18o_5d[i,:,:,l,n] = logcd13co_5d[i,:,:,l,n] - np.log10(cdc18o_results[matchup,:,:])
                    ratio12coc18o10_5d[i,:,:,l,n] = line_12co10[i,:,:]/line_c18o10[matchup,:,:]
                    tauc18o10_5d[i,:,:,l,n] = tau_c18o10[matchup,:,:]
                    linec18o10_5d[i,:,:,l,n] = line_c18o10[matchup,:,:]
                    tradc18o10_5d[i,:,:,l,n] = trad_c18o10[matchup,:,:]
                
logcdc18o_5d = logcd13co_5d - logabun_13coc18o_5d
ratio13coc18o10_5d = ratio12coc18o10_5d / ratio12co13co10_5d  
            
arylist = [logdens_5d, logtemp_5d, logcd12co_5d, logabun_12co13co_5d, logabun_13coc18o_5d, tau12co10_5d, \
           tau13co10_5d, tauc18o10_5d, ratio12co2110_5d, ratio12co13co10_5d, ratio12coc18o10_5d, \
               ratio13coc18o10_5d, ratio12co13co21_5d, line12co10_5d, line13co10_5d, linec18o10_5d, trad12co10_5d, trad13co10_5d, tradc18o10_5d]
namelist = ['logdens_5d', 'logtemp_5d', 'logcd12co_5d', 'logabun_12co13co_5d', 'logabun_13coc18o_5d', 'tau12co10_5d', \
            'tau13co10_5d', 'tauc18o10_5d', 'ratio12co2110_5d', 'ratio12co13co10_5d', 'ratio12coc18o10_5d', \
                'ratio13coc18o10_5d', 'ratio12co13co21_5d', 'line12co10_5d', 'line13co10_5d', 'linec18o10_5d', 'trad12co10_5d', 'trad13co10_5d', 'tradc18o10_5d']
for i in range(len(arylist)):
    hdu = fits.PrimaryHDU(arylist[i])
    hdu.writeto(namelist[i]+'.fits', overwrite=True)
