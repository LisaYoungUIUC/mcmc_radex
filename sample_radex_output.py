#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 Feb 2024 LMY

This is the code used by Young et al (2022) ApJ 933, 90 for the work described in Appendix B.
(Note to me: descended from bestfit_co_v4c.py)

PURPOSE: Following on the radex runs in radex_makecubes.py, find the best model conditions 
that reproduce observed CO line ratios.
Users will need to edit the line ratios in the setup section at the comment "User edit these values!"
There are also a couple of sanity check calls to radex itself so users should edit that radexcall 
string to match their setup.

Instead of emcee, Cappellari likes his sampler called adamet; see examples in
/opt/anaconda3/lib/python3.7/site-packages/adamet/examples.

The workflow for using these two scripts will be
1. Edit radex_makecubes.py; check the grid boundaries and step sizes to verify the parameter ranges of interest.
2. Run radex_makecubes.py.  The output FITS files may take up several GB on disk, depending on the size of the grids.
3. Edit sample_radex_output.py; adjust values of the measurements (the line ratios and 12CO brightness
    temperature) to match your data.
4. Run sample_radex_output.py and enjoy the plots.



WISH LIST:
    - parallel processing
    - there may be some subtleties about whether you do the likelihood calculation on the ratios or the
      individual line intensities themselves.  similarly on the ratios or the logs of the ratios.
    - try different interpolation schemes to see if that speeds it up.  Or do ONE
      massive interpolation in the original radex cubes and thereafter just read off the nearest
      grid point, rather than interpolating every time?
      
TO DO:

    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interpn
import emcee
import corner
from IPython.display import display, Math
import datetime
from matplotlib.lines import Line2D
# from adamet.adamet import adamet


def interp_radexgrid(params):
    # feed this a set of physical parameters and it interpolates to find the radex outputs 
    # produced by that set of parameters.
    # note this assumes the radex grid cubes are regularly spaced.  they almost are, within usu 10^-5
    allaxisvals = (logcd_axis_vals,logdens_axis_vals,logtemp_axis_vals,logabun1213_axis_vals,logabun1318_axis_vals)
    ans_12coc18o10 = interpn(allaxisvals, ratio_12coc18o10, params) # params must be in order coldens, n, T, [12/13]
    ans_12co13co10 = interpn(allaxisvals, ratio_12co13co10, params) 
    ans_13coc18o10 = interpn(allaxisvals, ratio_13coc18o10, params)
    ans_12co2110 = interpn(allaxisvals, ratio_12co2110, params) 
    ans_12co13co21 = interpn(allaxisvals, ratio_12co13co21, params) 
    ans_trad12co10 = interpn(allaxisvals, trad12co10, params)
    model = (ans_12coc18o10, ans_12co13co10, ans_13coc18o10, ans_12co2110, ans_12co13co21, ans_trad12co10)
    return model


def mycalcfn(obs, errs, rdx_outputs):   
    # this one contains the actual code calculating the log of the likelihood
    # obs and errs are global; rdx_outputs is a list containing either the values of the line
    # ratios produced by interp_radexgrid, or the big 5d radex output results cubes themselves
    lnl = np.zeros_like(rdx_outputs[0])
    for i in range(ndata-1): # standard treatment for the parameters that are not lower limits
        lnl += (obs[i] - rdx_outputs[i])**2 / errs[i]**2
    # special treatment for the 12CO brightness temperature to account for the fact that
    # observed brightness temps may be lower than calculated, due to the beam filling factor.
    # so observed TB is allowed to be lower than predicted, but not higher
    tdiff = obs[5] - rdx_outputs[5]
    badness = tdiff**2 / errs[5]**2
    if (type(rdx_outputs[0])==float):
        # it's a scalar.  add some badness if the model prediction is below the observation, otherwise do nothing.
        if (tdiff > 0.):
            lnl += badness
    else:
        # it's a 5d grid so we have to deal with conditions that work and conditions that don't work
        lnl[(tdiff <= 0.)] += 0.  # the ok values where model is above the observation
        lnl[(tdiff > 0.)] += badness[(tdiff > 0.)]  # standard Gaussian treatment where the model is below the observation
    lnl *= -0.5
    return lnl

def log_likelihood(theta):
    # use this version in the MCMC (or other sampling) algorithm
    rdx_outputs = interp_radexgrid(theta)
    lnlike = mycalcfn(obs, errs, rdx_outputs)
    return lnlike

def log_prior(theta):
    logcd, logn, logt, logab1213, logab1318 = theta
    # flat priors on the logs of all these parameters, within our grid range
    if (min(logcd_axis_vals) <= logcd <= max(logcd_axis_vals)) and (min(logdens_axis_vals) <= logn <= max(logdens_axis_vals)) \
        and (min(logtemp_axis_vals) <= logt <= max(logtemp_axis_vals)) \
        and (min(logabun1213_axis_vals) <= logab1213 <= max(logabun1213_axis_vals)) \
        and (min(logabun1318_axis_vals) <= logab1318 <= max(logabun1318_axis_vals)):
            return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return lp, -np.inf
    return lp + ll, lp
  
    
def write_input(filename,tkin,nh2,cd12co,abun1213,abun1318):
    # for the sanity check
    tbg = 2.73
    flow = 100
    fupp = 350
    bw = 0.01
    dv = 30.0
    cd13co = cd12co/abun1213
    cdc18o = cd13co/abun1318
    infile = open(filename+'.inp','w')
    # 12co run
    infile.write('12co.dat\n')  # record where we're getting the Einstein As and the collision probabilities etc from
    infile.write(filename+'.out\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')  # frequencies of the transitions going into the ratio
    infile.write(str(tkin)+'\n')   # kinetic temperature
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')  # density (n, not log n)
    infile.write(str(tbg)+'\n')
    infile.write(str(cd12co)+'\n')  # column density
    infile.write(str(dv)+'\n')   #  line width
    infile.write('1\n')
    # 13co run
    infile.write('13co.dat\n')  # record where we're getting the Einstein As and the collision probabilities etc from
    infile.write(filename+'.out\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')  # frequencies of the transitions going into the ratio
    infile.write(str(tkin)+'\n')   # kinetic temperature
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')  # density (n, not log n)
    infile.write(str(tbg)+'\n')
    infile.write(str(cd13co)+'\n')  # column density
    infile.write(str(dv)+'\n')   #  line width
    infile.write('1\n')
    # c18o run
    infile.write('c18o.dat\n')  # record where we're getting the Einstein As and the collision probabilities etc from
    infile.write(filename+'.out\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')  # frequencies of the transitions going into the ratio
    infile.write(str(tkin)+'\n')   # kinetic temperature
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')  # density (n, not log n)
    infile.write(str(tbg)+'\n')
    infile.write(str(cdc18o)+'\n')  # column density
    infile.write(str(dv)+'\n')   #  line width
    infile.write('0\n')
    infile.close()
    
    
#%%
# -------------  setup ---------------
    
doadamet = False # this one controls whether I try Cappellari's adaptive metropolis or just stick with emcee
radexcall = '/Users/lyoung/software/Radex/bin/radex < sanitycheck2.inp > /dev/null'

# get the output of the radex models
badval = -100.
ratio_12coc18o10 = fits.open('ratio12coc18o10_5d.fits')[0].data
ratio_12coc18o10[(ratio_12coc18o10 == 0.)] = badval # these were places where radex barfed, so make them very bad.
ratio_12coc18o10[np.isnan(ratio_12coc18o10)] = badval
ratio_12co13co10 = fits.open('ratio12co13co10_5d.fits')[0].data
ratio_12co13co10[(ratio_12co13co10 == 0.)] = badval
ratio_12co13co10[np.logical_not(np.isfinite(ratio_12co13co10))] = badval
ratio_13coc18o10 = fits.open('ratio13coc18o10_5d.fits')[0].data
ratio_13coc18o10[(ratio_13coc18o10 == 0.)] = badval # these were places where radex barfed, so make them very bad.
ratio_13coc18o10[np.isnan(ratio_13coc18o10)] = badval
ratio_12co2110 = fits.open('ratio12co2110_5d.fits')[0].data
ratio_12co2110[(ratio_12co2110 == 0.)] = badval
ratio_12co2110[np.isnan(ratio_12co2110)] = badval
ratio_12co13co21 = fits.open('ratio12co13co21_5d.fits')[0].data
ratio_12co13co21[(ratio_12co13co21 == 0.)] = badval
ratio_12co13co21[np.isnan(ratio_12co13co21)] = badval
tau12co10 = fits.open('tau12co10_5d.fits')[0].data
tau13co10 = fits.open('tau13co10_5d.fits')[0].data
tauc18o10 = fits.open('tauc18o10_5d.fits')[0].data
line12co10 = fits.open('line12co10_5d.fits')[0].data
line13co10 = fits.open('line13co10_5d.fits')[0].data
linec18o10 = fits.open('linec18o10_5d.fits')[0].data
trad12co10 = fits.open('trad12co10_5d.fits')[0].data
trad13co10 = fits.open('trad13co10_5d.fits')[0].data
tradc18o10 = fits.open('tradc18o10_5d.fits')[0].data

logdens = fits.open('logdens_5d.fits')[0].data
logtemp = fits.open('logtemp_5d.fits')[0].data
logcd12co = fits.open('logcd12co_5d.fits')[0].data  # this is now H2 column density and should be same for every species
logabun1213 = fits.open('logabun_12co13co_5d.fits')[0].data
logabun1318 = fits.open('logabun_13coc18o_5d.fits')[0].data
logdens_axis_vals = logdens[0,:,0,0,0]
logtemp_axis_vals = logtemp[0,0,:,0,0]
logcd_axis_vals = logcd12co[:,0,0,0,0]
blah = int(np.shape(logdens)[0]/2.) # picking the central pixel on the column density axis
logabun1213_axis_vals = logabun1213[blah,10,10,:,5]  # in the middle to avoid the goofy matchup range issues
logabun1318_axis_vals = logabun1318[blah,10,10,5,:]  # in the middle to avoid the goofy matchup range issues


# measured values. 
# ordering of these data is: 12coc18o10, 12co13co10, 13coc18o10, 12co2110, 12co13co21, 12co10TB
# co10TB is the peak Tb observed in the 12co1-0 cube at high resolution.
#  the deal here will be: RADEX predicted radiation temperatures must be >= tblimit.  filling factors <1 will make the observed values < theoretical values.
# User edit these values!
inner = np.array([17.5, 3.5, 5.0, 0.6, 4.6, 1.2])  
ring = np.array([17.5, 3.5, 5.0, 0.55, 4.6, 6.1])
outer = np.array([28.0, 3.5, 8.0, 0.5, 4.6, 3.])
# here's where you can control which set of values you want to study
modelme = 2 # 1, 2, or 3 for the inner disk, ring, or outer disk 
if (modelme == 1):
    obs = inner 
    plotname = 'inner'+str(datetime.date.today())+'.pdf'
    dumpfilename = 'inner_1318range.txt'
    cornerlabel = 'Nucleus'
if (modelme == 2):
    obs = ring
    plotname = 'ring'+str(datetime.date.today())+'.pdf'
    dumpfilename = 'ring_1318range.txt'
    cornerlabel = 'Ring'
if (modelme == 3):
    obs = outer
    plotname = 'outer'+str(datetime.date.today())+'.pdf'
    dumpfilename = 'outer_1318range.txt'
    cornerlabel = 'Outer Disk'
errs = np.array([obs[0]*0.1, obs[1]*0.1, obs[2]*0.1, obs[3]*0.2, 0.7, obs[5]*0.05])
# uncertainties on all of those can be 10%, except 12co2110 should probably be 20% as it's from carma

#%%

# emcee requires us to decide on a starting position for the walkers.

# here we are creating a 5d array that contains the log of the likelihood 
# of each set of physical conditions in the radex grids. 
ndata = len(obs) # number of constraints (measurements)
rdx_outputs = [ratio_12coc18o10, ratio_12co13co10, ratio_13coc18o10, ratio_12co2110, ratio_12co13co21, trad12co10]
# must preserve the order of the measurements here for the code to work properly
lnl = mycalcfn(obs, errs, rdx_outputs)

# find the gridpoint with the maximum likelihood.  this is a reasonable place to start the walkers.
# so will serve for our initial guess.
# you could also get a little more sophisticated with an optimizer, maybe not necessary
blah = np.unravel_index(np.argmax(lnl), lnl.shape)
init_logn = logdens[blah]
init_logt = logtemp[blah]
init_logcd = logcd12co[blah]
init_log1213 = logabun1213[blah]
init_log1318 = logabun1318[blah]
init_ff = obs[5]/trad12co10[blah]

print('')
print('Data 12co/c18o1-0, 12co/13co1-0, 13co/c18o1-0, 12co2-1/1-0, 12co/13co2-1, 12co1-0 TB(K)')
print('obs: ', obs)
print('')
print('Starting point for the walkers is the gridpoint with max likelihood.')
print('Ratios I chose:  %.2f   %.2f   %.2f   %.2f  %.2f'%(ratio_12coc18o10[blah], ratio_12co13co10[blah], ratio_13coc18o10[blah], ratio_12co2110[blah], ratio_12co13co21[blah]))
print('Filling factor I chose:', init_ff)
print('At log dens, temp, cd, 12/13,  13/18: %.2f  %.2f  %.2f  %.3f  %.3f'%(init_logn, init_logt, init_logcd, init_log1213, init_log1318))
print('and tau12co10, tau13co10, tauc18o10:  %.2f  %.2f  %.2f'%(tau12co10[blah], tau13co10[blah], tauc18o10[blah]))


#%%
#    ------------------- sampling: emcee unless doadamet says otherwise --------------------

# primary output of this section of code is flat_samples, which is an array with a shape (#samples, ndim) and
# the statistical analysis is done on it to obtain the posterior probability distributions.

ndim = 5 # 5 parameters at the moment: cd (aka N), n, T, 12/13, 12/18
labels = ["log N", "log n", "log T", "log 12/13", "log 13/18"]

if (doadamet):
    # use Michele's thing
    pos = np.array([init_logcd,init_logn,init_logt,init_log1213,init_log1318])
    sigpars0 = np.array([0.3, 0.5, 0.5, 0.1, 0.05]) # these are guesses about the uncertainties in the model parameters.  (they are not uncertainties in the measurements!)
    bounds = np.array([[logcd_axis_vals[0],logdens_axis_vals[0],logtemp_axis_vals[0],logabun1213_axis_vals[0],logabun1318_axis_vals[0]],\
                        [logcd_axis_vals[-1],logdens_axis_vals[-1],logtemp_axis_vals[-1],logabun1213_axis_vals[-1],logabun1318_axis_vals[-1]]])
    nstep=50000
    flat_samples, lnprob = adamet(log_likelihood, pos, sigpars0, bounds, nstep, labels=labels, nprint=100, quiet=False, fignum=None, plot=True, labels_scaling=1, seed=None, args=(), kwargs={})
else:
    # emcee
    nwalkers = 10
    pos = np.array([init_logcd,init_logn,init_logt,init_log1213,init_log1318]) + 0.1 * np.random.randn(nwalkers,ndim)
    # with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability) # , pool=pool)
    sampler.run_mcmc(pos, 10000, progress=True)
    samples = sampler.get_chain() # without flattening, the individual walkers are kept separate/identifiable.
    # plot the samples for a visual check on whether they are converging
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    # return a representative subset of the samples for further analysis, corner plots etc.  here the individual walkers are all mashed together.
    flat_samples = sampler.get_chain(discard=100, thin=9, flat=True)  # I think this means it is using every "thin"th step for plotting and analysis?
    print(flat_samples.shape)

#%% 
#  Diagnostic plots & other output

allaxisvals = (logcd_axis_vals,logdens_axis_vals,logtemp_axis_vals,logabun1213_axis_vals,logabun1318_axis_vals)
sampled_tau1310 = np.log10(interpn(allaxisvals, tau13co10, flat_samples))
sampled_tau1210 = np.log10(interpn(allaxisvals, tau12co10, flat_samples))
sampled_12coc18o10 = interpn(allaxisvals, ratio_12coc18o10, flat_samples)
sampled_12co13co10 = interpn(allaxisvals, ratio_12co13co10, flat_samples)
sampled_12co13co21 = interpn(allaxisvals, ratio_12co13co21, flat_samples)
sampled_13coc18o10 = interpn(allaxisvals, ratio_13coc18o10, flat_samples)
sampled_12co2110 = interpn(allaxisvals, ratio_12co2110, flat_samples)
sampled_line1210 = interpn(allaxisvals, line12co10, flat_samples)
sampled_line1310 = interpn(allaxisvals, line13co10, flat_samples)
sampled_line1810 = interpn(allaxisvals, linec18o10, flat_samples)
sampled_N12 = flat_samples[:,0]
sampled_N13 = sampled_N12 - flat_samples[:,3]  # recall the flat_samples stores the logs of everything
sampled_N18 = sampled_N13 - flat_samples[:,4]
sampled_abun1213 = flat_samples[:,3]
sampled_abun1318 = flat_samples[:,4]

stufftocorner = np.concatenate((flat_samples, np.expand_dims(sampled_tau1310,axis=1)), axis=1) # tacking tau on as a "physical parameter" even tho not independent of the others
observables = np.concatenate((np.expand_dims(sampled_12co13co10,axis=1), np.expand_dims(sampled_12coc18o10,axis=1), \
                                np.expand_dims(sampled_13coc18o10,axis=1), np.expand_dims(sampled_12co2110,axis=1), \
                                np.expand_dims(sampled_12co13co21,axis=1)), axis=1)
rawintensities = np.concatenate((np.expand_dims(sampled_line1210,axis=1), np.expand_dims(sampled_line1310,axis=1), \
                                np.expand_dims(sampled_line1810,axis=1)), axis=1)
newlabels = ['log N','log n','log T','log [12/13]','log [13/18]','log $\\tau_{13}$']
obslabels = ['12co/13co 1-0','12co/c18o','13co/c18o','12co21/10','12co/13co 2-1']
rawlabels = ['12co 1-0', '13co 1-0', 'c18o 1-0']


# Primary output figure is this big corner plot showing the probabilities of all the physical parameters.

fig = corner.corner(stufftocorner, labels=newlabels, quantiles=[0.16,0.84], show_titles=True, top_ticks=True, labelpad=0.08,\
                    range=[(min(logcd_axis_vals),20.),(min(logdens_axis_vals),max(logdens_axis_vals)),(min(logtemp_axis_vals),max(logtemp_axis_vals)),(0.5,1.8),(0.6,1.15),(-1,1)]) #  truths=[m_true, b_true, np.log(f_true)])
blah = np.percentile(flat_samples[:,4], [16,84])
# write those results on 13/18 to a file where I can read them and use for the orange & blue marker lines
dumpfile = open(dumpfilename, 'w')
dumpfile.write("# {:%Y-%b-%d %H:%M:%S} by bestfit_co_v4b.py\n".format(datetime.datetime.now()))
dumpfile.write('# 16% and 84% percentile values \n')
dumpfile.write('%7.5e %7.5e'%(blah[0],blah[1]))
dumpfile.close()
fig.subplots_adjust(bottom=0.1, top=0.9)
fig.text(0.65,0.65, cornerlabel+'\n'+ \
                 r'$^{12}$CO(1-0)/$^{13}$CO(1-0) = %.2f $\pm$ %.2f'%(obs[1],errs[1])+'\n'+ \
                  r'$^{13}$CO(1-0)/C$^{18}$O(1-0) = %.2f $\pm$ %.2f'%(obs[2],errs[2])+'\n'+ \
                  r'$^{12}$CO(2-1)/$^{12}$CO(1-0) = %.2f $\pm$ %.2f'%(obs[3],errs[3])+'\n'+ \
                  r'$^{12}$CO(2-1)/$^{13}$CO(2-1) = %.2f $\pm$ %.2f'%(obs[4],errs[4])+'\n'+ \
                      r'T$_\mathrm{B}(^{12}$CO 1-0) = %.2f $\pm$ %.2f K'%(obs[5],errs[5]), \
                  fontsize='x-large')
fig.savefig(plotname)

# Next comes a sanity check figure showing distributions of the observed line ratios.
# Purpose of this plot is just to verify that everything worked properly.
fig2 = corner.corner(observables, labels=obslabels, quantiles=[0.16,0.84], show_titles=True, top_ticks=True) #  truths=[m_true, b_true, np.log(f_true)])
# notes on the corner plots - you can also specify a range (e.g. taus cluster down at bottom end of range)
obs_reorder = np.array([obs[1],obs[0],obs[2],obs[3],obs[4]]) # I messed up the order when I designed the corner plot
err_reorder = np.array([errs[1],errs[0],errs[2],errs[3],errs[4]]) # I messed up the order when I designed the corner plot
# 2024 having trouble running overplot_lines...?
# corner.overplot_lines(fig2, obs_reorder, color='orange')
# corner.overplot_lines(fig2, obs_reorder+err_reorder, color='orange', linestyle=':')
# corner.overplot_lines(fig2, obs_reorder-err_reorder, color='orange', linestyle=':')
fig2.savefig('obs_'+plotname)

# Another sanity check figure.
fig3 = corner.corner(rawintensities, labels=rawlabels, quantiles=[0.16,0.84], show_titles=True, top_ticks=True, range=[(0,400),(0,100),(0,20)])

sanitycheck = np.zeros((ndim)) # this is going to hold a list of parameters to use in a quick sanity check
for i in range(len(newlabels)):
    mcmc = np.percentile(stufftocorner[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], newlabels[i])
    display(Math(txt))
    center = 10.**mcmc[1]
    print('%.2e + %.2e - %.2e'%(center, 10.**mcmc[2]-center, center-10.**mcmc[0]))
    if (i in range(ndim)): # only need this if clause because I tacked the tau13co10 onto the list of stuff to corner
        sanitycheck[i] = 10.**mcmc[1]  # I'm using the logs of the parameters but radex wants the non-log versions
        
# what kinds of line ratios are produced?
print('')
print('Predicted line ratios are as follows.')
print('Observed values should fall well within these ranges if everything went well.')
for i in range(len(obslabels)):
    pred = np.percentile(observables[:, i], [2.5, 97.5])
    txt = r"\mathrm{{{2}~in~}} ({0:.3f}, {1:.3f}) ~cf~obs~ {3:.3f}"
    txt = txt.format(pred[0], pred[1], obslabels[i], obs_reorder[i])
    display(Math(txt))
    
# what kinds of column densities are produced?
print('')
print('Predicted column densities are')
blah = np.percentile(sampled_N12, [16,84])
print('log N12 in ',blah)
blah = np.percentile(sampled_N13, [16,84])
print('log N13 in ',blah)
blah = np.percentile(sampled_N18, [16,84])
print('log N18 in ',blah)
blah = np.percentile(sampled_tau1210, [16,84])
print('log tau12co10 in ',blah)

# diagnostic plots: do I understand how this all works?
# tau ratios mostly just look like assumed abundance ratios tho at low densities there seem to be some excitation effects
# such that in general the rarer species has *higher* tau than you'd expect just from abundance ratio
# plt.figure()
# plt.plot(flat_samples[:,1], (sampled_tau1210-sampled_tau1310), '.', alpha=0.1, label='12/13')
# plt.plot(flat_samples[:,1], (sampled_tau1210-sampled_tau1810), '.', alpha=0.1, label='12/18')
# plt.plot(flat_samples[:,1], (sampled_tau1310-sampled_tau1810), '.', alpha=0.1, label='13/18')
# fiducial lines won't be useful anymore if you're allowing the abundances to vary
# plt.axhline(np.log10(60.)) # assumed 12co/13co
# plt.axhline(np.log10(600.)) # assumed 12co/c18o
# plt.axhline(np.log10(600./60.)) # assumed 12co/c18o
# plt.legend()
# plt.xlabel('log n')
# plt.ylabel('tau ratios')


# this always throws an exception so you probably want to run it last or even skip
# tau = sampler.get_autocorr_time()
# print(tau)


# sanity check on output
# current setup: order of values passed in the variable sanitycheck is N(12co), n, T, [12/13], [13/18]
# but the subroutine wants T, n, N12CO, [12/13], [13/18]
# you can look at 12co/13co10, 12co21/10, 12co/13co21 tho you'll have to do those by hand yourself out of the
# klutzy radex output file
# beware: due to funky correlations between parameters, the max likelihoods in the individual parameters do not necessarily correspond to the max likelihood in the whole system
write_input('sanitycheck2', sanitycheck[2], sanitycheck[1], sanitycheck[0], sanitycheck[3], sanitycheck[4])
os.system(radexcall)
# here's a different set of values, eyeballed off the corner plots
write_input('sanitycheck3', 10., 1000., 10.**17.8, 10., 5.8)
os.system(radexcall)



#%%

# reference info for paper
print('This grid spans values of')
gridlist = [logcd_axis_vals, logdens_axis_vals, logtemp_axis_vals, logabun1213_axis_vals, logabun1318_axis_vals]
gridlabellist = ['12co column density','n(H2)','T(kin)','12/13','13/18']
for i in range(len(gridlist)):
    blah = 10.**gridlist[i]
    step = np.median(np.diff(gridlist[i]))
    print('%s from %.3e to %.3e in steps of %.3f dex.'%(gridlabellist[i],blah[0],blah[-1],step))


#%%
#    ---------------- sanity check n vs T plots with observations as contours ------------------

# 1. select the correct N, [12/13], and [13/18] values to use
# currently just grabbing the closest grid value, but could possibly do an interpolation if desired
plotme_cd = np.percentile(sampled_N12, 50.)
plotme_abun1213 = np.percentile(sampled_abun1213, 50.)
plotme_abun1318 = np.percentile(sampled_abun1318, 50.)
index_cd = np.argmin(np.abs(logcd_axis_vals - plotme_cd))
index_1213 = np.argmin(np.abs(logabun1213_axis_vals - plotme_abun1213))
index_1318 = np.argmin(np.abs(logabun1318_axis_vals - plotme_abun1318))

# 2. extract 2d (n, T) arrays for the chosen N, 12/13, 13/18 values
ratio_12co13co10_2d = ratio_12co13co10[index_cd,:,:,index_1213,index_1318]
ratio_12co13co21_2d = ratio_12co13co21[index_cd,:,:,index_1213,index_1318]
ratio_13coc18o10_2d = ratio_13coc18o10[index_cd,:,:,index_1213,index_1318]
ratio_12co2110_2d = ratio_12co2110[index_cd,:,:,index_1213,index_1318]
trad_12co10_2d = trad12co10[index_cd,:,:,index_1213,index_1318]
trad_13co10_2d = trad13co10[index_cd,:,:,index_1213,index_1318]
trad_c18o10_2d = tradc18o10[index_cd,:,:,index_1213,index_1318]
logdens_2d = logdens[index_cd,:,:,index_1213,index_1318]
logtemp_2d = logtemp[index_cd,:,:,index_1213,index_1318]
nstep = np.median(np.diff(logdens_axis_vals))
tstep = np.median(np.diff(logtemp_axis_vals))
extent = [np.min(logtemp_axis_vals)-tstep, np.max(logtemp_axis_vals)+tstep, np.min(logdens_axis_vals)-nstep, np.max(logdens_axis_vals)+nstep]

# 3. plot 2d arrays and contours
# dens and temp were to check orientations on arrays; looks good, don't need anymore
# plt.figure('dens')
# plt.imshow(logdens_2d, origin='lower', extent=extent, aspect='auto')
# plt.xlabel('log T (K)')
# plt.ylabel('log n (cm$^{-3}$)')
# plt.figure('temp')
# plt.imshow(logtemp_2d, origin='lower', extent=extent, aspect='auto')
# plt.xlabel('log T (K)')
# plt.ylabel('log n (cm$^{-3}$)')
plt.figure()
# plt.imshow(ratio_12co13co21_2d, origin='lower', extent=extent, aspect='auto', cmap='Greys')
# contourf just draws diagonal lines thru pixels so doesn't look very different from my pixellated cband image
plt.contour(ratio_12co13co10_2d, origin='lower', extent=extent, levels=[obs[1]], colors='blue')
cband = ((ratio_12co13co10_2d <= (obs[1]+errs[1])) * (ratio_12co13co10_2d >= (obs[1]-errs[1]))).astype(float)
plt.imshow(cband, origin='lower', extent=extent, aspect='auto', cmap='Greys', alpha=0.5)
# plt.contourf(cband, origin='lower', extent=extent, levels=[0.5,1.5], cmap='viridis')
#
plt.contour(ratio_12co2110_2d, origin='lower', extent=extent, levels=[obs[3]], colors='C1')
cband = ((ratio_12co2110_2d <= (obs[3]+errs[3])) * (ratio_12co2110_2d >= (obs[3]-errs[3]))).astype(float)
plt.imshow(cband, origin='lower', extent=extent, aspect='auto', cmap='Greys', alpha=0.5)
#
plt.contour(ratio_12co13co21_2d, origin='lower', extent=extent, levels=[obs[4]], colors='C2')
cband = ((ratio_12co13co21_2d <= (obs[4]+errs[4])) * (ratio_12co13co21_2d >= (obs[4]-errs[4]))).astype(float)
plt.imshow(cband, origin='lower', extent=extent, aspect='auto', cmap='Greys', alpha=0.5)
#
plt.contour(ratio_13coc18o10_2d, origin='lower', extent=extent, levels=[obs[2]], colors='magenta')
cband = ((ratio_13coc18o10_2d <= (obs[2]+errs[2])) * (ratio_13coc18o10_2d >= (obs[2]-errs[2]))).astype(float)
plt.imshow(cband, origin='lower', extent=extent, aspect='auto', cmap='Greys', alpha=0.5)
#
plt.contour(trad_12co10_2d, origin='lower', extent=extent, levels=[obs[5]], colors='black')
cband = ((trad_12co10_2d >= 0.95*obs[5])).astype(float)  # filling factor <= 1.
plt.imshow(cband, origin='lower', extent=extent, aspect='auto', cmap='Reds', alpha=0.5)
# klutzy business here because I can't figure out how to do automatic detection of contours in the legend
custom_lines = [Line2D([0], [0], color='blue', lw=0),
                Line2D([0], [0], color='C1', lw=0),
                Line2D([0], [0], color='C2', lw=0),
                Line2D([0], [0], color='magenta', lw=0),
                Line2D([0], [0], color='black', lw=0)
                ]
ax = plt.gca()
textcolors=['blue','C1','C2','magenta', 'black']
labellist = ['$^{12}$CO/$^{13}$CO 1$-$0', '$^{12}$CO 2$-$1/1$-$0', '$^{12}$CO/$^{13}$CO 2$-$1', '$^{13}$CO/C$^{18}$O 1$-$0', '$^{12}$CO 1$-$0']
leg = ax.legend(custom_lines, labellist, loc='best', frameon=False, handlelength=0., scatterpoints=0)
for i in enumerate(leg.get_texts()):
     i[1].set_color(textcolors[i[0]])
ax.tick_params(which='both', direction='in', top=True, right=True)
plt.xlabel('log T (K)')
plt.ylabel('log n (cm$^{-3}$)')
# ax.set_rasterized(True) # kludge reproduces colors correctly in the pdf but looks yucky
plt.savefig('plotnT+contours_'+plotname)  # alphas are too light in the pdf for some reason - try jpg

# more diagnostic plot on my line intensity calculations
plt.figure()
plt.imshow(trad_12co10_2d, origin='lower', extent=extent, aspect='auto', cmap='viridis')
plt.colorbar()
plt.contour(trad_12co10_2d, origin='lower', extent=extent, levels=[6.1], colors='white')
plt.title('$^{12}$CO 1-0 intensity, K')
plt.xlabel('log T (K)')
plt.ylabel('log n (cm$^{-3}$)')
plt.savefig('plotnT12co.pdf')
plt.figure()
plt.imshow(trad_13co10_2d, origin='lower', extent=extent, aspect='auto', cmap='viridis')
plt.colorbar()
plt.contour(trad_13co10_2d, origin='lower', extent=extent, levels=[1.0], colors='white')
plt.title('$^{13}$CO 1-0 intensity, K')
plt.xlabel('log T (K)')
plt.ylabel('log n (cm$^{-3}$)')
plt.savefig('plotnT13co.pdf')
plt.figure()
plt.imshow(trad_c18o10_2d, origin='lower', extent=extent, aspect='auto', cmap='viridis')
plt.colorbar()
plt.contour(trad_c18o10_2d, origin='lower', extent=extent, levels=[0.16], colors='white')
plt.title('C$^{18}$O 1-0 intensity, K')
plt.xlabel('log T (K)')
plt.ylabel('log n (cm$^{-3}$)')
plt.savefig('plotnTc18o.pdf')


