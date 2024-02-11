Code associated with [Young et al (2022), ApJ 933, 90](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...90Y/abstract), Appendix B.

The workflow for using these two scripts will be
1. Edit radex_makecubes.py; check the grid boundaries and step sizes to verify the parameter ranges of interest.
   Also adjust the radex function call to match your system.
2. Run radex_makecubes.py.  The output FITS files may take up several GB on disk, depending on the size of the grids.
3. Edit sample_radex_output.py; adjust values of the measurements (the line ratios and 12CO brightness
    temperature) to match your data.
4. Run sample_radex_output.py and enjoy the plots.



Please cite as follows - thanks!

@ARTICLE{2022ApJ...933...90Y,  
       author = {{Young}, Lisa M. and {Meier}, David S. and {Crocker}, Alison and {Davis}, Timothy A. and {Topal}, Sel{\c{c}}uk},  
        title = "{Down but Not Out: Properties of the Molecular Gas in the Stripped Virgo Cluster Early-type Galaxy NGC 4526}",  
      journal = {\apj},  
     keywords = {Early-type galaxies, Interstellar molecules, CO line emission, Molecular gas, Galaxy evolution, Virgo Cluster, 429, 849, 262, 1073, 594, 1772, Astrophysics - Astrophysics of Galaxies},  
         year = 2022,  
        month = jul,  
       volume = {933},  
       number = {1},  
          eid = {90},  
        pages = {90},  
          doi = {10.3847/1538-4357/ac7149},  
archivePrefix = {arXiv},  
       eprint = {2204.02382},  
 primaryClass = {astro-ph.GA},  
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJ...933...90Y},  
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}  
}

