# GingaFit

The main class of GingaFit, Gazou, deals with the creation of galaxy stamps, psfs, segmentation maps, and Galfit-related input files and masks. 
As showed in the example Jupyter Notebooks, the user needs to create an instance of Gazou, and add the following information:
- input tile image and segmentation maps;
- input photometric catalogues;
- input PSF catalogues;
- output folders;
- psf-stamp size, pixel scale, magnitude zeropoint, exposure time, NCOMBINE.

The code can be run sequentially (see Jupyter notebooks) or in parallel, using the scripts pipeline_DES_parallel_script.py and pipeline_HST_parallel_script.py . 
All the outputs will be authomatically stored in the indicated output folders. They will include the derived images and several models for parametric fitting, plus python scripts to run then with Galfit either sequencially or in parallel.

Before running GingaFit:
- The example datasets (from public releases of HST and the Dark Energy Survey) can be downloaded following the link in the readme file dave in the Data folder;
- Install the required python packages using pip install -r requirements.txt

For further information, please contact the author, federica.tarsitano@unige.ch . Thank you!
