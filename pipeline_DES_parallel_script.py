from Gazou import Gazou
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import multiprocessing
import os

def gingafit(batch_number):
    
    ''' This function represents the code to be executed in each batch '''

    tile = 'Data/Dataset_DES/DES0008+0043_i.fits'
    tile_segmap = 'Data/Dataset_DES/DES0008+0043_i_segmap.fits'
    cat = 'Data/Dataset_DES/DES0008+0043_i_cat.fits'
    psfcat = 'Data/Dataset_DES/DES0008+0043_i_psfcat.psf'

    stamps_folder = 'Output/Stamps_DES/'
    galfit_folder = 'Output/Galfit_DES/'
    os.makedirs(stamps_folder, exist_ok = True)
    os.makedirs(galfit_folder, exist_ok = True)

    gazou = Gazou(tile, tile_segmap, cat, psfcat)
    psf_size, px_scale, mag_zp, exptime, ncombine = 49, 0.263, 30., 1, 16
    gazou.add_params(psf_size, px_scale, mag_zp, exptime, ncombine)
    gazou.add_paths(stamps_folder, galfit_folder)
        
    # >>> Indicate the number of cores for parallel computing
    num_batches = 4
    
    cat_file = fits.open(cat)
    catalogue = cat_file[1].data

    selection = catalogue[catalogue['FLAGS']==0]
    selection = selection[selection['MAG_AUTO']<19]
    selection = selection[selection['MAG_AUTO']>13]
    selection = selection[selection['CLASS_STAR']<=0.9]
    selection = selection[selection['KRON_RADIUS']*selection['A_IMAGE']>=50]
    print(len(selection))

    numbers = selection['NUMBER']
    collection = np.array_split(numbers, num_batches)
    
    files_collection = collection[int(batch_number-1)]
    print(f"Running Batch {batch_number}")
    print(files_collection)
    
    for k in range(0, len(files_collection)):
        print(k)
        object_id = files_collection[k]-1
        gazou.create_stamps(object_id)


if __name__ == "__main__":

    # >>> Indicate the number of cores for parallel computing
    num_batches = 4
    n_cores = 4

    # Collection of the dataset
    cat_file = fits.open('../../GingaFit/Dataset/match_AGN_fullcat.fits')
    perseus_cat = cat_file[1].data
    numbers = perseus_cat['NUMBER']
    
    sublists = np.array_split(numbers, num_batches)
    print(len(sublists))
    for collection in sublists:
        print(len(collection))

    # Create a Pool with the desired number of cores
    num_cores = n_cores
    pool = multiprocessing.Pool(processes=num_cores)

    # Map the process_batch function to each batch number
    pool.map(gingafit, range(1, num_batches + 1))

    # Close the pool to release resources
    pool.close()
    pool.join()
