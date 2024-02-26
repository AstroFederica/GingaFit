import multiprocessing
import numpy as np
import glob
import os, sys


def rungalfit(batch_number):
    # This function represents the code to be executed in each batch
    
    num_batches = 4
    files = np.array(glob.glob('*.feedme'))
    collection = np.array_split(files, num_batches)
    
    files_collection = collection[int(batch_number-1)]
    print(f"Running Batch {batch_number}")
    print(files_collection)
    
    for model in files_collection:
        os.system("galfit " + model)
        os.system("rm galfit.*")

if __name__ == "__main__":

    # Number of batches
    num_batches = 4
    n_cores = 4

    # Collection of the dataset
    files = np.array(glob.glob('*.feedme'))
    print(files.shape)
    
    sublists = np.array_split(files, num_batches)
    print(len(sublists))
    for collection in sublists:
        print(len(collection))

    # Create a Pool with the desired number of cores
    num_cores = n_cores
    pool = multiprocessing.Pool(processes=num_cores)

    # Map the process_batch function to each batch number
    pool.map(rungalfit, range(1, num_batches + 1))

    # Close the pool to release resources
    pool.close()
    pool.join()
    
    
