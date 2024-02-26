import glob
import os, sys
import time

files = glob.glob('*.feedme')
print(files)

for model in files:
    t_begin = time.time()
    os.system("galfit " + model)
    os.system("rm galfit.*")
    t_end = time.time()
    print("Time Elapsed for the search: %6.4f" % (t_end-t_begin))
    print()
