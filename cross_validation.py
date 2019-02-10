import sys
import numpy as np
import broccoli as bro

# PATH_TO_DATA_FILE(string), PRUNE(bool)

if len(sys.argv) != 3 or (sys.argv[2].lower() != "true" and sys.argv[2].lower() != "false"):
    print("Please run the script followed by the path to data file and if it should be pruned(true/false)")
    print("e.g. python3 cross_validation.py co395-cbc-dt/wifi_db/clean_dataset.txt true")
    exit()

dataset = np.loadtxt(sys.argv[1], usecols= (0, 1, 2, 3, 4, 5, 6, 7), unpack= False)

pruned = (sys.argv[2].lower() == "true")

if pruned:
    bro.ten_cross_validation_with_prun(dataset)
else:
    bro.ten_cross_validation_without_prun(dataset)
