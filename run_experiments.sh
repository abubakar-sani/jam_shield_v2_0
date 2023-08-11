#!/bin/bash
# Results bash script
# author: abubakar sani ali


# Experiment 1: wideband, 5GHz, csc = 0
python3 results/Anti_Jam.py 0 5 0

# Experiment 2: wideband, 5GHz, csc = 0.2
python3 results/Anti_Jam.py 0 5 0.2

# Experiment 3: wideband, 2.4GHz, csc = 0
python3 results/Anti_Jam.py 0 2.4 0

# Experiment 4: wideband, 2.4GHz, csc = 0.2
python3 results/Anti_Jam.py 0 2.4 0.2

# Experiment 5: broadband, 5GHz, csc = 0
python3 results/Anti_Jam.py 1 5 0

# Experiment 6: broadband, 5GHz, csc = 0.2
python3 results/Anti_Jam.py 1 5 0.2

# Experiment 7: broadband, 2.4GHz, csc = 0
python3 results/Anti_Jam.py 1 2.4 0

# Experiment 8: broadband, 2.4GHz, csc = 0.2
python3 results/Anti_Jam.py 1 2.4 0.2

echo All Done!
