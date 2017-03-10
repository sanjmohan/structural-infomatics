
# coding: utf-8

# In[22]:

import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[62]:

FILE = "data/cullpdb+profile_6133.npy.gz"
FILEPATH = os.path.abspath(os.path.join(os.getcwd(), FILE))

TRAIN = 5600
TEST = 5877
VAL = 6133

RESIDUES = 22
LABELS = 31


# The 57 features are:<br>
# "[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'"<br>
# "[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'"<br>
# "[31,33): N- and C- terminals;"<br>
# "[33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)"<br>
# "[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues"<br>
# <br>
# The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence.<br>
# "[22,31) and [33,35) are hidden during testing."<br>
# 
# 
# "The dataset division for the first ""cullpdb+profile_6133.npy.gz"" dataset is"<br>
# "[0,5600) training"<br>
# "[5605,5877) test "<br>
# "[5877,6133) validation"<br>

# In[51]:

#print(data[0][0:22])
#print(data[0][22:31])
#print(data[0][31:33])
#print(data[0][33:35])
#print(data[0][35:57])


# In[56]:

def load_file(file_path, absolute=False, verbose=True):
    if not absolute:
        file_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
    if verbose:
        print("Loading file from ", file_path, "...", sep="")
    try:
        data = np.load(file_path)
        if verbose:
            print("File Loaded.")
        return data
    except:
        print("\n\nFile could not be found at", file_path, "\n\n")
        return None


# In[75]:

def load_residues(file_path, absolute=False, verbose=True):
    if verbose:
        print("Loading protein residues and labels...")
    data = load_file(file_path, absolute, verbose)
    if data is None:
        return None, None, None
    
    # extract training residues (first 22 features of first 5600 proteins)
    train_x = np.array( [data[i][0:RESIDUES] for i in range(TRAIN)] )
    train_y = np.array( [data[i][RESIDUES:LABELS] for i in range(TRAIN)] )
    
    test_x = np.array( [data[i][0:RESIDUES] for i in range(TRAIN, TEST)] )
    test_y = np.array( [data[i][RESIDUES:LABELS] for i in range(TRAIN, TEST)] )
    
    val_x = np.array( [data[i][0:RESIDUES] for i in range(TEST, VAL)] )
    val_y = np.array( [data[i][RESIDUES:LABELS] for i in range(TEST, VAL)] )
    
    if verbose:
        print("Loaded protein residues and labels.")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# In[76]:




# In[81]:




# In[ ]:



