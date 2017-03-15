
# coding: utf-8

# In[1]:

import os
import numpy as np


# In[2]:

FILE = "data/cullpdb+profile_6133.npy.gz"
FILTERED = "data/cullpdb+profile_6133_filtered.npy.gz"
FILEPATH = os.path.abspath(os.path.join(os.getcwd(), FILE))

TRAIN = 5600  # [0, 5600)
TEST = 5877  # [5600, 5877)
VAL = 6133  # [5877, 6133)
DATA_SIZE = 6133

RESIDUE_IND = 22  # [0, 22) for each amino acid
LABEL_IND = 31  # [22, 31) for each amino acid
NUM_FEATURES = 57  # per residue
NUM_RESIDUES = 700  # per protein

RESIDUES = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',             'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
LABELS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']


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

# In[3]:

#print(data[0][0:22])
#print(data[0][22:31])
#print(data[0][31:33])
#print(data[0][33:35])
#print(data[0][35:57])


# In[4]:

def load_file(file_path, abspath=False, verbose=True):
    if not abspath:
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


# In[12]:

# train_only must be true to load filtered set
def load_residues(file_path, abspath=False, verbose=True, train_only=False, two_d=False):
    # extract first 22 columns of every 57 columns of each row
    residue_cols = [i for i in range(NUM_RESIDUES*NUM_FEATURES) if i % NUM_FEATURES < RESIDUE_IND]
    label_cols = [i for i in range(NUM_RESIDUES*NUM_FEATURES) if RESIDUE_IND <= i % NUM_FEATURES < LABEL_IND]
    
    if verbose:
        print("Loading protein residues and labels...")
    data = load_file(file_path, abspath, verbose)
    if data is None:
        return None, None, None
    
    # load only training data for filtered
    if train_only:
        train_x = np.array( data[:, residue_cols] )
        train_y = np.array( data[:, label_cols] )
        if verbose:
            print("Loaded protein residues and labels.")
        if two_d:
            if verbose:
                print("Reshaping...")
            train_x = train_x.reshape(len(train_x), 700, 22)
            train_y = train_y.reshape(len(train_y), 700, 9)
            if verbose:
                print("Reshaped")
        return (train_x, train_y)
    
    assert len(data) == DATA_SIZE, "Data has size: {0}".format(len(data))
    
    # extract training residues and labels
    train_x = np.array( data[:TRAIN, residue_cols] )
    train_y = np.array( data[:TRAIN, label_cols] )
    
    test_x = np.array( data[TRAIN:TEST, residue_cols] )
    test_y = np.array( data[TRAIN:TEST, label_cols] )
    
    val_x = np.array( data[TEST:VAL, residue_cols] )
    val_y = np.array( data[TEST:VAL, label_cols] )
    
    if two_d:
        if verbose:
            print("Reshaping...")
        # reshape to 3d matrices - one residue per slice, one protein per row
        train_x = train_x.reshape(TRAIN, 700, 22)
        train_y = train_y.reshape(TRAIN, 700, 9)
        test_x = test_x.reshape(TEST-TRAIN, 700, 22)
        test_y = test_y.reshape(TEST-TRAIN, 700, 9)
        val_x = val_x.reshape(VAL-TEST, 700, 22)
        val_y = val_y.reshape(VAL-TEST, 700, 9)
        if verbose:
            print("Reshaped")
    
    if verbose:
        print("Loaded protein residues and labels.")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# In[6]:

def print_residues(data, labels=None, two_d=False):
    rs = []
    lb = []
    # len(data) should == NUM_RESIDUES * RESIDUE_IND
    if two_d:
        data = data.reshape(700*22)
    interval = RESIDUE_IND
    for i in range(0, len(data), interval):
        res = RESIDUES[np.argmax(data[i:i+interval])]
        # break at end of protein
        if res == 'NoSeq':
            break
        rs.append(res)
            
    if labels is not None:
        if two_d:
            labels = labels.reshape(700*9)
        interval = LABEL_IND - RESIDUE_IND
        for i in range(0, len(rs)*interval, interval):
            label = LABELS[np.argmax(labels[i:i+interval])]
            lb.append(label)
        print("Residues:")
        print("".join(rs))
        print("Labels:")
        print("".join(lb))
        return rs, lb
    else:
        print("".join(rs))
        return rs


# In[7]:

def load_cb513(file_path, abspath=False, verbose=True, two_d=False):
    if not abspath:
        file_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        
    if verbose:
        print("Loading file from ", file_path, "...", sep="")
    try:
        data = np.load(file_path)
        if verbose:
            print("File Loaded.")
    except:
        print("\n\nFile could not be found at", file_path, "\n\n")
        return None
    
    # extract first 22 columns of every 57 columns of each row
    residue_cols = [i for i in range(NUM_RESIDUES*NUM_FEATURES) if i % NUM_FEATURES < RESIDUE_IND]
    label_cols = [i for i in range(NUM_RESIDUES*NUM_FEATURES) if RESIDUE_IND <= i % NUM_FEATURES < LABEL_IND]
    
    inputs = np.array( data[:, residue_cols] )
    labels = np.array( data[:, label_cols] )
    
    if two_d:
        inputs = inputs.reshape(len(inputs), 700, 22)
        labels = labels.reshape(len(labels), 700, 9)
        
    return (inputs, labels)


# In[8]:

def get_residues(): return RESIDUES[:]
def get_labels(): return LABELS[:]


# In[9]:

def _tester():
    path = "data/cullpdb+profile_6133.npy.gz"

    train, validation, test = load_residues_2D(path)

    train_x, train_y = train
    print(train_x.shape)
    print(train_y.shape)
#    i = 69
#    r, l = print_residues(train_x[i], labels=train_y[i])


# In[10]:

# _tester()

