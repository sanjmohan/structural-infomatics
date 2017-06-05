
# coding: utf-8

# In[1]:

import os
import numpy as np


# In[2]:

FILE = "data/cullpdb+profile_6133.npy.gz"
FILTERED = "data/cullpdb+profile_6133_filtered.npy.gz"
FILEPATH = os.path.abspath(os.path.join(os.getcwd(), FILE))

RESIDUE_SIZE = 22
NUM_LABELS = 9

# for unfiltered cbd dataset:
TRAIN = 5600  # [0, 5600)
TEST = 5877  # [5600, 5877)
VAL = 6133  # [5877, 6133)
DATA_SIZE = 6133

RESIDUE_IND = 22  # [0, 22) for each amino acid
LABEL_IND = 31  # [22, 31) for each amino acid
PSSM_IND = 35
NUM_FEATURES = 57  # per residue
NUM_RESIDUES = 700  # per protein

# Symbols: "-" placeholder for "NoSeq"
RESIDUES = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',             'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','-']
LABELS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','-']


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

def _get_cols(pssm=False):
    # extract columns for residues, labels, pssm (seq profile)
    feature_cols = []
    label_cols = []
    for i in range(NUM_RESIDUES*NUM_FEATURES):
        j = i % NUM_FEATURES
        if j < RESIDUE_IND:
            feature_cols.append(i)
        elif j < LABEL_IND:
            label_cols.append(i)
        elif pssm and PSSM_IND <= j:
            feature_cols.append(i)
    return feature_cols, label_cols


# In[5]:

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


# In[6]:

# split must be false to load filtered set
def load_residues(file_path, abspath=False, verbose=True, split=True, two_d=False, pssm=True):
    num_features = RESIDUE_SIZE
    if pssm:
        num_features += RESIDUE_SIZE
    feature_cols, label_cols = _get_cols(pssm)
    
    if verbose:
        print("Loading protein residues and labels...")
    data = load_file(file_path, abspath, verbose)
    if data is None:
        return None, None, None
    
    # load only training data (eg for filtered)
    if not split:
        train_x = np.array( data[:, feature_cols] )
        train_y = np.array( data[:, label_cols] )
        if verbose:
            print("Loaded protein residues and labels.")
        if two_d:
            if verbose:
                print("Reshaping...")
            train_x = train_x.reshape(-1, NUM_RESIDUES, num_features)
            train_y = train_y.reshape(-1, NUM_RESIDUES, NUM_LABELS)
            if verbose:
                print("Reshaped")
        return (train_x, train_y)
    
    assert len(data) == DATA_SIZE, "Data has size: {0}".format(len(data))
    
    # extract training residues and labels
    train_x = np.array( data[:TRAIN, feature_cols] )
    train_y = np.array( data[:TRAIN, label_cols] )
    
    test_x = np.array( data[TRAIN:TEST, feature_cols] )
    test_y = np.array( data[TRAIN:TEST, label_cols] )
    
    val_x = np.array( data[TEST:VAL, feature_cols] )
    val_y = np.array( data[TEST:VAL, label_cols] )
    
    if two_d:
        if verbose:
            print("Reshaping...")
        # reshape to 3d matrices - one residue per slice, one protein per row
        train_x = train_x.reshape(TRAIN, NUM_RESIDUES, num_features)
        train_y = train_y.reshape(TRAIN, NUM_RESIDUES, NUM_LABELS)
        test_x = test_x.reshape(TEST-TRAIN, NUM_RESIDUES, num_features)
        test_y = test_y.reshape(TEST-TRAIN, NUM_RESIDUES, NUM_LABELS)
        val_x = val_x.reshape(VAL-TEST, NUM_RESIDUES, num_features)
        val_y = val_y.reshape(VAL-TEST, NUM_RESIDUES, NUM_LABELS)
        if verbose:
            print("Reshaped")
    
    if verbose:
        print("Loaded protein residues and labels.")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# In[7]:

# short - if True, terminates sequence after finding first 'NoSeq'
def print_residues(data, labels=None, two_d=False, short=True):
    rs = []
    lb = []
    # len(data) should == NUM_RESIDUES * num_features
    if not two_d:
        data = data.reshape(NUM_RESIDUES, -1)
    for i in range(len(data)):
        res = RESIDUES[np.argmax(data[i][:RESIDUE_SIZE])]
        # break at end of protein
        if short and res == 'NoSeq':
            break
        rs.append(res)
            
    if labels is not None:
        if not two_d:
            labels = labels.reshape(NUM_RESIDUES, -1)
        for i in range(len(rs)):
            label = LABELS[np.argmax(labels[i][:NUM_LABELS])]
            lb.append(label)
        print("Residues:")
        print("".join(rs))
        print("Labels:")
        print("".join(lb))
        return rs, lb
    else:
        print("".join(rs))
        return rs


# In[8]:

def load_cb513(file_path, abspath=False, verbose=True, two_d=False, pssm=True):
    if not abspath:
        file_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        
    if verbose:
        print("Loading file from ", file_path, "...", sep="")
    try:
        data = np.load(file_path)
        if verbose:
            print("File Loaded.")
            
        num_features = RESIDUE_SIZE
        if pssm:
            num_features += RESIDUE_SIZE
        feature_cols, label_cols = _get_cols(pssm)

        inputs = np.array( data[:, feature_cols] )
        labels = np.array( data[:, label_cols] )

        if two_d:
            inputs = inputs.reshape(len(inputs), NUM_RESIDUES, num_features)
            labels = labels.reshape(len(labels), NUM_RESIDUES, NUM_LABELS)

        return (inputs, labels)
    except:
        print("\n\nFile could not be found at", file_path, "\n\n")
        return None


# In[9]:

def get_residues(): return RESIDUES[:]
def get_labels(): return LABELS[:]


# In[10]:

def _tester():
    path = "data/cullpdb+profile_6133.npy.gz"

    train, validation, test = load_residues_2D(path)

    train_x, train_y = train
    print(train_x.shape)
    print(train_y.shape)
#    i = 69
#    r, l = print_residues(train_x[i], labels=train_y[i])


# In[11]:

# _tester()


# In[ ]:



