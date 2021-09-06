import numpy as np

def select_random_index(n, k):
    
    """ Select randomly without repetition k integers taken in the interval [0, n-1].
        Parameters
        ----------
        n : int
            n-1 is the maximum value of the index that can be drawn
        k : int
            number of index drawn
    Returns
    -------
    list_index : list of integers
                 list containing the index randomly drawn
    """
    
    list_index = []
    while len(list_index) < k:
        random_index = np.random.randint(0, n)
        if random_index not in list_index:
            list_index.append(random_index)
    return list_index

def ionosphere(path):
    
    """ Extracts the data from .data file downloaded on the website : 
        http://archive.ics.uci.edu/ml/datasets/Ionosphere. Data are then
        randomly splitted into a train set (70% of the data) and a test set (30% of the data).
        Data are stored in bidimensional arrays.
    Parameters
    ----------
    path : str
           path locating the .data files
    Returns
    -------
    X_train      : 2-d array 
                   train set
    labels_train : 1-d array
                   labels of the observations in the train set
    X_test       : 2-d array 
                   test set
    labels_test  : 1-d array
                   labels of the observations in the test set                   
    """
    
    X, labels, X_train, labels_train, X_test, labels_test = [], [], [], [], [], []
    f = open(path + "ionosphere.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X.append(clean_line.split(","))
    for i in range(len(X)):
        for j in range(len(X[i]) - 1):
            X[i][j] = float(X[i][j])
        if X[i][-1] == 'g':
            labels.append(1)
        if X[i][-1] == 'b':
            labels.append(0)
        X[i] = X[i][:-1]
    list_index_train_set = select_random_index(len(X), int(0.7*len(X)))
    for i in range(len(X)):
        if i in list_index_train_set:
            X_train.append(X[i])
            labels_train.append(labels[i])
        else:
            X_test.append(X[i])
            labels_test.append(labels[i])
            
    return np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test)

def statlog(path):
    
    """ Extracts the data from .data file downloaded on the website : 
        http://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29. There is a 
        file for the train set and the test set. Data are stored in bidimensional arrays.
    Parameters
    ----------
    path : str
           path locating the .data files
    Returns
    -------
    X_train      : 2-d array 
                   train set
    labels_train : 1-d array
                   labels of the observations in the train set
    X_test       : 2-d array 
                   test set
    labels_test  : 1-d array
                   labels of the observations in the test set                   
    """
    
    X_train, labels_train, X_test, labels_test = [], [], [], []
    f = open(path + "sat_trn.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X_train.append(clean_line.split(" "))
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            X_train[i][j] = float(X_train[i][j])
        labels_train.append(int(X_train[i][-1]))
        X_train[i] = X_train[i][:-1]

    f = open(path + "sat_tst.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X_test.append(clean_line.split(" "))
    for i in range(len(X_test)):
        for j in range(len(X_test[i])):
            X_test[i][j] = float(X_test[i][j])
        labels_test.append(int(X_test[i][-1]))
        X_test[i] = X_test[i][:-1]
        
    return np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test)

def breast_cancer(path):
    
    """ Extracts the data from .data file downloaded on the website : 
        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
        Data file was renamed from breast-cancer-wisconsin to breast_cancer_wisconsin.
        Data are then randomly splitted into a train set (70% of the data) and a test set (30% of the data). 
    Data are stored in bidimensional arrays.
    Parameters
    ----------
    path : str
           path locating the .data files
    Returns
    -------
    X_train      : 2-d array 
                   train set
    labels_train : 1-d array
                   labels of the observations in the train set
    X_test       : 2-d array 
                   test set
    labels_test  : 1-d array
                   labels of the observations in the test set                   
    """
    
    X, labels, X_train, labels_train, X_test, labels_test = [], [], [], [], [], []
    f = open(path + "breast_cancer_wisconsin.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X.append(clean_line.split(","))
    # There are missing values for the 6th coordinate of some data, we replace it 
    # by the value of another observation randomly drawn.
    t = [402, 30, 28, 19, 30, 4, 8, 21, 9, 132] # numbers of 1.0, 2.0..., 10.0 values
    values_coordinate_6 = [] # empirical distribution of the 6th coordinate's values
    for i in range(10):
        values_coordinate_6 = values_coordinate_6 + [float(i+1) for j in range(t[i])]
    for i in [23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315, 321, 411, 617]: # corrupted observations
        X[i][6] = str(values_coordinate_6[np.random.randint(683)])
    for i in range(len(X)):
        for j in range(1, len(X[i]) - 1):
            X[i][j] = float(X[i][j])
        if X[i][-1] == '2':
            labels.append(0)
        if X[i][-1] == '4':
            labels.append(1)
        X[i] = X[i][1:-1]
    list_index_train_set = select_random_index(len(X), int(0.7*len(X)))
    for i in range(len(X)):
        if i in list_index_train_set:
            X_train.append(X[i])
            labels_train.append(labels[i])
        else:
            X_test.append(X[i])
            labels_test.append(labels[i])
            
    return np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test)
    
def ecoli(path):
    
    """ Extracts the data from .data file downloaded on the website : 
        https://archive.ics.uci.edu/ml/datasets/ecoli
        Data are then randomly splitted into a train set (70% of the data) and a test set (30% of the data). 
    Data are stored in bidimensional arrays.
    Parameters
    ----------
    path : str
           path locating the .data files
    Returns
    -------
    X_train      : 2-d array 
                   train set
    labels_train : 1-d array
                   labels of the observations in the train set
    X_test       : 2-d array 
                   test set
    labels_test  : 1-d array
                   labels of the observations in the test set                   
    """
    
    X, labels, X_train, labels_train, X_test, labels_test = [], [], [], [], [], []
    f = open(path + "ecoli.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X.append(clean_line.split("  "))
    for i in range(len(X)):
        try:
            X[i].remove("")
        except:
            pass
        for j in range(1, len(X[i]) - 1):
            X[i][j] = float(X[i][j])
        if X[i][-1] == ' cp':
            labels.append(0)
        if X[i][-1] == ' im':
            labels.append(1)
        if X[i][-1] == 'imS':
            labels.append(2)
        if X[i][-1] == 'imL':
            labels.append(3)
        if X[i][-1] == 'imU':
            labels.append(4)
        if X[i][-1] == ' om':
            labels.append(5)
        if X[i][-1] == 'omL':
            labels.append(6)
        if X[i][-1] == ' pp':
            labels.append(7)
        X[i] = X[i][1:-1]
    list_index_train_set = select_random_index(len(X), int(0.7*len(X)))
    for i in range(len(X)):
        if i in list_index_train_set:
            X_train.append(X[i])
            labels_train.append(labels[i])
        else:
            X_test.append(X[i])
            labels_test.append(labels[i])
            
    return np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test)
    
def spambase(path):
    
    """ Extracts the data from .data file downloaded on the website : 
        https://archive.ics.uci.edu/ml/datasets/spambase
        Data are then randomly splitted into a train set (70% of the data) and a test set (30% of the data). 
    Data are stored in bidimensional arrays.
    Parameters
    ----------
    path : str
           path locating the .data files
    Returns
    -------
    X_train      : 2-d array 
                   train set
    labels_train : 1-d array
                   labels of the observations in the train set
    X_test       : 2-d array 
                   test set
    labels_test  : 1-d array
                   labels of the observations in the test set                   
    """
    
    X, labels, X_train, labels_train, X_test, labels_test = [], [], [], [], [], []
    f = open(path + "spambase.data", "r")
    lines = f.readlines()
    for line in lines:
        clean_line = line[:-1]
        X.append(clean_line.split(","))
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = float(X[i][j])
        labels.append(int(X[i][-1]))
        X[i] = X[i][:-1]
    list_index_train_set = select_random_index(len(X), int(0.7*len(X)))
    for i in range(len(X)):
        if i in list_index_train_set:
            X_train.append(X[i])
            labels_train.append(labels[i])
        else:
            X_test.append(X[i])
            labels_test.append(labels[i])
            
    return np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test)