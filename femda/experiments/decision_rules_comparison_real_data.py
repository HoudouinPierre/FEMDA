from . import estimateurs as est
from . import decision_rules as dr
from . import preprocessing as pre

import numpy as np
import pandas as pd
import pickle as pk
import os
from numpy.random import multivariate_normal
import dataframe_image as dfi
import matplotlib.pyplot as pl
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import time    
import warnings
warnings.filterwarnings("ignore")
    
def time_needed(nb_sec):
    
    """ Returns a string with nb_sec converted in hours, minutes and seconds.
    """    
    
    nb_heures = nb_sec // 3600
    nb_min    = (nb_sec - nb_heures * 3600) // 60
    nb_s      = (nb_sec - nb_heures * 3600 - nb_min * 60)
    
    return str(nb_heures) + " h " + str(nb_min) + " min " + str(nb_s) + "s"

def contamine(X, p_conta, dataset_name):
    
    """ Contaminates observations of dataset X with probability p_conta. The
    type of contamination is specific for each dataset.
    
    Parameters
    ----------
    X              : 2-d array of size n*m
                     dataset 
    p_conta_distri : float between 0 and 1
                    probability to contaminate an observation
    dataset_name   : str
                    Dataset to contaminate
                    
    Returns
    -------
    X              : 2-d array of size n*m
                     dataset contaminated
    """

    if dataset_name == "Ionosphere":
        return contamine_ionosphere(X, p_conta)
    if dataset_name == "Statlog":
        return contamine_statlog(X, p_conta)
    if dataset_name == "Breast cancer":
        return contamine_breast_cancer(X, p_conta)
    if dataset_name == "Ecoli":
        return contamine_ecoli(X, p_conta)
    if dataset_name == "Spambase":
        return contamine_spambase(X, p_conta)
    
def contamine_ionosphere(X, p_conta):

    """ Contamination funtion for Ionosphere dataset.
    """
    
    for i in range(len(X)):
        if np.random.rand() < p_conta:
            X[i] = 2 * np.random.rand(34) - 1
    return contamine_statlog(X, p_conta)

def contamine_statlog(X, p_conta):

    """ Contamination funtion for Statlog Landsat Satellite dataset.
    """
    
    for i in range(len(X)):
        if np.random.rand() < p_conta:
            X[i] = multivariate_normal(np.zeros(len(X[i])), 1e8 * np.identity(len(X[i])))
            X[i] = np.random.rand(len(X[i])) * 140 + 20
    return X

def contamine_breast_cancer(X, p_conta):

    """ Contamination funtion for breast cancer dataset.
    """
    
    for i in range(len(X)):
        if np.random.rand() < p_conta:
            X[i] = multivariate_normal(np.zeros(len(X[i])), 1e8 * np.identity(len(X[i])))
            X[i] = np.random.rand(len(X[i])) * 140 + 20
    return X

def contamine_ecoli(X, p_conta):

    """ Contamination funtion for Ecoli dataset.
    """
    
    for i in range(len(X)):
        if np.random.rand() < p_conta:
            X[i] = multivariate_normal(np.zeros(len(X[i])), 1e8 * np.identity(len(X[i])))
            X[i] = np.random.rand(len(X[i])) * 140 + 20
    return X

def contamine_spambase(X, p_conta):

    """ Contamination funtion for Spambase dataset.
    """
    
    for i in range(len(X)):
        if np.random.rand() < p_conta:
            X[i] = multivariate_normal(np.zeros(len(X[i])), 1e8 * np.identity(len(X[i])))
            X[i] = np.random.rand(len(X[i])) * 140 + 20
    return X

def contamine_labels(labels, p_conta):

    confuse_dict = {}
    for i in list(set(labels)):        
        confuse_dict[str(i)] = list(set(labels))[np.random.randint(0, len(list(set(labels))))]
    for i in range(len(labels)):
        if np.random.rand() < p_conta:
            labels[i] = confuse_dict[str(labels[i])]
    return labels

def decision_rules_specific_performance_evaluation(X, labels, X_test, labels_test, p_conta, dataset_name):
    
    """ Evaluates the error of the estimators and the performances of the different
        decision rules : accuracy, ARI, NMI and AMI index are computed on the train and test set.
    
    Parameters
    ----------
    X           : 2-d array of size n*m
                  train set
    labels      : list of n integers
                  labels of the samples of the train set
    X_test      : 2-d array of size n*m
                  test set
    labels_test : list of n integers
                  labels of the samples of the test set
    X_test      : list containing 1-d array of size m
                  real mean of each cluster
    p_conta     : float between 0 and 1
                  probability to contaminate an observation
    
    Returns
    -------
    accuracy : list of floats
               accuracy of each decision rule for train and test set
    ARI      : list of floats
               ARI index of each decision rule for train and test set
    NMI      : list of floats
               NMI index of each decision rule for train and test set
    AMI      : list of floats
               AMI index of each decision rule for train and test set
    """  

    #X = contamine(X, p_conta, dataset_name)
    labels = contamine_labels(labels, p_conta)

    means_classic, covariances_classic = est.classic_estimator                       (X, labels)
    means_M      , covariances_M       = est.M_estimator                             (X, labels)
    means_t      , covariances_t, nus  = est.t_distribution_estimator                (X, labels)
    means_femda  , covariances_femda   = est.femda_estimator                         (X, labels)
    
    c_classic = dr.find_optimal_c(X, labels, means_classic, covariances_classic)
    c_M       = dr.find_optimal_c(X, labels, means_M, covariances_M)
    
    def same_covariance_estimator(covariances):
        
        covariance = 0
        for covariance_cluster in covariances:
            covariance = covariance + covariance_cluster / len(covariances)
            
        return covariance

    good_classification_LDA_g_classic_train = 0
    good_classification_LDA_g_M_train       = 0
    good_classification_QDA_g_classic_train = 0
    good_classification_QDA_g_M_train       = 0
    good_classification_GQDA_classic_train  = 0
    good_classification_GQDA_M_train        = 0
    good_classification_LDA_t_train         = 0
    good_classification_QDA_t_train         = 0
    good_classification_FEMDA_train         = 0
    
    predicted_label_LDA_g_classic_train = []
    predicted_label_LDA_g_M_train       = []
    predicted_label_QDA_g_classic_train = []
    predicted_label_QDA_g_M_train       = []
    predicted_label_GQDA_classic_train  = []
    predicted_label_GQDA_M_train        = []
    predicted_label_LDA_t_train         = []
    predicted_label_QDA_t_train         = []
    predicted_label_FEMDA_train         = []
    
    for i in range(len(X)):
        predicted_label = dr.LDA_g(X[i], means_classic, same_covariance_estimator(covariances_classic))
        if labels[i] == predicted_label:
            good_classification_LDA_g_classic_train = good_classification_LDA_g_classic_train + 1 / len(X)
        predicted_label_LDA_g_classic_train.append(predicted_label)
        
        predicted_label = dr.LDA_g(X[i], means_M, same_covariance_estimator(covariances_M))
        if labels[i] == predicted_label:
            good_classification_LDA_g_M_train       = good_classification_LDA_g_M_train       + 1 / len(X)
        predicted_label_LDA_g_M_train      .append(predicted_label)
            
        predicted_label = dr.QDA_g(X[i], means_classic, covariances_classic)
        if labels[i] == predicted_label:
            good_classification_QDA_g_classic_train = good_classification_QDA_g_classic_train + 1 / len(X)
        predicted_label_QDA_g_classic_train.append(predicted_label)
            
        predicted_label = dr.QDA_g(X[i], means_M, covariances_M)
        if labels[i] == predicted_label:
            good_classification_QDA_g_M_train       = good_classification_QDA_g_M_train       + 1 / len(X)
        predicted_label_QDA_g_M_train      .append(predicted_label)
            
        predicted_label = dr.GQDA (X[i], means_classic, covariances_classic, c_classic)
        if labels[i] == predicted_label:
            good_classification_GQDA_classic_train  = good_classification_GQDA_classic_train + 1 / len(X)
        predicted_label_GQDA_classic_train .append(predicted_label)
        
        predicted_label = dr.GQDA (X[i], means_M, covariances_M, c_M)
        if labels[i] == predicted_label:
            good_classification_GQDA_M_train  = good_classification_GQDA_M_train + 1 / len(X)
        predicted_label_GQDA_M_train .append(predicted_label)
        
        predicted_label = dr.LDA_t(X[i], means_t, same_covariance_estimator(covariances_t), nus)
        if labels[i] ==predicted_label:
            good_classification_LDA_t_train         = good_classification_LDA_t_train        + 1 / len(X)
        predicted_label_LDA_t_train        .append(predicted_label)
            
        predicted_label= dr.QDA_t(X[i], means_t, covariances_t, nus )
        if labels[i] == predicted_label:
            good_classification_QDA_t_train         = good_classification_QDA_t_train        + 1 / len(X)
        predicted_label_QDA_t_train        .append(predicted_label)
        
        predicted_label = dr.FEMDA(X[i], means_femda, covariances_femda)
        if labels[i] == predicted_label:
            good_classification_FEMDA_train         = good_classification_FEMDA_train        + 1 / len(X)
        predicted_label_FEMDA_train        .append(predicted_label)
        
    if good_classification_GQDA_M_train > good_classification_GQDA_classic_train:
        good_classification_GQDA_train = good_classification_GQDA_M_train
        predicted_label_GQDA_train     = predicted_label_GQDA_M_train
    else:
        good_classification_GQDA_train = good_classification_GQDA_classic_train
        predicted_label_GQDA_train     = predicted_label_GQDA_classic_train

    accuracy_train = [good_classification_LDA_g_classic_train, good_classification_LDA_g_M_train, 
                      good_classification_QDA_g_classic_train, good_classification_QDA_g_M_train,
                      good_classification_GQDA_train , good_classification_LDA_t_train  , 
                      good_classification_QDA_t_train  , good_classification_FEMDA_train]
    ARI_train      = [adjusted_rand_score(labels, predicted_label_LDA_g_classic_train), 
                      adjusted_rand_score(labels, predicted_label_LDA_g_M_train      ), 
                      adjusted_rand_score(labels, predicted_label_QDA_g_classic_train), 
                      adjusted_rand_score(labels, predicted_label_QDA_g_M_train      ), 
                      adjusted_rand_score(labels, predicted_label_GQDA_train         ), 
                      adjusted_rand_score(labels, predicted_label_LDA_t_train        ),
                      adjusted_rand_score(labels, predicted_label_QDA_t_train        ),
                      adjusted_rand_score(labels, predicted_label_FEMDA_train        )]
    NMI_train      = [normalized_mutual_info_score(labels, predicted_label_LDA_g_classic_train), 
                      normalized_mutual_info_score(labels, predicted_label_LDA_g_M_train      ), 
                      normalized_mutual_info_score(labels, predicted_label_QDA_g_classic_train), 
                      normalized_mutual_info_score(labels, predicted_label_QDA_g_M_train      ), 
                      normalized_mutual_info_score(labels, predicted_label_GQDA_train         ), 
                      normalized_mutual_info_score(labels, predicted_label_LDA_t_train        ),
                      normalized_mutual_info_score(labels, predicted_label_QDA_t_train        ),
                      normalized_mutual_info_score(labels, predicted_label_FEMDA_train        )]
    AMI_train      = [adjusted_mutual_info_score(labels, predicted_label_LDA_g_classic_train), 
                      adjusted_mutual_info_score(labels, predicted_label_LDA_g_M_train      ), 
                      adjusted_mutual_info_score(labels, predicted_label_QDA_g_classic_train), 
                      adjusted_mutual_info_score(labels, predicted_label_QDA_g_M_train      ), 
                      adjusted_mutual_info_score(labels, predicted_label_GQDA_train         ), 
                      adjusted_mutual_info_score(labels, predicted_label_LDA_t_train        ),
                      adjusted_mutual_info_score(labels, predicted_label_QDA_t_train        ),
                      adjusted_mutual_info_score(labels, predicted_label_FEMDA_train        )]
    
    good_classification_LDA_g_classic_test = 0
    good_classification_LDA_g_M_test       = 0
    good_classification_QDA_g_classic_test = 0
    good_classification_QDA_g_M_test       = 0
    good_classification_GQDA_classic_test  = 0
    good_classification_GQDA_M_test        = 0
    good_classification_LDA_t_test         = 0
    good_classification_QDA_t_test         = 0
    good_classification_FEMDA_test          = 0
    
    predicted_label_LDA_g_classic_test = []
    predicted_label_LDA_g_M_test       = []
    predicted_label_QDA_g_classic_test = []
    predicted_label_QDA_g_M_test       = []
    predicted_label_GQDA_classic_test  = []
    predicted_label_GQDA_M_test        = []
    predicted_label_LDA_t_test         = []
    predicted_label_QDA_t_test         = []
    predicted_label_FEMDA_test          = []
    
    for i in range(len(X_test)):
        predicted_label = dr.LDA_g(X_test[i], means_classic, same_covariance_estimator(covariances_classic))
        if labels_test[i] == predicted_label:
            good_classification_LDA_g_classic_test = good_classification_LDA_g_classic_test + 1 / len(X_test)
        predicted_label_LDA_g_classic_test.append(predicted_label)
        
        predicted_label = dr.LDA_g(X_test[i], means_M, same_covariance_estimator(covariances_M))
        if labels_test[i] == predicted_label:
            good_classification_LDA_g_M_test       = good_classification_LDA_g_M_test       + 1 / len(X_test)
        predicted_label_LDA_g_M_test      .append(predicted_label)
            
        predicted_label = dr.QDA_g(X_test[i], means_classic, covariances_classic)
        if labels_test[i] == predicted_label:
            good_classification_QDA_g_classic_test = good_classification_QDA_g_classic_test + 1 / len(X_test)
        predicted_label_QDA_g_classic_test.append(predicted_label)
            
        predicted_label = dr.QDA_g(X_test[i], means_M, covariances_M)
        if labels_test[i] == predicted_label:
            good_classification_QDA_g_M_test       = good_classification_QDA_g_M_test       + 1 / len(X_test)
        predicted_label_QDA_g_M_test      .append(predicted_label)
            
        predicted_label = dr.GQDA (X_test[i], means_classic, covariances_classic, c_classic)
        if labels_test[i] == predicted_label:
            good_classification_GQDA_classic_test  = good_classification_GQDA_classic_test + 1 / len(X_test)
        predicted_label_GQDA_classic_test .append(predicted_label)
        
        predicted_label = dr.GQDA (X_test[i], means_M, covariances_M, c_M)
        if labels_test[i] == predicted_label:
            good_classification_GQDA_M_test        = good_classification_GQDA_M_test + 1 / len(X_test)
        predicted_label_GQDA_M_test       .append(predicted_label)
        
        predicted_label = dr.LDA_t(X_test[i], means_t, same_covariance_estimator(covariances_t), nus)
        if labels_test[i] ==predicted_label:
            good_classification_LDA_t_test         = good_classification_LDA_t_test        + 1 / len(X_test)
        predicted_label_LDA_t_test        .append(predicted_label)
            
        predicted_label = dr.QDA_t(X_test[i], means_t, covariances_t, nus )
        if labels_test[i] == predicted_label:
            good_classification_QDA_t_test         = good_classification_QDA_t_test        + 1 / len(X_test)
        predicted_label_QDA_t_test        .append(predicted_label)
        
        predicted_label = dr.FEMDA(X_test[i], means_femda, covariances_femda)
        if labels_test[i] == predicted_label:
            good_classification_FEMDA_test          = good_classification_FEMDA_test         + 1 / len(X_test)
        predicted_label_FEMDA_test         .append(predicted_label)
        
    if good_classification_GQDA_M_test > good_classification_GQDA_classic_test:
        good_classification_GQDA_test = good_classification_GQDA_M_test
        predicted_label_GQDA_test     = predicted_label_GQDA_M_test
    else:
        good_classification_GQDA_test = good_classification_GQDA_classic_test
        predicted_label_GQDA_test     = predicted_label_GQDA_classic_test
    
    accuracy_test = [good_classification_LDA_g_classic_test, good_classification_LDA_g_M_test,
                     good_classification_QDA_g_classic_test, good_classification_QDA_g_M_test, 
                     good_classification_GQDA_test , good_classification_LDA_t_test  , 
                     good_classification_QDA_t_test  , good_classification_FEMDA_test]
    ARI_test      = [adjusted_rand_score(labels_test, predicted_label_LDA_g_classic_test), 
                     adjusted_rand_score(labels_test, predicted_label_LDA_g_M_test      ), 
                     adjusted_rand_score(labels_test, predicted_label_QDA_g_classic_test), 
                     adjusted_rand_score(labels_test, predicted_label_QDA_g_M_test      ), 
                     adjusted_rand_score(labels_test, predicted_label_GQDA_test         ), 
                     adjusted_rand_score(labels_test, predicted_label_LDA_t_test        ),
                     adjusted_rand_score(labels_test, predicted_label_QDA_t_test        ),
                     adjusted_rand_score(labels_test, predicted_label_FEMDA_test         )]
    NMI_test      = [normalized_mutual_info_score(labels_test, predicted_label_LDA_g_classic_test), 
                     normalized_mutual_info_score(labels_test, predicted_label_LDA_g_M_test      ), 
                     normalized_mutual_info_score(labels_test, predicted_label_QDA_g_classic_test), 
                     normalized_mutual_info_score(labels_test, predicted_label_QDA_g_M_test      ), 
                     normalized_mutual_info_score(labels_test, predicted_label_GQDA_test         ), 
                     normalized_mutual_info_score(labels_test, predicted_label_LDA_t_test        ),
                     normalized_mutual_info_score(labels_test, predicted_label_QDA_t_test        ),
                     normalized_mutual_info_score(labels_test, predicted_label_FEMDA_test         )]
    AMI_test     =  [adjusted_mutual_info_score(labels_test, predicted_label_LDA_g_classic_test), 
                     adjusted_mutual_info_score(labels_test, predicted_label_LDA_g_M_test      ), 
                     adjusted_mutual_info_score(labels_test, predicted_label_QDA_g_classic_test), 
                     adjusted_mutual_info_score(labels_test, predicted_label_QDA_g_M_test      ), 
                     adjusted_mutual_info_score(labels_test, predicted_label_GQDA_test         ), 
                     adjusted_mutual_info_score(labels_test, predicted_label_LDA_t_test        ),
                     adjusted_mutual_info_score(labels_test, predicted_label_QDA_t_test        ),
                     adjusted_mutual_info_score(labels_test, predicted_label_FEMDA_test         )]

    return accuracy_train, ARI_train, NMI_train, AMI_train, accuracy_test, ARI_test, NMI_test, AMI_test

def decision_rules_overall_performance_evaluation(path_dataset, nb_simulations_MC, p_conta, perc_train_set_used, dataset_name, freq_shuffle):
    
    """ Uses the function evalue_performances_on_simulated_data over multiple Monte Carlo simulations to evaluate
        the performances of the different decision rules over multiple samples of the dataset.
        Returns many lists of size nb_simulations_MC containing all the results. Saves all the results
        under dataframe images in the file created for the simulation.

    Parameters
    ----------
    path_dataset        : str
                          path to download and preprocess the dataset
    nb_simulations_MC   : integer > 0
                          number of Monte_Carlo simulations run to average the results
    p_conta             : float between 0 and 1
                          probability to contaminate an observation
    perc_train_set_used : float between 0 and 1
                          percentage of the train set that is used in practice
    dataset_name        : str
                          name of the dataset used, it is useful for the contamination function
    freq_shuffle         : integer > 0
                          if the index of the run modulo freq_shuffle == 0, we resplit the dataset
                          between a train and a test set, to have a different train and test set
    
    Returns
    -------
    vector_accuracy : list of list of floats
                      list of size nb_simulations_MC of all accuracy lists returned by decision_rules_specific_performances_evaluation
    vector_ARI      : list of list of floats
                      list of size nb_simulations_MC of all list of ARI index returned by decision_rules_specific_performances_evaluation
    vector_NMI      : list of list of floats
                      list of size nb_simulations_MC of all list of NMI index returned by decision_rules_specific_performances_evaluation
    vector_AMI      : list of list of floats
                      list of size nb_simulations_MC of all list of AMI index returned by decision_rules_specific_performances_evaluation
    """  
    
    vector_accuracy_train, vector_ARI_train, vector_NMI_train, vector_AMI_train, vector_accuracy_test, vector_ARI_test, vector_NMI_test, vector_AMI_test = [], [], [], [], [], [], [], []
    
    for nb_simu in range(nb_simulations_MC):
    
        start = time.time()
        if nb_simu % freq_shuffle == 0:
            if dataset_name == "Ionosphere":
                X_train, labels_train, X_test, labels_test = pre.ionosphere(path_dataset)
            if dataset_name == "Statlog":
                X_train, labels_train, X_test, labels_test = pre.statlog(path_dataset)
            if dataset_name == "Breast cancer":
                X_train, labels_train, X_test, labels_test = pre.breast_cancer(path_dataset)
            if dataset_name == "Ecoli":
                X_train, labels_train, X_test, labels_test = pre.ecoli(path_dataset)
            if dataset_name == "Spambase":
                X_train, labels_train, X_test, labels_test = pre.spambase(path_dataset)
        X_train_reduced, labels_train_reduced = [], []
        list_index_train_set = pre.select_random_index(len(X_train), int(len(X_train) * perc_train_set_used))
        for i in range(len(X_train)):
            if i in list_index_train_set:
                X_train_reduced.append(X_train[i])
                labels_train_reduced.append(labels_train[i])
        X_train_reduced = np.array(X_train_reduced)
        accuracy_train, ARI_train, NMI_train, AMI_train, accuracy_test, ARI_test, NMI_test, AMI_test = decision_rules_specific_performance_evaluation(X_train_reduced, labels_train_reduced, X_test, labels_test, p_conta, dataset_name)
        vector_accuracy_train.append(accuracy_train)
        vector_ARI_train.append(ARI_train)
        vector_NMI_train.append(NMI_train)
        vector_AMI_train.append(AMI_train)
        vector_accuracy_test.append(accuracy_test)
        vector_ARI_test.append(ARI_test)
        vector_NMI_test.append(NMI_test)
        vector_AMI_test.append(AMI_test)

        finish = time.time()
        print(str(np.round(100 * nb_simu / nb_simulations_MC)) + " %                  " + time_needed(int(finish-start)))
    
    return vector_accuracy_train, vector_ARI_train, vector_NMI_train, vector_AMI_train, vector_accuracy_test, vector_ARI_test, vector_NMI_test, vector_AMI_test
    
def save_results(path, dataset_name, nb_simulations_MC = 10, p_conta = 0, perc_train_set_used = 1, freq_shuffle = 10):
    
    """ Uses the function decision_rules_overall_performance_evaluation to evaluate the performances of the different 
        decision rules. The results are then saved in a dataframe image and the dataframe itself is saved in a pickle
        object.
    """
       
    start = time.time()
    print("Début de l'étude du dataset")
    path_dataset = ""
    for s in path.split("/")[:-2]:
        path_dataset = path_dataset + s + "/"
    path_dataset = path_dataset + "Datasets/"
    vector_accuracy_train, vector_ARI_train, vector_NMI_train, vector_AMI_train, vector_accuracy_test, vector_ARI_test, vector_NMI_test, vector_AMI_test = decision_rules_overall_performance_evaluation(path_dataset, nb_simulations_MC, p_conta, perc_train_set_used, dataset_name, freq_shuffle) 
    finish = time.time()
    print("Fin de l'étude du dataset -                   " + time_needed(int(finish-start)))
        
    restructured_vector_accuracy_train = [[] for i in range(8)]
    restructured_vector_ARI_train      = [[] for i in range(8)]
    restructured_vector_NMI_train      = [[] for i in range(8)]
    restructured_vector_AMI_train      = [[] for i in range(8)]
    
    for j in range(8):
        for k in range(nb_simulations_MC):
            restructured_vector_accuracy_train[j].append(vector_accuracy_train[k][j])
            restructured_vector_ARI_train     [j].append(vector_ARI_train     [k][j])
            restructured_vector_NMI_train     [j].append(vector_NMI_train     [k][j])
            restructured_vector_AMI_train     [j].append(vector_AMI_train     [k][j])
    
    results_array_train = np.zeros((20, 8))
    for j in range(8):
        results_array_train[0 ][j] = np.mean  (np.array(restructured_vector_accuracy_train[j]))
        results_array_train[1 ][j] = np.median(np.array(restructured_vector_accuracy_train[j]))
        results_array_train[2 ][j] = np.std   (np.array(restructured_vector_accuracy_train[j]))
        results_array_train[3 ][j] = np.min   (np.array(restructured_vector_accuracy_train[j]))
        results_array_train[4 ][j] = np.max   (np.array(restructured_vector_accuracy_train[j]))
        results_array_train[5 ][j] = np.mean  (np.array(restructured_vector_ARI_train     [j]))
        results_array_train[6 ][j] = np.median(np.array(restructured_vector_ARI_train     [j]))
        results_array_train[7 ][j] = np.std   (np.array(restructured_vector_ARI_train     [j]))
        results_array_train[8 ][j] = np.min   (np.array(restructured_vector_ARI_train     [j]))
        results_array_train[9 ][j] = np.max   (np.array(restructured_vector_ARI_train     [j]))
        results_array_train[10][j] = np.mean  (np.array(restructured_vector_NMI_train     [j]))
        results_array_train[11][j] = np.median(np.array(restructured_vector_NMI_train     [j]))
        results_array_train[12][j] = np.std   (np.array(restructured_vector_NMI_train     [j]))
        results_array_train[13][j] = np.min   (np.array(restructured_vector_NMI_train     [j]))
        results_array_train[14][j] = np.max   (np.array(restructured_vector_NMI_train     [j]))
        results_array_train[15][j] = np.mean  (np.array(restructured_vector_AMI_train     [j]))
        results_array_train[16][j] = np.median(np.array(restructured_vector_AMI_train     [j]))
        results_array_train[17][j] = np.std   (np.array(restructured_vector_AMI_train     [j]))
        results_array_train[18][j] = np.min   (np.array(restructured_vector_AMI_train     [j]))
        results_array_train[19][j] = np.max   (np.array(restructured_vector_AMI_train     [j]))
        
    lines            = [("Accuracy", "Mean"), ("Accuracy", "Median"), ("Accuracy", "Std"), ("Accuracy", "Min"), ("Accuracy", "Max"),
                        ("ARI     ", "Mean"), ("ARI     ", "Median"), ("ARI     ", "Std"), ("ARI     ", "Min"), ("ARI     ", "Max"),
                        ("NMI     ", "Mean"), ("NMI     ", "Median"), ("NMI     ", "Std"), ("NMI     ", "Min"), ("NMI     ", "Max"),
                        ("AMI     ", "Mean"), ("AMI     ", "Median"), ("AMI     ", "Std"), ("AMI     ", "Min"), ("AMI     ", "Max")]
    df_results_train = pd.DataFrame(results_array_train, index = pd.MultiIndex.from_tuples(lines) , columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_styled        = df_results_train.style.background_gradient(axis = 0)
    dfi.export(df_styled, path + dataset_name + "/Train set/" + "conta = " + str(np.round(p_conta, 2)) + " - data used = " + str(np.round(perc_train_set_used, 2)) + ".png")
    f = open(path + dataset_name + "/Pickles/" + "Train set - conta = " + str(np.round(p_conta, 2)) + " - data used = " + str(np.round(perc_train_set_used, 2)), "wb")
    pk.dump(results_array_train, f)
    f.close()
    
    restructured_vector_accuracy_test = [[] for i in range(8)]
    restructured_vector_ARI_test      = [[] for i in range(8)]
    restructured_vector_NMI_test      = [[] for i in range(8)]
    restructured_vector_AMI_test      = [[] for i in range(8)]
    
    for j in range(8):
        for k in range(nb_simulations_MC):
            restructured_vector_accuracy_test[j].append(vector_accuracy_test[k][j])
            restructured_vector_ARI_test     [j].append(vector_ARI_test     [k][j])
            restructured_vector_NMI_test     [j].append(vector_NMI_test     [k][j])
            restructured_vector_AMI_test     [j].append(vector_AMI_test     [k][j])
    
    results_array_test = np.zeros((20, 8))
    for j in range(8):
        results_array_test[0 ][j] = np.mean  (np.array(restructured_vector_accuracy_test[j]))
        results_array_test[1 ][j] = np.median(np.array(restructured_vector_accuracy_test[j]))
        results_array_test[2 ][j] = np.std   (np.array(restructured_vector_accuracy_test[j]))
        results_array_test[3 ][j] = np.min   (np.array(restructured_vector_accuracy_test[j]))
        results_array_test[4 ][j] = np.max   (np.array(restructured_vector_accuracy_test[j]))
        results_array_test[5 ][j] = np.mean  (np.array(restructured_vector_ARI_test     [j]))
        results_array_test[6 ][j] = np.median(np.array(restructured_vector_ARI_test     [j]))
        results_array_test[7 ][j] = np.std   (np.array(restructured_vector_ARI_test     [j]))
        results_array_test[8 ][j] = np.min   (np.array(restructured_vector_ARI_test     [j]))
        results_array_test[9 ][j] = np.max   (np.array(restructured_vector_ARI_test     [j]))
        results_array_test[10][j] = np.mean  (np.array(restructured_vector_NMI_test     [j]))
        results_array_test[11][j] = np.median(np.array(restructured_vector_NMI_test     [j]))
        results_array_test[12][j] = np.std   (np.array(restructured_vector_NMI_test     [j]))
        results_array_test[13][j] = np.min   (np.array(restructured_vector_NMI_test     [j]))
        results_array_test[14][j] = np.max   (np.array(restructured_vector_NMI_test     [j]))
        results_array_test[15][j] = np.mean  (np.array(restructured_vector_AMI_test     [j]))
        results_array_test[16][j] = np.median(np.array(restructured_vector_AMI_test     [j]))
        results_array_test[17][j] = np.std   (np.array(restructured_vector_AMI_test     [j]))
        results_array_test[18][j] = np.min   (np.array(restructured_vector_AMI_test     [j]))
        results_array_test[19][j] = np.max   (np.array(restructured_vector_AMI_test     [j]))
        
    df_results_test = pd.DataFrame(results_array_test, index = pd.MultiIndex.from_tuples(lines) , columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_styled        = df_results_test.style.background_gradient(axis = 0)
    dfi.export(df_styled, path + dataset_name + "/Test set/" + "conta = " + str(np.round(p_conta, 2)) + " - data used = " + str(np.round(perc_train_set_used, 2)) + ".png")
    f = open(path + dataset_name + "/Pickles/" + "/Test set - conta = " + str(np.round(p_conta, 2)) + " - data used = " + str(np.round(perc_train_set_used, 2)), "wb")
    pk.dump(results_array_test, f)
    f.close()
    
def plot_results_contamination_rate(path, dataset_name, methods, name, conta_min = 0, conta_max = 1, test_set_results = True):
    
    path_pickle     = path + dataset_name + "/Pickles/"
    list_conta      = []
    dict_accuracy   = {"LDA_g - classic" : [], "LDA_g - M" : [],
                       "QDA_g - classic" : [], "QDA_g - M" : [],
                       "GQDA" : [], "LDA_t" : [], "QDA_t" : [],
                       "FEMDA" : []} 
    if test_set_results:
        files = os.listdir(path_pickle)[:int(len(os.listdir(path_pickle))/2)]
    else:
        files = os.listdir(path_pickle)[int(len(os.listdir(path_pickle))/2):]
    files.sort

    for file in files:
        f            = open(path_pickle + file, "rb")
        results      = pk.load(f)
        conta        = float(file.split(" - ")[1][8:])
        dataset_used = float(file.split(" - ")[2][12:])
        if dataset_used > 0.999 and conta >= conta_min and conta <= conta_max:
            list_conta.append(conta)
            dict_accuracy["LDA_g - classic"].append(results[0][0])
            dict_accuracy["LDA_g - M"].append(results[0][1])
            dict_accuracy["QDA_g - classic"].append(results[0][2])
            dict_accuracy["QDA_g - M"].append(results[0][3])
            dict_accuracy["GQDA"].append(results[0][4])
            dict_accuracy["LDA_t"].append(results[0][5])
            dict_accuracy["QDA_t"].append(results[0][6])
            dict_accuracy["FEMDA"].append(results[0][7])
            
    pl.clf()
    for method in methods:
        pl.plot(list_conta, dict_accuracy[method], "s-", label = method)
    pl.legend()
    pl.xlabel("Contamination rate")
    pl.ylabel("Accuracy in %")
    pl.grid()
    pl.savefig(path + dataset_name + "/" + name + ".png")

def plot_results_dataset_used(path, dataset_name, methods, name, dataset_used_min = 0, dataset_used_max = 1, test_set_results = True):
    
    path_pickle       = path + dataset_name + "/Pickles/"
    list_dataset_used = []
    dict_accuracy     = {"LDA_g - classic" : [], "LDA_g - M" : [],
                       "QDA_g - classic" : [], "QDA_g - M" : [],
                       "GQDA" : [], "LDA_t" : [], "QDA_t" : [],
                       "FEMDA" : []} 
    
    if test_set_results:
        files = os.listdir(path_pickle)[:int(len(os.listdir(path_pickle))/2)]
    else:
        files = os.listdir(path_pickle)[int(len(os.listdir(path_pickle))/2):]
    files.sort()
    for file in files:
        f            = open(path_pickle + file, "rb")
        results      = pk.load(f)
        conta        = float(file.split(" - ")[1][8:])
        dataset_used = float(file.split(" - ")[2][12:])
        if conta < 0.001 and dataset_used >= dataset_used_min and dataset_used <= dataset_used_max:
            list_dataset_used.append(dataset_used)
            dict_accuracy["LDA_g - classic"].append(results[0][0])
            dict_accuracy["LDA_g - M"].append(results[0][1])
            dict_accuracy["QDA_g - classic"].append(results[0][2])
            dict_accuracy["QDA_g - M"].append(results[0][3])
            dict_accuracy["GQDA"].append(results[0][4])
            dict_accuracy["LDA_t"].append(results[0][5])
            dict_accuracy["QDA_t"].append(results[0][6])
            dict_accuracy["FEMDA"].append(results[0][7])
            
    pl.clf()
    for method in methods:
        pl.plot(list_dataset_used, dict_accuracy[method], "s-", label = method)
    pl.legend()
    pl.xlabel("Percentage of training set used")
    pl.ylabel("Accuracy in %")
    pl.grid()
    pl.savefig(path + dataset_name + "/" + name + ".png")