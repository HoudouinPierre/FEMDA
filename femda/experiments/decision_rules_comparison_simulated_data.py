from . import estimateurs as est
from . import decision_rules as dr
from . import simulateur as simu
import numpy as np
import pandas as pd
import pickle as pk
import dataframe_image as dfi
import os
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import time

def write_parameters_file(path, m, n, K, priors, p_conta, list_scenarios, simulation_id):

    """ Writes the parameters of the simulation in a file .txt located in the file created to contain the
        results of the simulation. The file is created by the function and is located by the variable path.
    """    

    os.mkdir(path+"Simulation " + str(simulation_id) + " - m="+str(m)+" - K="+str(K)+" - n="+str(n)+" - p_conta="+str(p_conta)) 
    
    fichier = open(path+"Simulation " + str(simulation_id) + " - m="+str(m)+" - K="+str(K)+" - n="+str(n)+" - p_conta="+str(p_conta) + "/" + "Simulation parameters.txt", "w")
    fichier.write("\n")
    fichier.write("                         ########################################################################################################################################## \n")
    fichier.write("                         #######################################################                            ####################################################### \n")
    fichier.write("                         #######################################################         Parameters         ####################################################### \n")
    fichier.write("                         #######################################################                            ####################################################### \n")
    fichier.write("                         ########################################################################################################################################## \n")
    fichier.write("                         ###                                                                                                                                    ### \n")
    fichier.write("                         ###     Dimension                                       : " + str(m) + "                                                                           ### \n")
    fichier.write("                         ###     Number of samples simulated                     : " + str(n) + "                                                                         ### \n")
    fichier.write("                         ###     Number of clusters                              : " + str(K) + "                                                                            ### \n")
    fichier.write("                         ###     Priors on the clusters                          : " + str(priors) + "                                                     ### \n")
    fichier.write("                         ###     Probability of contamination                    : " + str(p_conta) + "                                                                          ### \n")
    fichier.write("                         ###     Liste des scénarios étudiés                     :                                                                              ### \n")
    for scenario in list_scenarios:
        fichier.write("                         ###                                                     : " + scenario + "                                                                ### \n")
    fichier.write("                         ###                                                                                                                                    ### \n")
    fichier.write("                         ########################################################################################################################################## \n")
    fichier.write("                         ########################################################################################################################################## \n")
    fichier.write("                         ########################################################################################################################################## \n")
    fichier.close()

def time_needed(nb_sec):
    
    """ Returns a string with nb_sec converted in hours, minutes and seconds.
    """    
    
    nb_heures = nb_sec // 3600
    nb_min    = (nb_sec - nb_heures * 3600) // 60
    nb_s      = (nb_sec - nb_heures * 3600 - nb_min * 60)
    
    return str(nb_heures) + " h " + str(nb_min) + " min " + str(nb_s) + "s"

def decision_rules_specific_performance_evaluation(X, labels, all_mu, all_sigma):
    
    """ Evaluates the error of the estimators and the performances of the different
        decision rules : accuracy, ARI, NMI and AMI index are computed on the test set.
        The test set is built using the last 80% of the dataset generated.
    
    Parameters
    ----------
    X         : 2-d array of size n*m
                dataset generated
    labels    : list of n integers
                labels of the samples generated
    all_mu    : list containing 1-d array of size m
                real mean of each cluster
    all_sigma : list containing 2-d array of size m*m 
                real covariance matrix of each cluster
    
    Returns
    -------
    errors   : list of floats
               relative error of each estimator
    accuracy : list of floats
               accuracy of each decision rule
    ARI      : list of floats
               ARI index of each decision rule
    NMI      : list of floats
               NMI index of each decision rule
    AMI      : list of floats
               AMI index of each decision rule
    """  

    X_train, labels_train = X[:int(0.20*len(X))], labels[:int(0.20*len(X))]
    X_test , labels_test  = X[int(0.20*len(X)):], labels[int(0.20*len(X)):]

    means_classic, covariances_classic = est.classic_estimator                       (X_train, labels_train)
    means_M      , covariances_M       = est.M_estimator                             (X_train, labels_train)
    means_t      , covariances_t, nus  = est.t_distribution_estimator                (X_train, labels_train)
    means_femda  , covariances_femda   = est.femda_estimator                         (X_train, labels_train)

    c_classic = dr.find_optimal_c(X_train, labels_train, means_classic, covariances_classic)
    c_M       = dr.find_optimal_c(X_train, labels_train, means_M, covariances_M)

    def same_covariance_estimator(covariances):
        
        covariance = 0
        for covariance_cluster in covariances:
            covariance = covariance + covariance_cluster / len(covariances)
            
        return covariance

    avg_mean_error_classic = 0
    avg_mean_error_M       = 0
    avg_mean_error_t       = 0
    avg_mean_error_femda   = 0
    r                      = np.sqrt(np.sum(all_mu[0]**2))
    
    for k in range(len(all_mu)):
        avg_mean_error_classic = avg_mean_error_classic + np.sqrt(np.dot(np.array([all_mu[k]-means_classic[k]]), np.dot(np.linalg.inv(all_sigma[k]), np.array([all_mu[k]-means_classic[k]]).T)))[0][0] / (len(all_mu)*r)
        avg_mean_error_M       = avg_mean_error_M       + np.sqrt(np.dot(np.array([all_mu[k]-means_M      [k]]), np.dot(np.linalg.inv(all_sigma[k]), np.array([all_mu[k]-means_M      [k]]).T)))[0][0] / (len(all_mu)*r) 
        avg_mean_error_t       = avg_mean_error_t       + np.sqrt(np.dot(np.array([all_mu[k]-means_t      [k]]), np.dot(np.linalg.inv(all_sigma[k]), np.array([all_mu[k]-means_t      [k]]).T)))[0][0] / (len(all_mu)*r)
        avg_mean_error_femda   = avg_mean_error_femda   + np.sqrt(np.dot(np.array([all_mu[k]-means_femda  [k]]), np.dot(np.linalg.inv(all_sigma[k]), np.array([all_mu[k]-means_femda  [k]]).T)))[0][0] / (len(all_mu)*r)

    good_classification_LDA_g_classic = 0
    good_classification_LDA_g_M       = 0
    good_classification_QDA_g_classic = 0
    good_classification_QDA_g_M       = 0
    good_classification_GQDA_classic  = 0
    good_classification_GQDA_M        = 0
    good_classification_LDA_t         = 0
    good_classification_QDA_t         = 0
    good_classification_FEMDA         = 0
    
    predicted_label_LDA_g_classic = []
    predicted_label_LDA_g_M       = []
    predicted_label_QDA_g_classic = []
    predicted_label_QDA_g_M       = []
    predicted_label_GQDA_classic  = []
    predicted_label_GQDA_M        = []
    predicted_label_LDA_t         = []
    predicted_label_QDA_t         = []
    predicted_label_FEMDA         = []
    
    for i in range(len(X_test)):
        predicted_label = dr.LDA_g(X_test[i], means_classic, same_covariance_estimator(covariances_classic))
        if labels_test[i] == predicted_label:
            good_classification_LDA_g_classic = good_classification_LDA_g_classic + 1 / len(X_test)
        predicted_label_LDA_g_classic.append(predicted_label)
        
        predicted_label = dr.LDA_g(X_test[i], means_M, same_covariance_estimator(covariances_M))
        if labels_test[i] == predicted_label:
            good_classification_LDA_g_M       = good_classification_LDA_g_M       + 1 / len(X_test)
        predicted_label_LDA_g_M      .append(predicted_label)
            
        predicted_label = dr.QDA_g(X[i], means_classic, covariances_classic)
        if labels[i] == predicted_label:
            good_classification_QDA_g_classic = good_classification_QDA_g_classic + 1 / len(X_test)
        predicted_label_QDA_g_classic.append(predicted_label)
            
        predicted_label = dr.QDA_g(X_test[i], means_M, covariances_M)
        if labels_test[i] == predicted_label:
            good_classification_QDA_g_M       = good_classification_QDA_g_M       + 1 / len(X_test)
        predicted_label_QDA_g_M      .append(predicted_label)
        
        predicted_label = dr.GQDA (X_test[i], means_classic, covariances_classic, c_classic)
        if labels_test[i] == predicted_label:
            good_classification_GQDA_classic  = good_classification_GQDA_classic + 1 / len(X_test)
        predicted_label_GQDA_classic .append(predicted_label)
            
        predicted_label = dr.GQDA (X_test[i], means_M, covariances_M, c_M)
        if labels_test[i] == predicted_label:
            good_classification_GQDA_M        = good_classification_GQDA_M       + 1 / len(X_test)
        predicted_label_GQDA_M       .append(predicted_label)
        
        predicted_label = dr.LDA_t(X_test[i], means_t, same_covariance_estimator(covariances_t), nus)
        if labels_test[i] ==predicted_label:
            good_classification_LDA_t         = good_classification_LDA_t        + 1 / len(X_test)
        predicted_label_LDA_t        .append(predicted_label)
            
        predicted_label= dr.QDA_t(X_test[i], means_t, covariances_t, nus )
        if labels_test[i] == predicted_label:
            good_classification_QDA_t         = good_classification_QDA_t        + 1 / len(X_test)
        predicted_label_QDA_t        .append(predicted_label)
        
        predicted_label = dr.FEMDA(X_test[i], means_femda, covariances_femda)
        if labels_test[i] == predicted_label:
            good_classification_FEMDA         = good_classification_FEMDA        + 1 / len(X_test)
        predicted_label_FEMDA        .append(predicted_label)
        
    if good_classification_GQDA_M > good_classification_GQDA_classic:
        good_classification_GQDA = good_classification_GQDA_M
        predicted_label_GQDA     = predicted_label_GQDA_M
    else:
        good_classification_GQDA = good_classification_GQDA_classic
        predicted_label_GQDA     = predicted_label_GQDA_classic

    errors   = [avg_mean_error_classic, avg_mean_error_M, avg_mean_error_t, avg_mean_error_femda]
    accuracy = [good_classification_LDA_g_classic, good_classification_LDA_g_M,
                good_classification_QDA_g_classic, good_classification_QDA_g_M,
                good_classification_GQDA         , good_classification_LDA_t  , 
                good_classification_QDA_t        , good_classification_FEMDA]
    ARI      = [adjusted_rand_score(labels_test, predicted_label_LDA_g_classic), 
                adjusted_rand_score(labels_test, predicted_label_LDA_g_M      ), 
                adjusted_rand_score(labels_test, predicted_label_QDA_g_classic), 
                adjusted_rand_score(labels_test, predicted_label_QDA_g_M      ), 
                adjusted_rand_score(labels_test, predicted_label_GQDA         ), 
                adjusted_rand_score(labels_test, predicted_label_LDA_t        ),
                adjusted_rand_score(labels_test, predicted_label_QDA_t        ),
                adjusted_rand_score(labels_test, predicted_label_FEMDA        )]
    NMI      = [normalized_mutual_info_score(labels_test, predicted_label_LDA_g_classic), 
                normalized_mutual_info_score(labels_test, predicted_label_LDA_g_M      ), 
                normalized_mutual_info_score(labels_test, predicted_label_QDA_g_classic), 
                normalized_mutual_info_score(labels_test, predicted_label_QDA_g_M      ), 
                normalized_mutual_info_score(labels_test, predicted_label_GQDA         ), 
                normalized_mutual_info_score(labels_test, predicted_label_LDA_t        ),
                normalized_mutual_info_score(labels_test, predicted_label_QDA_t        ),
                normalized_mutual_info_score(labels_test, predicted_label_FEMDA        )]
    AMI      = [adjusted_mutual_info_score(labels_test, predicted_label_LDA_g_classic), 
                adjusted_mutual_info_score(labels_test, predicted_label_LDA_g_M      ), 
                adjusted_mutual_info_score(labels_test, predicted_label_QDA_g_classic), 
                adjusted_mutual_info_score(labels_test, predicted_label_QDA_g_M      ), 
                adjusted_mutual_info_score(labels_test, predicted_label_GQDA         ), 
                adjusted_mutual_info_score(labels_test, predicted_label_LDA_t        ),
                adjusted_mutual_info_score(labels_test, predicted_label_QDA_t        ),
                adjusted_mutual_info_score(labels_test, predicted_label_FEMDA        )] 

    return errors, accuracy, ARI, NMI, AMI

def decision_rules_overall_performance_evaluation(m, n, K, priors, scenario, p_conta, nb_simulations_MC):
    
    """ Uses the function evalue_performances_on_simulated_data over multiple Monte Carlo simulations to evaluate
        the performances of the different estimators and decision rules for given paramereters for the data
        generation. Returns many lists of size nb_simulations_MC containing all the results. Saves all the results
        under dataframe images in the file created for the simulation.
    """
    
    vector_errors, vector_accuracy, vector_ARI, vector_NMI, vector_AMI = [], [], [], [], []
    n, all_pi, all_mu, all_sigma, all_tau, all_PDF, p_conta, conta = simu.genere_parametres_simulation(m, n, K, priors, scenario, p_conta)
    
    for nb_simu in range(nb_simulations_MC):
        
        start = time.time()

        data_simul = simu.dataSimulation(n, all_pi, all_mu, all_sigma, all_tau, all_PDF, p_conta, conta)
        X, labels, _ = data_simul.generateSamples()
        errors, accuracy, ARI, NMI, AMI = decision_rules_specific_performance_evaluation(X, labels, all_mu, all_sigma)
        vector_errors.append(errors)
        vector_accuracy.append(accuracy)
        vector_ARI.append(ARI)
        vector_NMI.append(NMI)
        vector_AMI.append(AMI)

        finish = time.time()
        print(str(np.round(100 * nb_simu / nb_simulations_MC)) + " %                  " + time_needed(int(finish-start)))
    
    return vector_errors, vector_accuracy, vector_ARI, vector_NMI, vector_AMI
    
def save_results(path, m, n, K, priors, list_scenarios, p_conta, simulation_id, nb_simulations_MC = 10):
    
    """ Uses the function evalue_performances_for given_parameters to evaluate the performances of the different 
        estimators and decision rules for given paramereters for the data generation over multiple scenarios.
    """
    
    matrix_errors, matrix_accuracy, matrix_ARI, matrix_NMI, matrix_AMI = [], [], [], [], []
    
    for i in range(len(list_scenarios)):
        
        start = time.time()
        print("Début de l'étude du scénario")
        
        scenario = list_scenarios[i]
        vector_errors, vector_accuracy, vector_ARI, vector_NMI, vector_AMI = decision_rules_overall_performance_evaluation(m, n, K, priors, scenario, p_conta, nb_simulations_MC) 
        matrix_errors.append(vector_errors)
        matrix_accuracy.append(vector_accuracy)
        matrix_ARI.append(vector_ARI)
        matrix_NMI.append(vector_NMI)
        matrix_AMI.append(vector_AMI)
        
        finish = time.time()
        print("Fin de l'étude du scénario - "+ str(1+i) + " / " + str(len(list_scenarios)) + "                  " + time_needed(int(finish-start)))
        
    restructured_matrix_errors   = [[[] for i in range(5 )] for i in range(len(list_scenarios))]
    restructured_matrix_accuracy = [[[] for i in range(11)] for i in range(len(list_scenarios))]
    restructured_matrix_ARI      = [[[] for i in range(11)] for i in range(len(list_scenarios))]
    restructured_matrix_NMI      = [[[] for i in range(11)] for i in range(len(list_scenarios))]
    restructured_matrix_AMI      = [[[] for i in range(11)] for i in range(len(list_scenarios))]
    matrix_mean_errors  , matrix_median_errors  , matrix_std_errors   = np.zeros([len(list_scenarios), 4]), np.zeros([len(list_scenarios), 4]), np.zeros([len(list_scenarios), 4])
    matrix_mean_accuracy, matrix_median_accuracy, matrix_std_accuracy = np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8])
    matrix_mean_ARI     , matrix_median_ARI     , matrix_std_ARI      = np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8])
    matrix_mean_NMI     , matrix_median_NMI     , matrix_std_NMI      = np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8])
    matrix_mean_AMI     , matrix_median_AMI     , matrix_std_AMI      = np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8]), np.zeros([len(list_scenarios), 8])

    for i in range(len(list_scenarios)):
        for j in range(8):
            for k in range(nb_simulations_MC):
                if j < 4:
                    restructured_matrix_errors[i][j].append(matrix_errors[i][k][j])
                restructured_matrix_accuracy[i][j].append(matrix_accuracy[i][k][j])
                restructured_matrix_ARI[i][j].append(matrix_ARI[i][k][j])
                restructured_matrix_NMI[i][j].append(matrix_NMI[i][k][j])
                restructured_matrix_AMI[i][j].append(matrix_AMI[i][k][j])

    for i in range(len(list_scenarios)):
        for j in range(8):
            if j < 4:
                matrix_mean_errors  [i][j] = np.mean  (np.array(restructured_matrix_errors  [i][j]))
                matrix_median_errors[i][j] = np.median(np.array(restructured_matrix_errors  [i][j]))
                matrix_std_errors   [i][j] = np.std   (np.array(restructured_matrix_errors  [i][j]))
            matrix_mean_accuracy  [i][j]   = np.mean  (np.array(restructured_matrix_accuracy[i][j]))
            matrix_median_accuracy[i][j]   = np.median(np.array(restructured_matrix_accuracy[i][j]))
            matrix_std_accuracy   [i][j]   = np.std   (np.array(restructured_matrix_accuracy[i][j]))
            matrix_mean_ARI       [i][j]   = np.mean  (np.array(restructured_matrix_ARI     [i][j]))
            matrix_median_ARI     [i][j]   = np.median(np.array(restructured_matrix_ARI     [i][j]))
            matrix_std_ARI        [i][j]   = np.std   (np.array(restructured_matrix_ARI     [i][j]))
            matrix_mean_NMI       [i][j]   = np.mean  (np.array(restructured_matrix_NMI     [i][j]))
            matrix_median_NMI     [i][j]   = np.median(np.array(restructured_matrix_NMI     [i][j]))
            matrix_std_NMI        [i][j]   = np.std   (np.array(restructured_matrix_NMI     [i][j]))
            matrix_mean_AMI       [i][j]   = np.mean  (np.array(restructured_matrix_AMI     [i][j]))
            matrix_median_AMI     [i][j]   = np.median(np.array(restructured_matrix_AMI     [i][j]))
            matrix_std_AMI        [i][j]   = np.std   (np.array(restructured_matrix_AMI     [i][j]))

    df_mean_errors     = pd.DataFrame(matrix_mean_errors    , index = [scenario for scenario in list_scenarios], columns = ['Classic estimator', 'M-estimator', 't-distribution estimator', 'FEMDA estimator'])
    df_median_errors   = pd.DataFrame(matrix_median_errors  , index = [scenario for scenario in list_scenarios], columns = ['Classic estimator', 'M-estimator', 't-distribution estimator', 'FEMDA estimator'])
    df_std_errors      = pd.DataFrame(matrix_std_errors     , index = [scenario for scenario in list_scenarios], columns = ['Classic estimator', 'M-estimator', 't-distribution estimator', 'FEMDA estimator'])
    df_mean_accuracy   = pd.DataFrame(matrix_mean_accuracy  , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_median_accuracy = pd.DataFrame(matrix_median_accuracy, index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_std_accuracy    = pd.DataFrame(matrix_std_accuracy   , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_mean_ARI        = pd.DataFrame(matrix_mean_ARI       , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_median_ARI      = pd.DataFrame(matrix_median_ARI     , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_std_ARI         = pd.DataFrame(matrix_std_ARI        , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_mean_NMI        = pd.DataFrame(matrix_mean_NMI       , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_median_NMI      = pd.DataFrame(matrix_median_NMI     , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_std_NMI         = pd.DataFrame(matrix_std_NMI        , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_mean_AMI        = pd.DataFrame(matrix_mean_AMI       , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_median_AMI      = pd.DataFrame(matrix_median_AMI     , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])
    df_std_AMI         = pd.DataFrame(matrix_std_AMI        , index = [scenario for scenario in list_scenarios], columns = ['LDA_g - Classic', 'LDA_g - M', 'QDA_g - Classic', 'QDA_g - M', 'GQDA', 'LDA_t', 'QDA_t', 'FEMDA'])

    path2 = path + "Simulation " + str(simulation_id) + " - m="+str(m)+" - K="+str(K)+" - n="+str(n)+" - p_conta="+str(p_conta)+ "/"

    f = open(path2 + "Matrix errors", "wb")
    pk.dump(restructured_matrix_errors, f)
    f.close()
    f = open(path2 + "Matrix Accuracy", "wb")
    pk.dump(restructured_matrix_accuracy, f)
    f.close()
    f = open(path2 + "Matrix ARI", "wb")
    pk.dump(restructured_matrix_ARI, f)
    f.close()
    f = open(path2 + "Matrix NMI", "wb")
    pk.dump(restructured_matrix_NMI, f)
    f.close()
    f = open(path2 + "Matrix AMI", "wb")
    pk.dump(restructured_matrix_AMI, f)
    f.close()
    
    df_styled = df_mean_errors.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Errors - mean.png")
    df_styled = df_median_errors.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Errors - median.png")
    df_styled = df_std_errors.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Errors - standard deviation.png")
    df_styled = df_mean_accuracy.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Accuracy - mean.png")
    df_styled = df_median_accuracy.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Accuracy - median.png")
    df_styled = df_std_accuracy.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - Accuracy - standard deviation.png")
    df_styled = df_mean_ARI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - ARI - mean.png")
    df_styled = df_median_ARI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - ARI - median.png")
    df_styled = df_std_ARI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - ARI - standard deviation.png")
    df_styled = df_mean_NMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - NMI - mean.png")
    df_styled = df_median_NMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - NMI - median.png")
    df_styled = df_std_NMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - NMI - standard deviation.png")
    df_styled = df_mean_AMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - AMI - mean.png")
    df_styled = df_median_AMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - AMI - median.png")
    df_styled = df_std_AMI.style.background_gradient(axis = 0)
    dfi.export(df_styled, path2 + "Comparaison des datasets - AMI - standard deviation.png")
    
    df_styled = df_mean_errors.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Errors - mean.png")
    df_styled = df_median_errors.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Errors - median.png")
    df_styled = df_std_errors.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Errors - standard deviation.png")
    df_styled = df_mean_accuracy.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Accuracy - mean.png")
    df_styled = df_median_accuracy.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Accuracy - median.png")
    df_styled = df_std_accuracy.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - Accuracy - standard deviation.png")
    df_styled = df_mean_ARI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - ARI - mean.png")
    df_styled = df_median_ARI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - ARI - median.png")
    df_styled = df_std_ARI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - ARI - standard deviation.png")
    df_styled = df_mean_NMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - NMI - mean.png")
    df_styled = df_median_NMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - NMI - median.png")
    df_styled = df_std_NMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - NMI - standard deviation.png")
    df_styled = df_mean_AMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - AMI - mean.png")
    df_styled = df_median_AMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - AMI - median.png")
    df_styled = df_std_AMI.style.background_gradient(axis = 1)
    dfi.export(df_styled, path2 + "Comparaison des decision rules - AMI - standard deviation.png")
    
    print("La liste des scénarios a été traitée et sauvegardée")