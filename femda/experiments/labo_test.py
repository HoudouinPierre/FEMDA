from . import decision_rules_comparison_simulated_data as drcs
from . import decision_rules_comparison_real_data as drcr

simulation_gi                        = False
etude_ionosphere                     = True
etude_statlog                        = False
etude_breast_cancer                  = False
etude_ecoli                          = False
etude_spambase                       = False
visualization                        = True

# "0-0.605-0.295-0.1-0 ; 1-0-0-0-0 ; 0-0.5-0.5-0-0 : 2"
path_results_simulated_data = "C:/Users/a/Desktop/Stage Robust QDA_LDA L2S/TEST GIT/Simulations données simulées/"
path_results_real_data      = "C:/Users/a/Desktop/Stage Robust QDA_LDA L2S/TEST GIT/Simulations données réelles/"
path_dataset                = "C:/Users/a/Desktop/Stage Robust QDA_LDA L2S/TEST GIT/Datasets/"

nb_simulations_MC   = 100
p_conta             = 0.
perc_train_set_used = 1
freq_shuffle        = 10
pas                 = 0.1


def labo_test(path_results_simulated_data=path_results_simulated_data,
              path_results_real_data=path_results_real_data,
              path_dataset=path_dataset
             ):

    ####################################################################################################################################
    ###################################                                                              ###################################
    ###################################                  Simulations données simulées                ###################################
    ###################################                                                              ###################################
    ####################################################################################################################################

    if simulation_gi:
    
        n                 = 25000 # 80% for train set and 20% for test set
        K                 = 5
        priors            = [1/K for i in range(K)]
        m                 = 10
        simulation_id     = 1
        list_scenarios    = ["1-0-0-0-0 : 1",
                            "0-1-0-0-0 : 1",
                            "0-0-1-0-0 : 1",
                            "0-0-0-1-0 : 1",
                            "0-0-0-0-1 : 1",
                            "1-0-0-0-0 : 2",
                            "0-1-0-0-0 : 2",
                            "0-0-1-0-0 : 2",
                            "0-0-0-1-0 : 2",
                            "0-0-0-0-1 : 2",
                            "1-0-0-0-0 : 3",
                            "0-1-0-0-0 : 3",
                            "0-0-1-0-0 : 3",
                            "0-0-0-1-0 : 3",
                            "0-0-0-0-1 : 3",
                            "0-0.5-0-0.5-0 : 1",
                            "0-0.5-0-0.5-0 : 2",
                            "0-0.5-0-0.5-0 : 3",
                            "0-0.34-0-0.33-0.33 : 1",
                            "0-0.34-0-0.33-0.33 : 2",
                            "0-0.34-0-0.33-0.33 : 3",
                            "0-0.25-0.25-0.25-0.25 : 1",
                            "0-0.25-0.25-0.25-0.25 : 2",
                            "0-0.25-0.25-0.25-0.25 : 3",
                            "0-0.25-0.5-0.25-0 : 3"
                            ]
        
        drcs.write_parameters_file(path_results_simulated_data, m, n, K, priors, 0.00, list_scenarios, simulation_id)
        drcs.save_results(path_results_simulated_data, m, n, K, priors, list_scenarios, 0.00, simulation_id, nb_simulations_MC)

        drcs.write_parameters_file(path_results_simulated_data, m, n, K, priors, 0.10, list_scenarios, simulation_id)
        drcs.save_results(path_results_simulated_data, m, n, K, priors, list_scenarios, 0.10, simulation_id, nb_simulations_MC)

        drcs.write_parameters_file(path_results_simulated_data, m, n, K, priors, 0.25, list_scenarios, simulation_id)
        drcs.save_results(path_results_simulated_data, m, n, K, priors, list_scenarios, 0.25, simulation_id, nb_simulations_MC)

    ####################################################################################################################################
    ###################################                                                              ###################################
    ###################################                  Simulations données réelles                 ###################################
    ###################################                                                              ###################################
    ####################################################################################################################################

    if etude_ionosphere:
        dataset_name = "Ionosphere"
        for P_conta in [i * pas for i in range(int(1/pas) + 1)]:
            drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = P_conta, perc_train_set_used = perc_train_set_used)
        #for Perc_train_set_used in [i * pas for i in range(1, int(1/pas))]:
            #drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = p_conta, perc_train_set_used = Perc_train_set_used)
                
    if etude_statlog:
        dataset_name = "Statlog"
        for P_conta in [i * pas for i in range(int(1/pas) + 1)]:
            drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = P_conta, perc_train_set_used = perc_train_set_used)
        #for Perc_train_set_used in [i * pas for i in range(1, int(1/pas))]:
            #drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = p_conta, perc_train_set_used = Perc_train_set_used)

    if etude_breast_cancer:
        dataset_name = "Breast cancer"
        for P_conta in [i * pas for i in range(int(1/pas) + 1)]:
            drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = P_conta, perc_train_set_used = perc_train_set_used)
        #for Perc_train_set_used in [i * pas for i in range(1, int(1/pas))]:
            #drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = p_conta, perc_train_set_used = Perc_train_set_used)
            
    if etude_ecoli:
        dataset_name = "Ecoli"
        for P_conta in [i * pas for i in range(int(1/pas) + 1)]:
            drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = P_conta, perc_train_set_used = perc_train_set_used)
        #for Perc_train_set_used in [i * pas for i in range(1, int(1/pas))]:
            #drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = p_conta, perc_train_set_used = Perc_train_set_used)
            
    if etude_spambase:
        dataset_name = "Spambase"
        for P_conta in [i * pas for i in range(int(1/pas) + 1)]:
            drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = P_conta, perc_train_set_used = perc_train_set_used)
        #for Perc_train_set_used in [i * pas for i in range(1, int(1/pas))]:
            #drcr.save_results(path_results_real_data, dataset_name, nb_simulations_MC = nb_simulations_MC, freq_shuffle = freq_shuffle, p_conta = p_conta, perc_train_set_used = Perc_train_set_used)
                
    ####################################################################################################################################
    ###################################                                                              ###################################
    ###################################                     Tracé des résultats                      ###################################
    ###################################                                                              ###################################
    ####################################################################################################################################     
        
    if visualization :   
        conta_min         = 0.
        conta_max         = 1
        dataset_used_min  = 0.1
        dataset_used_max  = 1
        methods1          = ["LDA_g - classic", "LDA_g - M", "QDA_g - classic", "LDA_t"]
        methods2          = [ "QDA_g - M", "GQDA", "QDA_t", "FEMDA"]
        
        dataset_name      = "Ionosphere"
        name_conta        = "evolution - contamination - All results"
        name_dataset_used = "evolution - dataset used - All results"
        
        drcr.plot_results_contamination_rate(path_results_real_data, dataset_name, methods2, name_conta, conta_min = conta_min, conta_max = conta_max)
        #drcr.plot_results_dataset_used(path_results_real_data, dataset_name, methods2, name_dataset_used, dataset_used_min = dataset_used_min, dataset_used_max = dataset_used_max)


if __name__ == "__main__":
    labo_test()