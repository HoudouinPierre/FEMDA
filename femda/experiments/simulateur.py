from numpy.random import multivariate_normal, gamma, wald
from scipy.linalg import sqrtm
import numpy as np

I_WANT_FIXED_PARAMETERS_FOR_SIMULATIONS = True
# Works with K = 5 and m = 10

SIGMA0 = np.array([[ 0.65963099, -0.23174501, -0.37255014,  0.16047307, -0.06986632,
                    -0.20351644, -0.04244815, -0.20316376,  0.01801177, -0.12203187],
                   [-0.23174501,  0.77008355,  0.2700138 , -0.05366752,  0.11668053,
                    0.01736836,  0.38286752, -0.43575001,  0.15860259,  0.11176599],
                   [-0.37255014,  0.2700138 ,  0.80912915,  0.1266928 ,  0.28429233,
                    0.21658211, -0.15497937, -0.01667027, -0.11857219, -0.2050802 ],
                   [ 0.16047307, -0.05366752,  0.1266928 ,  1.07968243,  0.13513597,
                    0.04036425,  0.15355428, -0.19240103, -0.02517887,  0.05847   ],
                   [-0.06986632,  0.11668053,  0.28429233,  0.13513597,  0.70265271,
                    -0.19692394, -0.15044429, -0.02987165, -0.26396171,  0.070615  ],
                   [-0.20351644,  0.01736836,  0.21658211,  0.04036425, -0.19692394,
                    0.97534942, -0.02522256, -0.07920685, -0.1409119 ,  0.08512524],
                   [-0.04244815,  0.38286752, -0.15497937,  0.15355428, -0.15044429,
                    -0.02522256,  1.21658996, -0.07048257,  0.15690868, -0.16288668],
                   [-0.20316376, -0.43575001, -0.01667027, -0.19240103, -0.02987165,
                    -0.07920685, -0.07048257,  1.2744286 ,  0.02920179,  0.25563273],
                   [ 0.01801177,  0.15860259, -0.11857219, -0.02517887, -0.26396171,
                    -0.1409119 ,  0.15690868,  0.02920179,  1.38021804, -0.12277992],
                   [-0.12203187,  0.11176599, -0.2050802 ,  0.05847   ,  0.070615  ,
                    0.08512524, -0.16288668,  0.25563273, -0.12277992,  1.13223515]])

SIGMA1 = np.array([[ 1.57255113,  0.15394067,  0.05463296,  0.00341056,  0.11387236,
                    0.07881721, -0.42883195, -0.07760481,  0.13847189, -0.07038395],
                   [ 0.15394067,  0.94004185, -0.01871733,  0.0551    , -0.10265995,
                    0.03227707, -0.1653636 , -0.13222739,  0.02981121,  0.01419475],
                   [ 0.05463296, -0.01871733,  0.76406987,  0.03940517, -0.02125184,
                    0.00638847, -0.07361563,  0.00727309,  0.08105846, -0.12739615],
                   [ 0.00341056,  0.0551    ,  0.03940517,  0.96789186,  0.3015674 ,
                    0.01943675, -0.26457125,  0.36272305, -0.11250757,  0.07590622],
                   [ 0.11387236, -0.10265995, -0.02125184,  0.3015674 ,  1.12694743,
                    0.27093418, -0.23708886,  0.25502555,  0.0948158 ,  0.01077614],
                   [ 0.07881721,  0.03227707,  0.00638847,  0.01943675,  0.27093418,
                    1.10353602,  0.04659414,  0.46909059,  0.03359708,  0.20640832],
                   [-0.42883195, -0.1653636 , -0.07361563, -0.26457125, -0.23708886,
                    0.04659414,  0.82719324,  0.19670008, -0.10413831, -0.01143578],
                   [-0.07760481, -0.13222739,  0.00727309,  0.36272305,  0.25502555,
                    0.46909059,  0.19670008,  0.79450201, -0.12368953, -0.21250651],
                   [ 0.13847189,  0.02981121,  0.08105846, -0.11250757,  0.0948158 ,
                    0.03359708, -0.10413831, -0.12368953,  0.39661602,  0.23270783],
                   [-0.07038395,  0.01419475, -0.12739615,  0.07590622,  0.01077614,
                    0.20640832, -0.01143578, -0.21250651,  0.23270783,  1.50665057]])

SIGMA2 = np.array([[ 0.74616095, -0.14289427,  0.24475873, -0.34032334, -0.46570708,
                    0.13192312, -0.0472028 , -0.08081196,  0.03042543,  0.2510243 ],
                   [-0.14289427,  1.59983138,  0.11662554,  0.21404798, -0.15736453,
                    0.30960642,  0.572066  , -0.1396824 ,  0.33606045,  0.00773204],
                   [ 0.24475873,  0.11662554,  1.35307178, -0.26712472,  0.38760697,
                    0.5444736 , -0.19175407, -0.07336361, -0.14210624,  0.11434187],
                   [-0.34032334,  0.21404798, -0.26712472,  1.033906  ,  0.29934718,
                    -0.17624372, -0.11972883,  0.2397499 ,  0.20891969,  0.18148452],
                   [-0.46570708, -0.15736453,  0.38760697,  0.29934718,  1.24835245,
                    0.22939918, -0.23697436,  0.21181617,  0.0595435 ,  0.18773035],
                   [ 0.13192312,  0.30960642,  0.5444736 , -0.17624372,  0.22939918,
                    0.74671528, -0.00181501, -0.05344971,  0.01432925, -0.10097918],
                   [-0.0472028 ,  0.572066  , -0.19175407, -0.11972883, -0.23697436,
                    -0.00181501,  0.47972939,  0.0031951 ,  0.09609802,  0.00612826],
                   [-0.08081196, -0.1396824 , -0.07336361,  0.2397499 ,  0.21181617,
                    -0.05344971,  0.0031951 ,  0.67084171,  0.04583288,  0.14154079],
                   [ 0.03042543,  0.33606045, -0.14210624,  0.20891969,  0.0595435 ,
                    0.01432925,  0.09609802,  0.04583288,  0.91453598,  0.35854144],
                   [ 0.2510243 ,  0.00773204,  0.11434187,  0.18148452,  0.18773035,
                    -0.10097918,  0.00612826,  0.14154079,  0.35854144,  1.20685509]])

SIGMA3 = np.array([[ 0.68654009, -0.18871367,  0.00418124, -0.2736131 , -0.23854285,
                    0.11708568, -0.17460049,  0.09652099, -0.06888699,  0.07304049],
                   [-0.18871367,  0.73127449,  0.17724311,  0.02935562, -0.09440512,
                    0.30058656,  0.34717253,  0.10387657,  0.364108  , -0.00674574],
                   [ 0.00418124,  0.17724311,  1.13667454, -0.00905685,  0.24217548,
                    0.44949837,  0.08138781,  0.0207203 ,  0.28462587, -0.15617147],
                   [-0.2736131 ,  0.02935562, -0.00905685,  0.91970987, -0.19524422,
                    0.14813278, -0.06289064, -0.16049415, -0.01612038, -0.26884326],
                   [-0.23854285, -0.09440512,  0.24217548, -0.19524422,  0.63638707,
                    -0.26494111,  0.14423224,  0.2617986 , -0.06097454, -0.13733077],
                   [ 0.11708568,  0.30058656,  0.44949837,  0.14813278, -0.26494111,
                    0.83568667, -0.23482211,  0.10365356,  0.00956471, -0.19038602],
                   [-0.17460049,  0.34717253,  0.08138781, -0.06289064,  0.14423224,
                    -0.23482211,  1.18284553, -0.28575775,  0.01723174, -0.4623737 ],
                   [ 0.09652099,  0.10387657,  0.0207203 , -0.16049415,  0.2617986 ,
                    0.10365356, -0.28575775,  1.05365748, -0.42985385, -0.08982747],
                   [-0.06888699,  0.364108  ,  0.28462587, -0.01612038, -0.06097454,
                    0.00956471,  0.01723174, -0.42985385,  1.82280996, -0.01411021],
                   [ 0.07304049, -0.00674574, -0.15617147, -0.26884326, -0.13733077,
                    -0.19038602, -0.4623737 , -0.08982747, -0.01411021,  0.99441431]])

SIGMA4 = np.array([[ 0.79614009, -0.15534088, -0.3745037 , -0.1634612 ,  0.08233212,
                    -0.04322898,  0.05513867, -0.0729146 ,  0.1232276 ,  0.09514593],
                   [-0.15534088,  0.77474391,  0.36996305,  0.11754211, -0.1706926 ,
                    -0.07565772,  0.13957162,  0.21140293,  0.06393028,  0.00444412],
                   [-0.3745037 ,  0.36996305,  1.2007165 ,  0.06394929, -0.47870594,
                    -0.25006592, -0.28264067,  0.13747703, -0.08897225, -0.14165621],
                   [-0.1634612 ,  0.11754211,  0.06394929,  1.04927075, -0.03410715,
                    0.37253947, -0.114177  ,  0.26939607, -0.12586309,  0.18616308],
                   [ 0.08233212, -0.1706926 , -0.47870594, -0.03410715,  0.86659916,
                    -0.00596463,  0.03910985, -0.16473423,  0.04261439,  0.07442695],
                   [-0.04322898, -0.07565772, -0.25006592,  0.37253947, -0.00596463,
                    1.24058473, -0.19709553, -0.13078999, -0.28692008,  0.09286908],
                   [ 0.05513867,  0.13957162, -0.28264067, -0.114177  ,  0.03910985,
                    -0.19709553,  0.87597244,  0.13034726,  0.4095738 ,  0.31523726],
                   [-0.0729146 ,  0.21140293,  0.13747703,  0.26939607, -0.16473423,
                    -0.13078999,  0.13034726,  0.94480859,  0.22053224,  0.19272972],
                   [ 0.1232276 ,  0.06393028, -0.08897225, -0.12586309,  0.04261439,
                    -0.28692008,  0.4095738 ,  0.22053224,  1.17925115,  0.3258996 ],
                   [ 0.09514593,  0.00444412, -0.14165621,  0.18616308,  0.07442695,
                    0.09286908,  0.31523726,  0.19272972,  0.3258996 ,  1.07191267]])

SIGMA_NOISE_0 = np.array([[ 0.70387844,  0.14173733, -0.1872618 , -0.16934332, -0.0779969 ,
                          0.01233009,  0.22669491,  0.13406542,  0.02045725, -0.60579917],
                          [ 0.14173733,  1.77504211, -0.17394353, -0.48658065, -0.23040451,
                         -0.48490723,  0.05100652,  0.04386135, -0.02668856, -0.41524843],
                          [-0.1872618 , -0.17394353,  1.16927814,  0.10914491, -0.01737274,
                          0.13384749, -0.10386102, -0.45846455,  0.86628261, -0.32060205],
                          [-0.16934332, -0.48658065,  0.10914491,  1.075194  ,  0.462886  ,
                          0.3316134 , -0.2486594 , -0.16670795, -0.09845273,  0.34838196],
                          [-0.0779969 , -0.23040451, -0.01737274,  0.462886  ,  0.55475284,
                         -0.25200362, -0.10616487,  0.10608942, -0.22494921,  0.06748856],
                          [ 0.01233009, -0.48490723,  0.13384749,  0.3316134 , -0.25200362,
                          1.14017806, -0.09850892, -0.24585623,  0.33054262,  0.15891042],
                          [ 0.22669491,  0.05100652, -0.10386102, -0.2486594 , -0.10616487,
                         -0.09850892,  0.27150049,  0.15222821,  0.04563598, -0.26080494],
                          [ 0.13406542,  0.04386135, -0.45846455, -0.16670795,  0.10608942,
                         -0.24585623,  0.15222821,  0.65093622, -0.44480501,  0.17001313],
                          [ 0.02045725, -0.02668856,  0.86628261, -0.09845273, -0.22494921,
                          0.33054262,  0.04563598, -0.44480501,  1.48565505, -0.37306758],
                          [-0.60579917, -0.41524843, -0.32060205,  0.34838196,  0.06748856,
                          0.15891042, -0.26080494,  0.17001313, -0.37306758,  1.17358465]])

SIGMA_NOISE_1 = np.array([[ 0.71380881, -0.22519285, -0.48650475, -0.48859699, -0.03111683,
                        -0.23206183,  0.23228126, -0.2687057 ,  0.34174352, -0.35568404],
                         [-0.22519285,  0.81695701,  0.12153592, -0.23279644, -0.06985542,
                         0.01058409,  0.0554797 , -0.2229638 , -0.06271049, -0.34301576],
                         [-0.48650475,  0.12153592,  0.93295689,  0.3588545 ,  0.22169986,
                         0.19905399, -0.38066591, -0.10445448, -0.48790529,  0.10227753],
                         [-0.48859699, -0.23279644,  0.3588545 ,  1.25567426, -0.22228897,
                         0.49895338, -0.06066179,  0.39322836, -0.50709515,  0.65615351],
                         [-0.03111683, -0.06985542,  0.22169986, -0.22228897,  2.13340116,
                        -0.88626188, -0.19748381, -0.01316109, -0.39868582,  0.33222362],
                         [-0.23206183,  0.01058409,  0.19905399,  0.49895338, -0.88626188,
                         0.85506613,  0.03975364,  0.07713491, -0.14040749,  0.17435679],
                         [ 0.23228126,  0.0554797 , -0.38066591, -0.06066179, -0.19748381,
                         0.03975364,  0.40687872, -0.19462902,  0.04109253, -0.13466775],
                         [-0.2687057 , -0.2229638 , -0.10445448,  0.39322836, -0.01316109,
                         0.07713491, -0.19462902,  0.75310185, -0.10314714,  0.3866746 ],
                         [ 0.34174352, -0.06271049, -0.48790529, -0.50709515, -0.39868582,
                        -0.14040749,  0.04109253, -0.10314714,  0.83657234, -0.17576316],
                         [-0.35568404, -0.34301576,  0.10227753,  0.65615351,  0.33222362,
                         0.17435679, -0.13466775,  0.3866746 , -0.17576316,  1.29558282]])

MU0 = np.array([-0.13040322,  0.21831241,  0.13650351,  0.43166859, -0.37257364,
                0.6214003 ,  0.02152636,  0.33358624,  0.306053  , -0.00162893])

MU1 = np.array([ 0.06371455,  0.43615313, -0.21163921, -0.31489917,  0.23063918,
                0.50978355,  0.36228166, -0.1824809 ,  0.42808702, -0.02964434])

MU2 = np.array([ 0.16112972, -0.32765945,  0.00568319,  0.44179632,  0.21672135,
                0.29812011, -0.13066803,  0.51344744, -0.10274407, -0.49432552])

MU3 = np.array([ 0.52828442,  0.03491522,  0.18162774,  0.31647269,  0.24746236,
                -0.48090486, -0.10598252,  0.39150647,  0.26663308,  0.24174984])

MU4 = np.array([ 0.12424547,  0.04525731, -0.23328742,  0.22147227,  0.003485  ,
             -0.20504156, -0.06600664,  0.07885775, -0.9089108 , -0.0171292 ])

MU_NOISE_0 = np.array([ 0.09142525, -0.21008614,  0.12088316, -0.1330825 , -0.22217068,
                       -0.4905775 , -0.07622752, -0.54425252, -0.36449634,  0.43620687])

MU_NOISE_1 = np.array([-0.07642326, -0.21307132,  0.39790428, -0.4972497 , -0.07474425,
                       -0.10843697, -0.18178622, -0.4420889 ,  0.54399567,  0.03754497])

def genereRandomCovarianceMatrix(m, shape=1):
    
    """ Randomly generates a covariance matrix with Tr = m by first generating random eigenvalues and then 
        a random orthogonal matrix. The orthogonal matrix is drawn uniformly on O(m) and the 
        eigenvalues are drawn with a truncated N(1, shape**2).
    
    Parameters
    ----------
    m     : integer > 0
            dimension of the data
    shape : float 
            Standard deviation of the gaussian distribution of eigenvalues 
    Returns
    -------
    sigma : 2-d array, positive-definite matrix
            random covariance matrix with controlled eigenvalues and Tr = m
    """
    
    stretch = 1 # parameter to stretch the covariance matrix 
    sigma_diag = np.diag(shape*np.random.rand(m))
    for i in range(m):
        rnd = np.random.rand()
        if 0.00 < rnd < 0.25:
            sigma_diag[i][i] = sigma_diag[i][i] / stretch
        if 0.25 < rnd < 0.50:
            sigma_diag[i][i] = sigma_diag[i][i] / (2 * stretch)
        if 0.50 < rnd < 0.75:
            sigma_diag[i][i] = sigma_diag[i][i] * stretch
        if 0.75 < rnd < 1.00:
            sigma_diag[i][i] = sigma_diag[i][i] * 2 * stretch
            
    u, s, vh   = np.linalg.svd(np.random.randn(m, m), full_matrices=False)
    mat_rot    = np.dot(u, vh)
    sigma      = np.dot(mat_rot, np.dot(sigma_diag, mat_rot.T))

    return sigma * m / np.matrix.trace(sigma)

def genere_all_mu(m, K, r=1):
    
    """ Randomly generates the centers of the clusters on the m-dimensional r-sphere.
    
    Parameters
    ----------
    m      : integer > 0
             dimension of the data
    K      : integer > 0
             number of clusters
    r      : float > 0
             radius of the sphere where the centers are randomly drawn
    Returns
    -------
    all_mu         : 2-d array of size K*m
                     Matrix of the mean vectors of size m of the K clusters
    """
    
    all_mu = []
    for k in range(K):
         all_mu.append(random_sphere_point(m)*r)
         
    if I_WANT_FIXED_PARAMETERS_FOR_SIMULATIONS:
        return np.array([MU0, MU1, MU2, MU3, MU4])
    else:
        return np.array(all_mu)

def genere_all_sigma(m, K):
    
    """ Randomly generates the shape matrix of the clusters.
    
    Parameters
    ----------
    m      : integer > 0
             dimension of the data
    K      : integer > 0
             number of clusters
    Returns
    -------
    all_sigma      : 3-d array of size K*m*m
                     Tensor of the shape matrix of the K clusters
    """
    
    all_sigma = []
    for k in range(K):
        all_sigma.append(genereRandomCovarianceMatrix(m))
        
    if I_WANT_FIXED_PARAMETERS_FOR_SIMULATIONS:
        return np.array([SIGMA0, SIGMA1, SIGMA2, SIGMA3, SIGMA4])  
    else:       
        return np.array(all_sigma)

def genere_all_PDF(scenario, K, test_real_shift, range_beta = 10, range_nu = 10, q = 1000):
    
    """ Randomly generates the matrix of the eventually identical K*q distributions that will be used
        to generate the points of each cluster. Row k of the matrix corresponds to the q distributions
        available to generate the points of cluster k. There are four families of disributions available :
            -> Generalized gaussian distributions
            -> Inverse gaussian distributions
            -> t-distributions
            -> k-distributions
        distributions are generated according to a scenario. For example,  "0-0.5-0-0.5-0 : 1" means :
            -> 0%  of multivariate classic gaussian distributions
            -> 50% of multivariate generalized gaussian distributions
            -> 0%  of multivariate inverse gaussian distributions
            -> 50% of multivariate t-distributions
            -> 0%  of multivariate k-distributions
            
            -> 1 : parameters for all distributions families will be the same for all the points of all clusters
            -> 2 : parameters for all distributions families will be the same for all the points of the same clusters
            -> 3 : parameters for all distributions families will be different for all the points
        
        finally, it is possible to combine different scenarios for the clusters by concatenating different mixture with
        a ; such as "0-0.25-0.25-0.25-0.25 ; 0-0.5-0-0.5-0 ; 0-0.34-0-0.33-0.33 ; 0-1-0-0-0 : 3".
            
    Parameters
    ----------
    scenario   : str
                 scenario used to generate the data
    K          : integer > 0
                 number of clusters
    range_beta : integer >=0
                 beta parameter for generalized and inverse gaussian distribution families are drawn in [1 ; 1 + range_beta]
    range_nu   : integer >=0
                 nu parameter for generalized and inverse gaussian distribution families are drawn in [1 ; 1 + range_beta]
    q          : integer >=0
                 number of distributions used to generate the points of one cluster
    Returns
    -------
    all_PDF    : 2-d array of distributions of size K*q
                 matrix of the eventually identical K*q distributions used to generate the points of all the clusters
    """
    
    type_melanges, parametres = scenario.split(" : ")[0], int(scenario.split(" : ")[1])
    types_clusters    = type_melanges.split(" ; ")
    nb_types_clusters = len(types_clusters)
    
    if parametres == 1:
        matrix_beta = np.ones([K, q]) * (0.25 + np.random.rand() * range_beta)
        matrix_nu   = np.ones([K, q]) * (1 + np.random.rand() * range_nu)
    if parametres == 2:
        matrix_beta = 0.25 + np.random.rand(K, 1) @ np.ones([1, q]) * range_beta
        matrix_nu   = 1 + np.random.rand(K, 1) @ np.ones([1, q]) * range_nu
    if parametres == 3:
        matrix_beta = 0.25 + np.random.rand(K, q) * range_beta
        matrix_nu   = 1 + np.random.rand(K, q) * range_nu
    
    def genere_cluster_PDF(type_cluster):
        
        a, b, c, d, _ = float(type_cluster.split("-")[0]), float(type_cluster.split("-")[1]), float(type_cluster.split("-")[2]), float(type_cluster.split("-")[3]), float(type_cluster.split("-")[4])
        rnd = np.random.rand(q)
        return [0*(rnd[j]<a) + 1*(a<=rnd[j]<a+b) + 2*(a+b<=rnd[j]<a+b+c) + 3*(a+b+c<=rnd[j]<a+b+c+d) + 4*(a+b+c+d <= rnd[j]) for j in range(q)]
    
    matrix_PDF = [genere_cluster_PDF(types_clusters[np.random.randint(nb_types_clusters)]) for i in range(K)]
    all_PDF    = [[lambda mu, sigma, tau, nu = matrix_nu[i][j], beta = matrix_beta[i][j], PDF = matrix_PDF[i][j] : multivariate_generalized_gaussian(mu, sigma, tau, 1) * (PDF == 0) + multivariate_generalized_gaussian(mu, sigma, tau, beta) * (PDF == 1) + multivariate_inverse_gaussian(mu, sigma, tau, beta) * (PDF == 2) + multivariate_t(mu, sigma, tau, nu) * (PDF == 3) + multivariate_k(mu, sigma, tau, nu) * (PDF == 4) for j in range(q)] for i in range(K)]
    
    if test_real_shift:
        list_betas = [i for i in range(1,100,10)]
        all_PDF    = [[lambda mu, sigma, tau, beta = list_betas[i] : multivariate_generalized_gaussian(mu, sigma, tau, beta)] for i in range(10)]
    
    return all_PDF

def genere_parametres_simulation(m, n, K, priors, scenario, p_conta, test_real_shift = False):
    
    """ Generates the parameters for the simulation.
    
    Parameters
    ----------
    m        : integer > 0
               dimension of the data
    n        : integer > 0
               number of samples generated      
    K        : integer > 0
               number of clusters
    priors   : 1-d list of float of size K
               list of probability of all clusters
    scenario : str
               scenario used to generate the data
    p_conta  : float >= 0
               probability of drawing a contaminated sample  
    Returns
    -------
    n         : integer > 0
                number of samples generated 
    priors    : 1-d list of float of size K
                list of probability of all clusters
    all_mu    : 2-d array of size K*m
                matrix of the mean vectors of size m of the K clusters
    all_sigma : 3-d array of size K*m*m
                tensor of the shape matrix of the K clusters
    all_tau   : 1-d list of size K
                list of K functions to simulate tau for each cluster
    all_PDF        : list of K lists of potentially different sizes, each sub-list
                     indicates all PDF available to generate a sample for each cluster. 
                     For each generation, a PDF is chosen uniformly randomly among the 
                     ones availables.
    p_conta : float >= 0
                     Probability of drawing a contaminated sample
    conta   : function 
                     Takes as input mean and covariance matrix and returns a contaminated sample      
    """
    
    def conta():
    
        """ Generate a contaminated sample using one of the two fixed-distributions to add noise.
        
        Returns
        -------
        x     : 1-d array of size m
                contaminated sample generated
        """
        if np.random.rand() > 0.5:
            return multivariate_normal(MU_NOISE_0, SIGMA_NOISE_0)
        else:
            return multivariate_normal(MU_NOISE_1, SIGMA_NOISE_1)

    def Tau(a, b):
        
        """ Generates a nuisance parameter as a random real drawn between a and b.
        
        Parameters
        ----------
        a   : float > 0
              lower bound for the random drawing
        b   : float > 0
              upper bound for the random drawing
        Returns
        -------
        tau : float > 0
              nuisance parameter
        """
        return a + np.random.rand() * (b - a)

    list_range_tau = [(1, 1) for k in range(K)]
    all_tau = [lambda a = list_range_tau[i][0], b = list_range_tau[i][1] : Tau(1,10) for i in range(K)]
    all_PDF = genere_all_PDF(scenario, K, test_real_shift)

    return n, priors, genere_all_mu(m, K), genere_all_sigma(m, K), all_tau, all_PDF, p_conta, conta
    
def random_sphere_point(m):

    """ Generate a point uniformly drawn on the unit m-dimensional sphere
    
    Parameters
    ----------
    m : integer > 0
        dimension of the data
    Returns
    -------
    x : 1-d array of size m
        sample generated
    """   
    
    Z = np.random.normal(0, 1, m)
    
    return Z / np.sqrt(sum(Z**2))

def multivariate_generalized_gaussian(mu, sigma, p, beta):
    
    """ Generate a sample drawn from a multivariate generalized gaussian distribution.
    
    Parameters
    ----------
    mu    : 1-d array of size m
            mean of the distribution
    sigma : 2-d array of size m*m
            shape matrix with det = 1
    p     : float > 0
            scale parameter
    beta  : float > 0
            shape parameter
    Returns
    -------
    x     : 1-d array of size m
            sample generated
    """

    return mu + gamma(len(mu) / (2 * beta), 2) ** (1 / (2 * beta)) * np.dot(sqrtm(p * sigma), random_sphere_point(len(mu)))

def multivariate_inverse_gaussian(mu, sigma, p, beta):
    
    """ Generate a sample drawn from a multivariate t distribution.
    
    Parameters
    ----------
    mu    : 1-d array of size m
            mean of the distribution
    sigma : 2-d array of size m*m
            shape matrix with det = 1
    p     : float > 0
            scale parameter
    beta  : float > 0
            shape parameter
    Returns
    -------
    x     : 1-d array of size m
            sample generated
    """
    
    return mu + multivariate_normal(np.zeros(len(mu)), p * sigma) * np.sqrt(wald(1, beta))

def multivariate_t(mu, sigma, p, nu):
    
    """ Generate a sample drawn from a multivariate t distribution.
    
    Parameters
    ----------
    mu    : 1-d array of size m
            mean of the distribution
    sigma : 2-d array of size m*m
            shape matrix with det = 1
    p     : float > 0
            scale parameter
    nu    : integer > 0
            Degree of freedom of the distribution
    Returns
    -------
    x     : 1-d array of size m
            sample generated
    """
    
    return mu + multivariate_normal(np.zeros(len(mu)), p * sigma) * np.sqrt(1/gamma(nu/2, 2/nu))

def multivariate_k(mu, sigma, p, nu):
    
    """ Generate a sample drawn from a multivariate t distribution.
    
    Parameters
    ----------
    mu    : 1-d array of size m
            mean of the distribution
    sigma : 2-d array of size m*m
            shape matrix with det = 1
    p     : float > 0
            scale parameter
    nu    : integer > 0
            Degree of freedom of the distribution
    Returns
    -------
    x     : 1-d array of size m
            sample generated
    """
  
    return mu + multivariate_normal(np.zeros(len(mu)), p * sigma) * np.sqrt(gamma(nu, 1/nu))

class dataSimulation():
    
    """ Implements an object to simulate data.   
    
    Parameters
    ----------
    n         : integer > 0
                number of samples generated
    all_pi    : 1-d array of size K
                vector of probability of all clusters
    all_mu    : 2-d array of size K*m
                matrix of the mean vectors of size m of the K clusters
    all_sigma : 3-d array of size K*m*m
                tensor of the shape matrix of the K clusters
    all_tau   : 1-d list of size K
                list of K functions to simulate tau for each cluster
    all_PDF   : list of K lists of potentially different sizes, each sub-list
                indicates all PDF available to generate a sample for each cluster. 
                For each generation, a PDF is chosen uniformly randomly among the 
                ones availables.
    p_conta   : float >= 0
                probability of drawing a contaminated sample
    conta     : function 
                takes as input mean and covariance matrix and returns a contaminated sample
    Attributes
    ----------
    m         : integer > 0  
                dimension of each sample
    K         : integer > 0
                number of clusters
    n         : integer > 0
                number of samples generated
    all_pi    : 1-d array of size K
                vector of probability of all clusters
    all_mu    : 2-d array of size K*m
                matrix of the mean vectors of size m of the K clusters
    all_sigma : 3-d array of size K*m*m
                tensor of the shape matrix of the K clusters
    all_tau   : 1-d list of size K
                list of K functions to simulate tau for each cluster
    all_PDF   : list of K lists of potentially different sizes, each sub-list
                indicates all PDF available to generate a sample for each cluster. 
                For each generation, a PDF is chosen uniformly randomly among the 
                ones availables.
    p_conta   : float >= 0
                probability of drawing a contaminated sample
    conta     : function 
                takes as input mean and covariance matrix and returns a contaminated sample
    X         : 2-d array of size n*m
                matrix of all the samples generated
    labels    : 1-d array of size n
                vector of the label of each sample
    PDFs      : 1-d array of size n
                vector of the index of the distribution chosen to draw the samples 
    Methods
    ----------
    generateSample  : Generates a random sample for cluster k
    generateSamples : Generates a random dataset of size n
    """
    
    
    def __init__(self, n, all_pi, all_mu, all_sigma, all_tau, all_PDF, p_conta, conta):
        
        self.m         = all_sigma.shape[1]
        self.K         = all_sigma.shape[0]
        self.n         = n
        self.all_pi    = all_pi
        self.all_mu    = all_mu
        self.all_sigma = all_sigma
        self.all_tau   = all_tau
        self.all_PDF   = all_PDF
        self.p_conta   = p_conta
        self.conta     = conta
        self.X         = np.zeros([n, all_sigma.shape[1]])
        self.labels    = np.zeros(n)
        self.PDFs      = np.zeros(n)
        
    def generateSample(self, k):

        mu    = self.all_mu[k]
        sigma = self.all_sigma[k]
        tau   = self.all_tau[k]()
        j     = np.random.randint(0, len(self.all_PDF[k]))
        PDF   = self.all_PDF[k][j]
        if np.random.rand() < self.p_conta:
            return self.conta(mu, tau*sigma), -1
        else:
            return PDF(mu, sigma, tau), j
    
    def generateSamples(self):
        
        for i in range(self.n):
            RND = np.random.rand()
            k = 0
            while RND > self.all_pi[k]:
                
                RND = RND - self.all_pi[k]
                k = k + 1
            self.X[i], self.PDFs[i] = self.generateSample(k)    
            self.labels[i] = k
        return self.X, self.labels, self.PDFs