import numpy as np 

from math import log
from pot_utils.utils.grimshaw import grimshaw


def pot(data:np.array, risk:float=1e-4, init_level:float=0.98, num_candidates:int=10, epsilon:float=1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

    References: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold
    t = np.sort(data)[int(init_level * data.size)]
    print("t: ",t)
    return t,0
    peaks = data[data > t] - t 
    if len(peaks) == 0:
        return t,t
    if peaks.min()==peaks.max():
        return t,t
    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks, 
                            threshold=t, 
                            num_candidates=num_candidates, 
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        print("gamma: ",(sigma / gamma) * (pow(r, -gamma) - 1))
    else: 
        z = t - sigma * log(r)
        print("else: ",sigma * log(r))

    return z, t
    

def pot_min(data: np.array, risk: float = 1e-4, init_level: float = 0.02, num_candidates: int = 10, epsilon: float = 1e-8) -> float:
    ''' Peak-over-Threshold Algorithm for minima

    References: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold for minima
    t = np.sort(data)[int(init_level * data.size)]
    peaks = t - data[data < t]

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks, 
                            threshold=t, 
                            num_candidates=num_candidates, 
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t - (sigma / gamma) * (pow(r, -gamma) - 1)
    else: 
        z = t + sigma * log(r)

    return z, t
