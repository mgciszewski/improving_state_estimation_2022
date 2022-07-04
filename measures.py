from itertools import zip_longest
import numpy as np

def lts_measure(true_jumps, true_states, est_jumps, est_states, l = 60, w = 0.6, sigma = 0.35, lam = 0.01, zeta = 0.5):
    """
    Calculates the LTS measure of the estimated labels.

    Parameters
    ----------
    correct_df : pandas Series
                 Series containing true labels
    estimated_df : pandas Series
                   Series containing estimated labels
    w : double
        controls the weight of misclassification occurring from the uncertainty of the true labels
    sigma : double
            controls the magnitude of the shift of activities
    lam : double
          the penalty for each violation of the lower bound condition.
    zeta : double
           the lower bound on the lengths of the events as determined by the domain knowledge

    Returns
    -------
    The LTS measure of the estimated labels for given set of parameters.
    """
    error = 0
    all_jumps = list(zip_longest([1] * (len(true_jumps) + 1), true_jumps + [l], true_states, true_states[1:])) + list(zip_longest([2] * (len(est_jumps) + 1), (est_jumps + [l]), est_states, est_states[1:]))
    all_jumps = sorted(all_jumps, key = lambda tup: tup[1])
    prev_jump, prev_true_state, prev_est_state = all_jumps[0], 1, 1
    
    for jump in all_jumps[1:]:
        if jump[0] == 1:
            prev_true_state = jump[2]
            if prev_jump[0] == 2:
                prev_est_state = prev_jump[3]
        else:
            prev_est_state = jump[2]
            if prev_jump[0] == 1:
                prev_true_state = prev_jump[3]
            
        if prev_true_state != prev_est_state:
            if (jump[1] - prev_jump[1] <= sigma) & (jump[0] != prev_jump[0]):
                error += w * (jump[1] - prev_jump[1])
            else:
                error += jump[1] - prev_jump[1]
        prev_jump = jump
    
    penalty_term = 0
    prev_jump = est_jumps[0]
    for curr_jump in est_jumps[1:]:
        if (curr_jump - prev_jump) < zeta:
            penalty_term += lam
        prev_jump = curr_jump
                
    return np.exp(-error / 60 - penalty_term)