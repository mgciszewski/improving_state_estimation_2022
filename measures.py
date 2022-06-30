import numpy as np
from misc_func import detect_jumps

def lts_measure(correct_df, estimated_df, fs = 1000, w = 0.6, sigma = 0.35, lam = 0.01, zeta = 0.5):
    """
    Calculates the LTS measure of the estimated labels.

    Parameters
    ----------
    correct_df : pandas Series
                 Series containing true labels
    estimated_df : pandas Series
                   Series containing estimated labels
    fs : int
         sampling frequency
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
    true_jumps = detect_jumps(correct_df)
    pp_jumps = detect_jumps(estimated_df)

    error = 0
    all_jumps = np.unique(true_jumps + pp_jumps) # segments
    prev_jump = all_jumps[0]
    for curr_jump in all_jumps[1:]:
        prev_true_state = correct_df[prev_jump]
        prev_pp_state = estimated_df[prev_jump]
        
        if prev_true_state != prev_pp_state:
            if (abs(curr_jump - prev_jump) / fs <= sigma) & (curr_jump < len(correct_df) - 1):
                if (correct_df[prev_jump - 1] == estimated_df[prev_jump - 1]) & (correct_df[curr_jump + 1] == estimated_df[curr_jump + 1]):
                    error += w * (curr_jump - prev_jump) / fs
                else:
                    error += (curr_jump - prev_jump) / fs
            else:
                error += (curr_jump - prev_jump) / fs
        
        prev_jump = curr_jump
    
    penalty_term = 0
    prev_jump = pp_jumps[0] / fs
    for curr_jump in pp_jumps[1:]:
        if (curr_jump - prev_jump) / fs < zeta:
            penalty_term += lam
        prev_jump = curr_jump / fs
                
    max_seconds = len(correct_df) / fs
    return np.exp(-error / max_seconds - penalty_term)