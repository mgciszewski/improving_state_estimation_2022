import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import choices

def detect_jumps(l):
    """
    Detects all locations of the jumps in a list of numbers.

    Parameters
    ----------
    l : array of numbers
        an array of numbers to detect the locations of the jumps
        
    Returns
    -------
    an array consisting of all locations of the jumps in l
    """
    return [0] + [int(i) for i in range(1,len(l)) if l[i] != l[i-1]] + [len(l) - 1]
    
def flatten_list(l):
    """
    Flattens array of arrays; creates a single array with elements of all arrays in l.

    Parameters
    ----------
    l : array of arrays
        an array of arrays to flatten
        
    Returns
    -------
    an array consisting of elements from all arrays in l
    """
    return [item for sublist in l for item in sublist]

def generate_state_seq(mu_1, mu_2):
    """
    Generate a state sequence of noisy labels as prescribed in section 4.1 of the manuscript.

    Parameters
    ----------
    mu_1 : double
           the parameter of the exponential distribution (time spent in the correct state)
    mu_2 : double
           the parameter of the exponential distribution (time spent in the incorrect state)
        
    Returns
    -------
    is_correct : a list of boolean values referring to whether the noisy labels are correct at each timepoint
    wrong_states: a list of incorrect states (the length of wrong_states is equal to the number of False values in is_correct)
    """
    total_time = 0
    is_correct = []
    wrong_states = []
    
    jumps = []
    states = []
    accuracy = 0
    
    true_dict = {5.0: 1,
                 15.0: 2,
                 30.0: 3,
                 40.0: 2,
                 55.0: 3,
                 60.0: 1}
    
    while total_time < 60:
        waiting_time = np.random.exponential(mu_1)
        if total_time + waiting_time > 60:
            waiting_time = 60 - total_time
            states.append(1)
        else:
            prev_key = 0
            for key in true_dict:
                if (total_time < key) & (total_time + waiting_time >= key):
                    jumps.append(key)
                    states.append(true_dict[key])
                if (total_time + waiting_time < key) & (total_time + waiting_time >= prev_key):
                    states.append(true_dict[key])
                prev_key = key
            jumps.append(total_time + waiting_time)
        accuracy += waiting_time
        total_time += waiting_time

        if total_time < 60:
            duration_time = np.random.exponential(mu_2)
            if total_time + duration_time > 59.999:
                duration_time = 59.999 - total_time
            if total_time < 5: # in state 1
                incorrect = choices([2, 3])[0]
            elif total_time < 15: # in state 2
                incorrect = choices([1, 3])[0]
            elif total_time < 30: # in state 3
                incorrect = choices([1, 2])[0]
            elif total_time < 40: # in state 2
                incorrect = choices([1, 3])[0]
            elif total_time < 55: # in state 3
                incorrect = choices([1, 2])[0]
            else: # in state 1
                incorrect = choices([2, 3])[0]
            total_time += duration_time
        
            jumps.append(total_time)
            states.append(incorrect)
                
    return jumps, states, accuracy / 60

def identify_jump_subseq(jumps, gam):
    """
    Returns array of arrays of jumps. 
    Jumps are put into one block (array) if the distance between subsequent jumps in the block is smaller than gam.

    Parameters
    ----------
    jumps : array of doubles
        an array of jumps
    gam : double
        lower bound on the lengths of the intervals (also double the weight of the jump)

    Returns
    -------
    res: an array of arrays of doubles
        a list of all blocks of jumps such that each block contains jumps that have at least one gamma neighbour in it or the block is a singleton
    """
    # res is initialized to a list containing one empty array; last refers to the last jump considered; at this moment it's None
    res, last = [[]], None
    
    # we iterate over all jumps
    for x in jumps:
        # if this is the first jump (last is None) or the previous jump was in gamma vicinity of 
        # current jump x then x is added to the most recent block in res
        if last is None or abs(last - x) < gam:
            res[-1].append(x)
        # otherwise a new block is created
        else:
            res.append([x])
        # we update last
        last = x
    
    return res

def mystep(x,y, ax=None, **kwargs):
    """ 
    Plots discontinuous functions without vertical lines at jumps

    Parameters
    ----------
    x: list
        list of arguments
    y: list
        list of values
    ax: axis object
        Axis object on which the plot should appear

    Returns
    -------
    the plot of y against x
    """
    # convert x and y into numpy array
    x = np.array(x)
    y = np.array(y)
    
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y[:-1],y[:-1],np.zeros_like(x[:-1])*np.nan]
    
    # if ax parameter is not given, use current figure
    if not ax: ax=plt.gca()
        
    return ax.plot(X.flatten(), Y.flatten(), **kwargs)

def set_size(width = 345.0, fraction=1, subplots=(1, 1)):
    """ 
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    # Dimensions of the figure as a tuple (in inches)
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def split_function(jumps, states, gam):
    """
    Returns array of arrays of jumps. 
    Jumps are put into one block (array) if the distance between subsequent jumps in the block is smaller than gam.

    Parameters
    ----------
    jumps : array of doubles
            an array of jumps
    states : array of doubles
            an array of states
    gam : double
        lower bound on the lengths of the intervals (also double the weight of the jump)

    Returns
    -------
    res: an array of arrays of doubles
         a list of all blocks of jumps such that each block contains jumps that have at least one gamma neighbour in it or the block is a singleton
    res_states : an array of arrays of integers
                 a list of all blocks of states divided in the same manner as res
    """
    res, res_states, last = [[]], [[states[0]]], None
    
    for idx in range(len(jumps)):
        jump = jumps[idx]
        if last is None or abs(last - jump) < gam:
            res[-1].append(jump)
            res_states[-1].append(states[idx + 1])
        else:
            res.append([jump])
            res_states.append([states[idx], states[idx + 1]])
        last = jump

    return res, res_states