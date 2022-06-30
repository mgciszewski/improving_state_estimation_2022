from misc_func import flatten_list, identify_jump_subseq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def find_projection(f_vals, gamma, fs):
    """
    Find a projection of a piecewise constant function on the space of functions with lower bound on the minimum duration of activities.

    Parameters
    ----------
    f_vals : pandas Series
             a Series representing the state sequence f
    gamma : double
            the penalty for the jump in a projection
    fs : integer
         sampling frequency
            
    Returns
    -------
    f_hat_vals : list of states of a projection of f_vals
    """
    all_est_states = []
    all_est_vertices = []

    jump_list = f_vals.index[abs(f_vals.diff()) > 0].tolist()
    
    jump_sets = identify_jump_subseq(jump_list, 2 * gamma * fs)

    for jump_set in jump_sets:
        states = [f_vals.loc[jump_set[k]] for k in range(len(jump_set))]
        states.insert(0, f_vals.loc[jump_set[0] - 1])

        est_vertices, est_states = find_projection_single_jump_set(jump_set, states, gamma, fs)

        all_est_vertices.append(est_vertices[1:-1])
        all_est_states.append(est_states[:-1])

    all_est_states.append(est_states[-1:])

    all_est_vertices = [-np.inf] + flatten_list(all_est_vertices) + [np.inf]
    all_est_states = flatten_list(all_est_states)

    f_hat_vals = [all_est_states[0]] * len(f_vals)
    
    for state_idx in range(len(all_est_states) - 1):
        f_hat_vals[all_est_vertices[state_idx + 1]:len(f_hat_vals)] = [all_est_states[state_idx + 1]] * (len(f_vals) - all_est_vertices[state_idx + 1])
        
    return f_hat_vals

def find_projection_single_jump_set(jump_set, states, gamma, fs):
    """
    Find a projection of a piecewise constant function on the space of functions with lower bound on the minimum duration of activities.
    This function is specifically designed for a case when the consecutive jumps of the function are distant by at most 2 * gamma from each other 

    Parameters
    ----------
    jump_set : list of integers
               a list of jumps of the function
    states : list of integers
             a list of states of the function; states[0] is a state of the function before first jump, states[1] is a state of the function between the first two jumps and so on
    gamma : double
            the penalty for the jump in a projection
    fs : integer
         sampling frequency
            
    Returns
    -------
    est_vertices : a list of jumps of the function (includes -np.inf and np.inf)
    est_states : a list of states of the function
    """
    vertices = jump_set

    vertices.insert(0, -np.inf)
    vertices.append(np.inf)

    distance_matrix = []
    estimate_matrix = []

    for i in range(len(vertices) - 1):
        vertex_i = vertices[i]
        curr_vertex = vertex_i
        state_i = states[i]
        curr_state = state_i
        state_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        row = []
        estimate_row = []
        for j in range(len(vertices)):
            if j > i:
                vertex_j = vertices[j]
                state_dict[curr_state] += vertex_j - curr_vertex
                curr_vertex = vertex_j
                est_state = max(state_dict, key=state_dict.get)
                estimate_row.append(est_state)
                est_state_dur = max(state_dict.values())
                if vertex_j == np.inf:
                    temp_val = sum([state_dict[key] for key in state_dict.keys() if key != est_state])
                    if temp_val == np.inf:
                        row.append(0)
                    elif temp_val == 0:
                        row.append(0.0000000001)
                    else:
                        row.append(temp_val)
                elif vertex_j-vertex_i < gamma * fs:
                    curr_state = states[j]
                    row.append(0)
                else:
                    curr_state = states[j]
                    row.append(gamma * fs + sum([state_dict[key] for key in state_dict.keys() if key != est_state]))
            else:
                row.append(0)
                estimate_row.append(0)
        distance_matrix.append(row)
        estimate_matrix.append(estimate_row)

    distance_matrix.append([0 for i in range(len(vertices))])
    estimate_matrix.append([0 for i in range(len(vertices))])

    graph = csr_matrix(distance_matrix)
    dist_matrix, predecessors = shortest_path(csgraph=graph, 
                                              directed=True, 
                                              indices=0, 
                                              return_predecessors=True)

    vv = np.inf
    path = []
    path.append(vv)
    vv = predecessors[-1]
    while vv > 0:
        path.append(vv)
        vv = predecessors[vv]

    path.append(vv)
    path = path[::-1]

    est_vertices = [vertices[path[j]] for j in range(len(path) - 1)]
    est_vertices.append(vertices[len(vertices) - 1])
    
    est_states = [estimate_matrix[path[j]][path[j + 1]] for j in range(len(path) - 2)]
    est_states.append(estimate_matrix[path[len(path) - 2]][len(vertices) - 1])
    
    return est_vertices, est_states