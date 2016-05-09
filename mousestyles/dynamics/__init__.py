from __future__ import print_function, absolute_import, division

import pandas as pd
import numpy as np
from math import ceil
from mousestyles import data


def create_time_matrix(combined_gap=4, time_gap=1, days_index=137):
    r"""
    Return a time matrix for estimate the MLE parobability.
    The rows are 137 mousedays. The columns are time series
    in a day. The data are the mouse activity at that time.
    0 represents IS, 1 represents eating, 2 represents
    drinking, 3 represents others activity in AS.

    Parameters
    ----------
    combined_gap: nonnegative float or int
        The threshold for combining small intervals. If next start time
        minus last stop time is smaller than combined_gap than combined
        these two intervals.
    time_gap: positive float or int
        The time gap for create the columns time series
    days_index: nonnegative int
        The number of days to process, from day 0 to day days_index.

    Returns
    -------
    time: Pandas.DataFrame
        a matrix represents the activity for a certain
        mouse day and a certain time.

    Examples
    --------
    >>> time = create_time_matrix(combined_gap=4, time_gap=1).iloc[0, 0:10]
    >>> strain    0
        mouse     0
        day       0
        48007     0
        48008     0
        48009     0
        48010     0
        48011     0
        48012     0
        48013     0
        Name: 0, dtype: float64
    """
    # check all the inputs
    condition_combined_gap = ((type(combined_gap) == int or
                              type(combined_gap) == float) and
                              combined_gap >= 0)
    condition_time_gap = ((type(time_gap) == int or type(time_gap) ==
                           float) and time_gap > 0)
    condition_days_index = (type(days_index) == int and days_index >= 0)
    if not condition_time_gap:
        raise ValueError("time_gap should be nonnegative int or float")
    if not condition_combined_gap:
        raise ValueError("combined_gap should be nonnegative int or float")
    if not condition_days_index:
        raise ValueError("days_index should be nonnegative int")

    intervals_AS = data.load_intervals('AS')
    intervals_F = data.load_intervals('F')
    intervals_W = data.load_intervals('W')
    intervals_IS = data.load_intervals('IS')
    # 137 days totally
    days = np.array(intervals_AS.iloc[:, 0:3].drop_duplicates().
                    reset_index(drop=True))
    # set time range for columns
    initial = int(min(intervals_IS['stop']))
    end = int(max(intervals_IS['stop'])) + 1
    columns = np.arange(initial, end + 1, time_gap)
    # result matrix
    matrix = np.zeros((days.shape[0], len(columns)))
    # we set 0 as IS, 1 as F, 2 as W, 3 as Others
    for i in range(days.shape[0]):
        W = np.array(intervals_W[(intervals_W['strain'] == days[i, 0]) &
                                 (intervals_W['mouse'] == days[i, 1]) &
                                 (intervals_W['day'] == days[i, 2])].
                     iloc[:, 3:5])
        F = np.array(intervals_F[(intervals_F['strain'] == days[i, 0]) &
                                 (intervals_F['mouse'] == days[i, 1]) &
                                 (intervals_F['day'] == days[i, 2])].
                     iloc[:, 3:5])
        AS = np.array(intervals_AS[(intervals_AS['strain'] == days[i, 0]) &
                                   (intervals_AS['mouse'] == days[i, 1]) &
                                   (intervals_AS['day'] == days[i, 2])].
                      iloc[:, 3:5])
        n = W.shape[0]
        index = (np.array(np.where(W[1:, 0]-W[0:n - 1, 1] >
                                   combined_gap))).ravel()
        stop_W = W[np.append(index, n - 1), 1]
        start_W = W[np.append(0, index + 1), 0]
        n = F.shape[0]
        index = (np.array(np.where(F[1:, 0]-F[0:n-1, 1] >
                                   combined_gap))).ravel()
        stop_F = F[np.append(index, n - 1), 1]
        start_F = F[np.append(0, index + 1), 0]
        n = AS.shape[0]
        index = (np.array(np.where(AS[1:, 0]-AS[0:n - 1, 1] >
                                   combined_gap))).ravel()
        stop_AS = AS[np.append(index, n - 1), 1]
        start_AS = AS[np.append(0, index + 1), 0]
        for j in range(len(columns)):
            if sum(np.logical_and(columns[j] > start_AS, columns[j] <
                                  stop_AS)) != 0:
                if sum(np.logical_and(columns[j] > start_F, columns[j] <
                                      stop_F)) != 0:
                    matrix[i, j] = 1  # food
                elif sum(np.logical_and(columns[j] > start_W, columns[j] <
                                        stop_W)) != 0:
                    matrix[i, j] = 2  # water
                else:
                    matrix[i, j] = 3  # others
        # give you the precent of matrix has been processed
        print(i / days.shape[0], 'has been processed')
        if i > days_index:
            break
    # format data frame
    matrix = pd.DataFrame(matrix, columns=columns)
    title = pd.DataFrame(days, columns=['strain', 'mouse', 'day'])
    time_matrix = pd.concat([title, matrix], axis=1)
    return(time_matrix)


def get_prob_matrix_list(time_df, interval_length=1000):
    r"""
    returns a list of probability transition matrices
    that will be later used to characterize and simu-
    late the behavior dynamics of different strains of
    mice. The data used as input is the pandas DataFrame
    generated by function create_time_matrix with de-
    fault parameters. The output is a list of numpy
    arrays, each being a transition matrix characterizing
    one small time interval. The interval length could
    be chosen.

    Parameters
    ----------
    time_df: Pandas.DataFrame
        a huge data frame containing info on strain, mouse
        no., mouse day, and different states at chosen time
        points.
    interval_length: int
        an integer specifying the desired length of each
        small time interval.

    Returns
    -------
    matrix_list: list
        a list of the mle estimations of the probability tran-
        sition matrices for each small time interval stored in
        the format of numpy array. Each element of this list
        is a numpy array matrix.

    Examples
    --------
    >>> row_i = np.hstack((np.zeros(13), np.ones(10),
                            np.ones(10)*2, np.ones(10)*3))
    >>> time_df_eg = np.vstack((row_i, row_i,row_i))
    >>> time_df_eg = pd.DataFrame(time_df_eg)
    >>> mat_list = get_prob_matrix_list(time_df_eg,
                                        interval_length=10)
    >>> mat_list[0]
    >>> array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
    >>> mat_list[1]
    >>> array([[ 0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
    >>> mat_list[2]
    >>> array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  0.]])
    >>> mat_list[3]
    >>> array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

    """
    # check all the inputs
    condition_time_df = (type(time_df) == pd.core.frame.DataFrame)
    condition_interval_length = (type(interval_length) == int and
                                 interval_length > 0)
    if not condition_time_df:
        raise ValueError("time_df should be pandas DataFrame")
    if not condition_interval_length:
        raise ValueError("interval_length should be positive int")

    time_array = np.array(time_df)[:, 3:]
    n = ceil(time_array.shape[1]/interval_length)
    matrix_list = [None] * int(n)
    for i in np.arange(n):
        i = int(i)
        ind = [(i * interval_length), ((i+1) * interval_length)]
        small_time_array = time_array[:, ind[0]:ind[1]]
        small_time_list = list(small_time_array)
        small_time_str_list = ["".join(np.char.mod('%i', a))
                               for a in small_time_list]
        matrix_list[i] = get_prob_matrix_small_interval(small_time_str_list)
    return matrix_list


def get_prob_matrix_small_interval(string_list):
    r"""
    return the MLE estimate of the probability matrix
    of the markov chain model. The data used as input
    is a list of strings that contains the information
    regarding the transition of states of the mouse be-
    havior. The output is a matrix stored in the format
    of numpy array, where the i,j th term indicates the
    probability of transiting from state i to state j.

    Parameters
    ----------
    string_list: list
        a list of strings of the states in the given
        time slot.

    Returns
    -------
    M: numpy.ndarray
        the MLE estimation of the probability tran-
        sition matrix. Each entry M_ij represents the
        probability of transiting from state i to state
        j.

    Examples
    --------
    >>> time_list = ['002', '001', '012']
    >>> get_prob_matrix_small_interval(time_list)
    >>> array([[ 0.4,  0.4,  0.2,  0. ],
               [ 0. ,  0. ,  1. ,  0. ],
               [ 0. ,  0. ,  0. ,  0. ],
               [ 0. ,  0. ,  0. ,  0. ]])

    """
    # check all the inputs
    condition_string_list = (type(string_list) == list)
    condition_list_item = (type(string_list[0]) == str)
    print(string_list[0])
    if not condition_string_list:
        raise ValueError("string_list should be a list")
    if not condition_list_item:
        raise ValueError("items in string_list should be str")

    Mat_prob = np.zeros(4*4).reshape(4, 4)
    for i in np.arange(4):
        i = int(i)
        for j in np.arange(4):
            j = int(j)
            ijth = str(i) + str(j)
            Mat_prob[i, j] = sum([string.count(ijth) for string
                                 in string_list])
    for k in np.arange(4):
        k = int(k)
        if sum(Mat_prob[k, :]) != 0:
            Mat_prob[k, :] = Mat_prob[k, :]/sum(Mat_prob[k, :])
    return Mat_prob


def mcmc_simulation(mat_list, n_per_int):
    r"""
    This function gives the Monte Carlo simulation
    of the stochastic process modeling the dynamic
    changes of states of the mice behavior. The in-
    put of this function is a list of probability
    transition matrices and an integer indicates
    how many outputs for each matrix. This number
    is related to the interval_length parameter in
    function get_prob_matrix_list. The output is an
    array of numbers, each indicates one state.

    Parameters
    ----------
    mat_list: List
        a list of numpy arrays storing the probabi-
        lity transition matrices for each small time
        interval chosen.
    n_per_int: int
        an integer specifying the desired output
        length of each probability transition matrix.
        This is the same as the parameter
        interval_length used in the function
        get_prob_matrix_small_interval

    Returns
    -------
    simu_result: numpy.array
        an array of integers indicating the simulated
        states given a list of probability transition
        matrices.

    Examples
    --------
    >>> mat0 = np.zeros(16).reshape(4, 4)
    >>> np.fill_diagonal(mat0, val=1)
    >>> mat1 = np.zeros(16).reshape(4, 4)
    >>> mat1[0, 1] = 1
    >>> mat1[1, 0] = 1
    >>> mat1[2, 2] = 1
    >>> mat1[3, 3] = 1
    >>> mat_list_example = [mat0, mat1]
    >>> mcmc_simulation(mat_list_example, 10)
    >>> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    """
    # check all the inputs
    condition_mat_list = (type(mat_list) == list)
    condition_list_item = (type(mat_list[0]) == np.ndarray)
    condition_n_per_int = (type(n_per_int) == int and n_per_int > 0)
    if not condition_mat_list:
        raise ValueError("mat_list should be a list")
    if not condition_list_item:
        raise ValueError("items in mat_list should be numpy array")
    if not condition_n_per_int:
        raise ValueError("n_per_int should be positive int")

    n = len(mat_list)
    simu_result = np.zeros(n * n_per_int)
    for i in np.arange(n):
        for j in np.arange(n_per_int):
            index = int(i * n_per_int + j)
            if index == 0:
                simu_result[index] = 0
            state_i = simu_result[int(index-1)]
            prob_trans = mat_list[i][int(state_i), :]
            prob_trans = np.cumsum(prob_trans)
            rand = np.random.uniform()
            state_j = sum(rand > prob_trans)
            simu_result[index] = int(state_j)
    simu_result = np.array(simu_result, dtype=int)
    return simu_result


def get_score(true_day, simulated_day, weight=[1, 10, 50, 1]):
    r"""
    Returns the evaluation score for the simulted day
    that will be later used to choose the best time
    interval for different strains. The input data
    should be two numpy arrays and one list, the two arrays
    are possibly with different lengths, one being the
    activities for one particular day of one particular
    mouse, the other array is our simulated day for this
    mouse from the mcmc_simulation function. And the list
    is the weight for different status. We should give different
    rewards for making correct simulations on various status.
    The output will be one number between 0 and max(weight),
    indicating the similiary of the true day of a mouse and
    a simulated day of the same mouse. We will use
    this function to measure the performance of the
    simulation and then choose the appropriate time
    interval.

    Parameters
    ----------
    true_day: numpy.array
        a numpy.array containing the activities for one
        particular mouse on a specific day
    simulated_day: numpy.array
        a numpy.array containing the simulated activities
        for this particular mouse.
    weight: list
        a list with positive numbers showing the rewards
        for making the right predictions of various status.

    Returns
    -------
    score: float
        a float from 0 to max(weight), indicating the similarity of
        the simulated data with the actual value, and therefore,
        the performance of the simulation, with max(weight) being the
        most similar, and 0 being the least similar.

    Examples
    --------
    >>> true_day_1 = np.zeros(13)
    >>> simulated_day_1 = np.ones(13)
    >>> score_1 = get_score(true_day_1, simulated_day_1)
    >>> 0.0
    >>> true_day_2 = np.ones(13)
    >>> simulated_day_2 = np.ones(13)
    >>> score_2 = get_score(true_day_2, simulated_day_2)
    >>> 10.0

    """
    # check all the inputs
    condition_true_day = (isinstance(true_day, (np.ndarray, np.generic)))
    condition_simulated_day = (isinstance(simulated_day,
                               (np.ndarray, np.generic)))
    condition_weight = (isinstance(weight, list))

    if not condition_true_day:
        raise ValueError("true_day should be numpy array!")
    if not condition_simulated_day:
        raise ValueError("simulated_day should be numpy array!")
    if not condition_weight:
        raise ValueError("weight should be list!")

    len_weight = len(weight)
    if len_weight != 4:
        raise ValueError("Length of weight should be 4!")

    # check all the weights are positive
    for w in weight:
        if w <= 0:
            raise ValueError("All the weights should be positive!")

    # make sure these two arrays have the same length
    len_true = len(true_day)
    len_simulated = len(simulated_day)
    if len_true > len_simulated:
        raise ValueError("Length of simulated_day is smaller than true_day!")
    simulated_same_length = simulated_day[:len_true]

    score = 0
    for i in np.arange(len_true):
        if true_day[i] == simulated_same_length[i]:
            status = true_day[i]
            score += weight[int(status)]
    score = score/len_true
    return score
