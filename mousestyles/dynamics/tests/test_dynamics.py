from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np
import pandas as pd

from mousestyles.dynamics import create_time_matrix
from mousestyles.dynamics import get_prob_matrix_list
from mousestyles.dynamics import get_prob_matrix_small_interval
from mousestyles.dynamics import mcmc_simulation, get_score


def test_creat_time_matrix_input():
    # checking functions raise the correct errors for wrong input
    # time_gap is zeros
    err_string = "time_gap should be nonnegative int or float"
    with pytest.raises(ValueError) as excinfo:
        create_time_matrix(combined_gap=4, time_gap=0, days_index=137)
    assert excinfo.value.args[0] == err_string
    # time_gap is negative
    with pytest.raises(ValueError) as excinfo:
        create_time_matrix(combined_gap=4, time_gap=-1, days_index=137)
    assert excinfo.value.args[0] == err_string
    # combined_ap is negative value
    err_string = "combined_gap should be nonnegative int or float"
    with pytest.raises(ValueError) as excinfo:
        create_time_matrix(combined_gap=-1, time_gap=1, days_index=137)
    assert excinfo.value.args[0] == err_string
    # min_path_length cannot be floating number
    # days_index is negative value
    with pytest.raises(ValueError) as excinfo:
        create_time_matrix(combined_gap=4, time_gap=1, days_index=-1)
    assert excinfo.value.args[0] == "days_index should be nonnegative int"
    # days_index is float value
    with pytest.raises(ValueError) as excinfo:
        create_time_matrix(combined_gap=4, time_gap=1, days_index=0.1)
    assert excinfo.value.args[0] == "days_index should be nonnegative int"


def test_creat_time_matrix():
    # Checking functions output the correct time matrix
    matrix = create_time_matrix(combined_gap=4, time_gap=1, days_index=0)
    assert matrix.iloc[0, 2181] == 1.0


def test_get_prob_matrix_list_input():
    # checking functions raise the correct errors for wrong input
    # time_df is not DataFrame
    with pytest.raises(ValueError) as excinfo:
        get_prob_matrix_list(time_df=5, interval_length=1000)
    assert excinfo.value.args[0] == "time_df should be pandas DataFrame"
    # interval_length is 0
    row_i = np.hstack((np.zeros(13), np.ones(10),
                      np.ones(10)*2, np.ones(10)*3))
    time_df_eg = np.vstack((row_i, row_i, row_i))
    time_df_eg = pd.DataFrame(time_df_eg)
    with pytest.raises(ValueError) as excinfo:
        get_prob_matrix_list(time_df=time_df_eg, interval_length=0)
    assert excinfo.value.args[0] == "interval_length should be positive int"
    # interval_length is not int
    with pytest.raises(ValueError) as excinfo:
        get_prob_matrix_list(time_df=time_df_eg, interval_length=0.5)
    assert excinfo.value.args[0] == "interval_length should be positive int"


def test_get_prob_matrix_list():
    # Checking functions output the correct matrix list
    row_i = np.hstack((np.zeros(13), np.ones(10),
                      np.ones(10)*2, np.ones(10)*3))
    time_df_eg = np.vstack((row_i, row_i, row_i))
    time_df_eg = pd.DataFrame(time_df_eg)
    mat_list = get_prob_matrix_list(time_df_eg,
                                    interval_length=10)
    assert mat_list[0][0, 0] == 1.
    assert sum(sum(mat_list[0])) == 1.


def test_get_prob_matrix_small_interval_input():
    # checking functions raise the correct errors for wrong input
    # string_list is not list
    with pytest.raises(ValueError) as excinfo:
        get_prob_matrix_small_interval(string_list=np.array([1, 2]))
    assert excinfo.value.args[0] == "string_list should be a list"
    # items in string_list is not string
    time_list = [0, 1, 2]
    with pytest.raises(ValueError) as excinfo:
        get_prob_matrix_small_interval(string_list=time_list)
    assert excinfo.value.args[0] == "items in string_list should be str"


def test_get_prob_matrix_small_interval():
    # Checking functions output the correct matrix
    time_list = [str('002'), '001', '012']
    example = get_prob_matrix_small_interval(time_list)
    assert example[0, 0] == 0.4
    assert example[0, 1] == 0.4
    assert example[0, 2] == 0.2
    assert example[1, 2] == 1.
    assert sum(example[0, :]) == 1.


def test_mcmc_simulation_input():
    # checking functions raise the correct errors for wrong input
    # mat_list is not list
    with pytest.raises(ValueError) as excinfo:
        mcmc_simulation(mat_list=np.array([1, 2]), n_per_int=10)
    assert excinfo.value.args[0] == "mat_list should be a list"
    # items in mat_list is not string
    time_list = [0, 1, 2]
    with pytest.raises(ValueError) as excinfo:
        mcmc_simulation(mat_list=time_list, n_per_int=10)
    assert excinfo.value.args[0] == "items in mat_list should be numpy array"
    # n_per_int is not integer
    mat0 = np.zeros(16).reshape(4, 4)
    np.fill_diagonal(mat0, val=1)
    mat1 = np.zeros(16).reshape(4, 4)
    mat1[0, 1] = 1
    mat1[1, 0] = 1
    mat1[2, 2] = 1
    mat1[3, 3] = 1
    mat_list_example = [mat0, mat1]
    with pytest.raises(ValueError) as excinfo:
        mcmc_simulation(mat_list=mat_list_example, n_per_int=0.5)
    assert excinfo.value.args[0] == "n_per_int should be positive int"
    # n_per_int negative integer
    with pytest.raises(ValueError) as excinfo:
        mcmc_simulation(mat_list=mat_list_example, n_per_int=-1)
    assert excinfo.value.args[0] == "n_per_int should be positive int"


def test_mcmc_simulation():
    # Checking functions output the array
    mat0 = np.zeros(16).reshape(4, 4)
    np.fill_diagonal(mat0, val=1)
    mat1 = np.zeros(16).reshape(4, 4)
    mat1[0, 1] = 1
    mat1[1, 0] = 1
    mat1[2, 2] = 1
    mat1[3, 3] = 1
    mat_list_example = [mat0, mat1]
    example = mcmc_simulation(mat_list_example, 10)
    assert sum(example[:10]) == 0.
    assert sum(example[10:]) == 5.
    assert example[10] == 1.
    assert example[11] == 0.


def test_get_score_input():
    # checking functions raise the correct errors for wrong input
    # true_day is not numpy.array
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=0, simulated_day=np.zeros(13))
    assert excinfo.value.args[0] == "true_day should be numpy array"
    # simulated_day is not numpy.array
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=np.zeros(13), simulated_day=0)
    assert excinfo.value.args[0] == "simulated_day should be numpy array"
    # weight should be list
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=np.zeros(13), simulated_day=np.zeros(13),
                  weight=0)
    assert excinfo.value.args[0] == "weight should be list"
    # length of weight should be exactly 4
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=np.zeros(13), simulated_day=np.zeros(13),
                  weight=[0])
    assert excinfo.value.args[0] == "Length of weight should be 4"
    # check lengths of true_day and simulated_day
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=np.zeros(13), simulated_day=np.zeros(5))
    error_message = "Length of simulated_day is smaller than true_day"
    assert excinfo.value.args[0] == error_message
    # check all the weights are positive
    with pytest.raises(ValueError) as excinfo:
        get_score(true_day=np.zeros(13), simulated_day=np.zeros(13),
                  weight=[-1, 2, 3, 4])
    assert excinfo.value.args[0] == "All the weights should be positive"


def test_get_score():
    # Checking functions output the correct score
    true_day_1 = np.zeros(13)
    simulated_day_1 = np.ones(13)
    score_1 = get_score(true_day_1, simulated_day_1)

    true_day_2 = np.ones(13)
    simulated_day_2 = np.ones(13)
    score_2 = get_score(true_day_2, simulated_day_2)

    assert score_1 == 0.0
    assert score_2 == 10.0
