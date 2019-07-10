from pysprite import Sprite
import pytest
import numpy as np

values_success = [[39, 37.21, 20.01, 2, 2, -2, 76, 'minvar'], [41, 1.98, 2.02, 2, 2, -1, 6, 'maxvar'],
                  [25, 31.52, 15.66, 2, 2, -2, 66, 'minvar'], [46, 33.7, 26.26, 2, 2, -10, 82, 'minvar'],
                  [21, 25.1, 18.91, 2, 2, -6, 62, 'maxvar'], [48, 33.81, 23.93, 2, 2, -9, 75, 'maxvar'],
                  [20, 36.1, 17.1, 2, 2, -1, 67, 'random'], [10, 18.6, 12.75, 2, 2, -10, 39, 'random'],
                  [43, 31.93, 20.37, 2, 2, -3, 66, 'random'], [19, 17.79, 13.84, 2, 2, -4, 38, 'random'],
                  [26, 5.54, 10.73, 2, 2, -10, 24, 'maxvar'], [39, 44.59, 23.58, 2, 2, -1, 87, 'random'],
                  [32, 38.78, 31.86, 2, 2, -8, 92, 'minvar'], [22, 33.14, 25.19, 2, 2, -3, 83, 'random'],
                  [24, 42.33, 29.56, 2, 2, -8, 92, 'minvar'], [22, 22.18, 14.35, 2, 2, -4, 41, 'random'],
                  [16, 17.0, 14.49, 2, 2, -4, 43, 'minvar'], [43, 33.81, 22.26, 2, 2, -1, 77, 'minvar'],
                  [21, 3.1, 4.86, 2, 2, -3, 13, 'minvar'], [15, 23.07, 14.89, 2, 2, -4, 48, 'random'],
                  [21, 13.43, 13.18, 2, 2, -6, 33, 'random'], [12, 13.08, 11.9, 2, 2, -4, 30, 'minvar'],
                  [34, 2.18, 8.19, 2, 2, -10, 18, 'minvar'], [35, 14.17, 11.7, 2, 2, -8, 33, 'random'],
                  [26, 34.35, 18.11, 2, 2, -2, 67, 'minvar'], [32, 39.41, 25.33, 2, 2, -1, 86, 'random'],
                  [37, 14.3, 12.35, 2, 2, -7, 36, 'random'], [43, 39.0, 24.38, 2, 2, -8, 87, 'minvar'],
                  [33, 44.61, 29.4, 2, 2, -1, 92, 'minvar'], [28, 42.5, 26.84, 2, 2, -7, 96, 'random'],
                  [35, 0.37, 4.0, 2, 2, -5, 8, 'maxvar'], [46, 4.26, 4.81, 2, 2, -3, 13, 'minvar'],
                  [34, 17.76, 15.72, 2, 2, -4, 43, 'random'], [43, 23.0, 17.09, 2, 2, -5, 51, 'maxvar'],
                  [11, 49.64, 29.2, 2, 2, -9, 93, 'random'], [31, 20.77, 18.38, 2, 2, -9, 57, 'random'],
                  [22, 19.77, 13.07, 2, 2, -3, 46, 'maxvar'], [16, 22.94, 14.5, 2, 2, -2, 41, 'minvar'],
                  [18, 29.83, 18.63, 2, 2, -5, 61, 'minvar'], [34, -1.85, 3.73, 2, 2, -8, 4, 'maxvar']]

values_against_SPRITE = [[46, 39.89, 22.45, 2, 2, -1, 97, 'minvar'], [27, 42.37, 27.01, 2, 2, -10, 85, 'random'],
                         [42, 37.07, 23.48, 2, 2, -3, 76, 'maxvar'], [47, 29.13, 21.6, 2, 2, -1, 71, 'minvar'],
                         [36, -0.03, 5.14, 2, 2, -8, 9, 'random'], [37, -1.65, 1.76, 2, 2, -4, 2, 'random'],
                         [12, 6.75, 12.21, 2, 2, -10, 30, 'minvar'], [32, 38.53, 25.93, 2, 2, -5, 80, 'maxvar'],
                         [29, 7.83, 11.94, 2, 2, -9, 28, 'maxvar'], [23, 29.48, 21.92, 2, 2, -2, 69, 'random'],
                         [30, 7.27, 4.55, 2, 2, -1, 16, 'maxvar'], [15, 3.8, 7.59, 2, 2, -7, 17, 'random'],
                         [26, 40.0, 20.38, 2, 2, -8, 80, 'random'], [20, 30.05, 25.75, 2, 2, -5, 85, 'maxvar'],
                         [41, 33.54, 21.77, 2, 2, -3, 73, 'minvar'], [39, 40.97, 24.17, 2, 2, -1, 81, 'minvar'],
                         [45, 16.58, 11.44, 2, 2, -4, 36, 'maxvar'], [18, 36.44, 17.36, 2, 2, -8, 65, 'random'],
                         [37, 9.65, 10.88, 2, 2, -10, 28, 'minvar'], [45, 14.84, 10.58, 2, 2, -6, 32, 'maxvar'],
                         [16, 34.81, 24.65, 2, 2, -6, 83, 'minvar'], [28, 0.32, 5.22, 2, 2, -10, 12, 'random'],
                         [20, 18.3, 16.8, 2, 2, -7, 49, 'minvar'], [11, 5.0, 4.95, 2, 2, -3, 16, 'random'],
                         [43, 27.53, 20.17, 2, 2, -5, 63, 'minvar'], [46, 38.13, 21.82, 2, 2, -2, 73, 'maxvar'],
                         [25, 25.24, 14.2, 2, 2, -4, 57, 'random'], [24, -4.33, 2.09, 2, 2, -8, -1, 'minvar'],
                         [24, 11.38, 8.28, 2, 2, -1, 24, 'random'], [23, 6.65, 7.97, 2, 2, -8, 22, 'maxvar'],
                         [13, 29.85, 17.03, 2, 2, -1, 58, 'minvar'], [11, 41.91, 31.25, 2, 2, -8, 85, 'maxvar'],
                         [35, 28.6, 16.36, 2, 2, -3, 54, 'minvar'], [30, 26.0, 15.71, 2, 2, -1, 57, 'maxvar'],
                         [17, 9.47, 7.19, 2, 2, -6, 23, 'minvar'], [19, 20.95, 18.17, 2, 2, -8, 51, 'maxvar'],
                         [22, 34.68, 21.85, 2, 2, -3, 70, 'minvar'], [41, 27.49, 20.9, 2, 2, -10, 67, 'minvar'],
                         [43, 12.65, 12.77, 2, 2, -8, 32, 'random'], [11, 32.55, 27.81, 2, 2, -3, 78, 'random']]

values_inclusions = [[36, 8.81, 7.87, 2, 2, -5, 29, {1: 3, 2: 2, 3: 4, 4: 1}],
                     [30, 11.6, 9.44, 2, 2, -1, 33, {1: 3, 2: 5, 3: 1, 4: 1}],
                     [57, 24.58, 20.24, 2, 2, -4, 63, {1: 2, 2: 1, 3: 1, 4: 6}],
                     [47, 15.45, 15.8, 2, 2, -5, 46, {1: 3, 2: 6, 3: 1}],
                     [48, 8.92, 6.47, 2, 2, -5, 22, {1: 5, 3: 3, 4: 2}],
                     [39, 11.74, 8.09, 2, 2, -4, 25, {1: 2, 2: 1, 3: 2, 4: 5}],
                     [58, 36.47, 26.8, 2, 2, -4, 84, {1: 2, 2: 3, 4: 5}],
                     [55, 26.8, 23.63, 2, 2, -3, 71, {1: 5, 2: 2, 3: 2, 4: 1}],
                     [53, 28.98, 24.78, 2, 2, -4, 80, {2: 2, 3: 5, 4: 3}],
                     [52, 35.83, 27.78, 2, 2, -2, 79, {1: 2, 2: 4, 3: 1, 4: 3}],
                     [49, 39.31, 31.45, 2, 2, -2, 99, {1: 4, 2: 2, 3: 3, 4: 1}],
                     [46, 19.0, 14.3, 2, 2, -4, 43, {1: 3, 2: 2, 3: 1, 4: 4}],
                     [54, 8.43, 8.88, 2, 2, -5, 23, {2: 3, 3: 2, 4: 5}],
                     [44, 42.18, 33.21, 2, 2, -5, 97, {1: 3, 2: 4, 3: 2, 4: 1}],
                     [48, 26.31, 19.79, 2, 2, -4, 68, {1: 4, 2: 1, 3: 4, 4: 1}],
                     [47, 32.79, 26.07, 2, 2, -2, 87, {2: 5, 3: 5}],
                     [48, 24.67, 21.03, 2, 2, -3, 80, {2: 5, 3: 3, 4: 2}],
                     [33, 20.55, 18.79, 2, 2, -1, 61, {1: 2, 2: 3, 3: 2, 4: 3}],
                     [33, 20.27, 19.2, 2, 2, -4, 66, {1: 4, 2: 2, 3: 3, 4: 1}],
                     [42, 15.21, 13.73, 2, 2, -5, 39, {1: 8, 2: 2}]]

values_exclusions = [[35, 2.31, 1.62, 2, 2, 0, 6, {4: 0}], [35, 3.46, 2.55, 2, 2, 0, 8, {3: 0, 2: 0}],
                     [26, 3.0, 2.35, 2, 2, 0, 7, {1: 0, 4: 0}], [25, 3.12, 2.22, 2, 2, 0, 7, {1: 0, 5: 0}],
                     [49, 2.78, 1.82, 2, 2, 0, 6, {1: 0, 4: 0}], [27, 2.89, 2.24, 2, 2, 0, 7, {2: 0}],
                     [32, 2.03, 1.8, 2, 2, 0, 5, {2: 0, 3: 0}], [20, 2.8, 1.58, 2, 2, 0, 6, {1: 0}],
                     [22, 2.82, 2.32, 2, 2, 0, 7, {4: 0, 5: 0}], [31, 4.29, 1.57, 2, 2, 0, 7, {1: 0, 2: 0}],
                     [27, 2.56, 2.17, 2, 2, 0, 8, {3: 0, 6: 0}], [38, 2.74, 2.2, 2, 2, 0, 7, {5: 0, 2: 0}],
                     [20, 3.25, 2.05, 2, 2, 0, 8, {4: 0, 6: 0}], [43, 2.51, 1.64, 2, 2, 0, 6, {1: 0}],
                     [40, 3.12, 1.84, 2, 2, 0, 6, {1: 0, 2: 0}], [44, 2.7, 2.29, 2, 2, 0, 7, {4: 0, 5: 0}],
                     [37, 2.46, 1.97, 2, 2, 0, 6, {3: 0, 2: 0}], [27, 3.93, 2.4, 2, 2, 0, 9, {7: 0, 2: 0}],
                     [27, 2.85, 1.49, 2, 2, 0, 6, {1: 0}], [47, 1.6, 1.64, 2, 2, 0, 5, {2: 0}]]

values_GRIMfail = [[20, 4.21, 2.12, 2, 1, 0, 10], [20, 4.23, 2.12, 2, 2, 0, 10], [10, 8, 2.12, 2, 2, 0, 7],
                   [10, 0.1, 2.12, 2, 2, 1, 7], [10, -6, 2.12, 2, 2, -5, 5], [10, 6, 2.12, 2, 2, -5, 5]]

values_twoitems = [[26, 3.29, 2.15, 2, 2, 0, 7, 2, 'random'], [20, 5.65, 2.53, 2, 2, 0, 9, 2, 'maxvar'],
                   [30, 3.97, 2.83, 2, 2, 0, 8, 2, 'random'], [45, 2.86, 1.83, 2, 2, 0, 6, 2, 'maxvar'],
                   [25, 3.3, 2.61, 2, 2, 0, 8, 2, 'maxvar'], [46, 3.11, 2.09, 2, 2, 0, 7, 2, 'maxvar'],
                   [26, 3.21, 1.99, 2, 2, 0, 7, 2, 'minvar'], [26, 2.83, 1.89, 2, 2, 0, 6, 2, 'minvar'],
                   [46, 4.43, 2.35, 2, 2, 0, 8, 2, 'maxvar'], [48, 3.39, 1.89, 2, 2, 0, 6, 2, 'maxvar'],
                   [34, 4.0, 2.45, 2, 2, 0, 7, 2, 'maxvar'], [40, 3.7, 2.33, 2, 2, 0, 8, 2, 'minvar'],
                   [40, 2.85, 1.96, 2, 2, 0, 6, 2, 'minvar'], [37, 4.12, 2.31, 2, 2, 0, 8, 2, 'random'],
                   [25, 5.18, 2.71, 2, 2, 0, 9, 2, 'minvar'], [41, 3.79, 2.16, 2, 2, 0, 7, 2, 'random'],
                   [49, 3.17, 2.15, 2, 2, 0, 7, 2, 'random'], [48, 4.72, 2.63, 2, 2, 0, 9, 2, 'minvar'],
                   [43, 3.42, 2.21, 2, 2, 0, 7, 2, 'random'], [34, 2.34, 1.78, 2, 2, 0, 7, 2, 'random']]

values_threeitems = [[39, 3.72, 2.16, 2, 2, 0, 9, 3, 'maxvar'], [38, 4.85, 2.73, 2, 2, 0, 9, 3, 'random'],
                     [27, 3.07, 1.6, 2, 2, 0, 6, 3, 'minvar'], [45, 4.5, 2.49, 2, 2, 0, 9, 3, 'minvar'],
                     [24, 3.92, 1.87, 2, 2, 0, 7, 3, 'minvar'], [25, 4.0, 2.64, 2, 2, 0, 9, 3, 'minvar'],
                     [33, 4.02, 1.68, 2, 2, 0, 7, 3, 'maxvar'], [32, 4.93, 2.23, 2, 2, 0, 9, 3, 'maxvar'],
                     [26, 2.29, 1.39, 2, 2, 0, 5, 3, 'maxvar'], [39, 2.27, 1.58, 2, 2, 0, 5, 3, 'maxvar'],
                     [31, 3.23, 1.59, 2, 2, 0, 6, 3, 'minvar'], [46, 3.59, 2.35, 2, 2, 0, 8, 3, 'minvar'],
                     [25, 3.12, 2.11, 2, 2, 0, 7, 3, 'minvar'], [38, 4.08, 2.41, 2, 2, 0, 8, 3, 'maxvar'],
                     [38, 4.14, 2.57, 2, 2, 0, 9, 3, 'maxvar'], [34, 4.68, 2.59, 2, 2, 0, 9, 3, 'random'],
                     [40, 4.7, 2.5, 2, 2, 0, 9, 3, 'maxvar'], [25, 2.96, 1.81, 2, 2, 0, 6, 3, 'random'],
                     [37, 2.34, 1.31, 2, 2, 0, 5, 3, 'minvar'], [32, 2.35, 1.7, 2, 2, 0, 6, 3, 'random']]


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, method", values_success)
def test_success(npart, m, sd, m_prec, sd_prec, min_val, max_val, method):
    """
    Validate that pysprite can find a solution when a solution exists
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)
    assert (s.find_possible_distribution(init_method=method)[0] == "Success")


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, method", values_against_SPRITE)
def test_against_SPRITE(npart, m, sd, m_prec, sd_prec, min_val, max_val, method):
    """
    Validate that pysprite and the original SPRITE library come to the same conclusion.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)
    answer_pysprite = (s.find_possible_distribution(init_method=method)[0] == "Success")
    answer_psprite = (SPRITE(m, m_prec, sd, sd_prec, npart, min_val, max_val)[0] == "solution")
    assert (answer_psprite == answer_pysprite)


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions", values_inclusions)
def test_inclusions(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions):
    """
    Validate that pysprite can find distributions with must-include restrictions.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions)
    values = s.find_possible_distribution()[1]
    values_dist = {k: v for k, v in zip(*np.unique(values, return_counts=True))}
    match_count = [values_dist.get(k) == v for k, v in restrictions.items()]
    assert all(match_count)


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions", values_exclusions)
def test_exclusions(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions):
    """
    Validate that pysprite can find distributions with must-exclude restrictions.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions)
    values = s.find_possible_distribution()[1]
    dict_values = {k: v for k, v in zip(*np.unique(values, return_counts=True))}
    match = [dict_values.get(k, 0) == v for k, v in restrictions.items()]
    assert all(match)


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val", values_GRIMfail)
def test_GRIMfail(npart, m, sd, m_prec, sd_prec, min_val, max_val):
    """
    Validate that pysprite will fail initialization when the parameters are incorrect.
    """
    with pytest.raises(ValueError):
        Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items, method", values_twoitems)
def test_twoitems(npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items, method):
    """
    Validate that pysprite can find a solution for two-items scales.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items=n_items)
    assert (s.find_possible_distribution(init_method=method)[0] == "Success")


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items, method", values_threeitems)
def test_threeitems(npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items, method):
    """
    Validate that pysprite can find a solution for three-items scales.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, n_items=n_items)
    assert (s.find_possible_distribution(init_method=method)[0] == "Success")
