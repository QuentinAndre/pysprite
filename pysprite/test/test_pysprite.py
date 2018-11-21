from pysprite import Sprite, SPRITE
import pytest
import numpy as np

values_success = [[39, 37.21, 20.01, 2, 2, -2, 76], [41, 1.98, 2.02, 2, 2, -1, 6], [25, 31.52, 15.66, 2, 2, -2, 66],
                  [46, 33.7, 26.26, 2, 2, -10, 82], [21, 25.1, 18.91, 2, 2, -6, 62], [48, 33.81, 23.93, 2, 2, -9, 75],
                  [20, 36.1, 17.1, 2, 2, -1, 67], [10, 18.6, 12.75, 2, 2, -10, 39], [43, 31.93, 20.37, 2, 2, -3, 66],
                  [19, 17.79, 13.84, 2, 2, -4, 38], [26, 5.54, 10.73, 2, 2, -10, 24], [39, 44.59, 23.58, 2, 2, -1, 87],
                  [32, 38.78, 31.86, 2, 2, -8, 92], [22, 33.14, 25.19, 2, 2, -3, 83], [24, 42.33, 29.56, 2, 2, -8, 92],
                  [22, 22.18, 14.35, 2, 2, -4, 41], [16, 17.0, 14.49, 2, 2, -4, 43], [43, 33.81, 22.26, 2, 2, -1, 77],
                  [21, 3.1, 4.86, 2, 2, -3, 13], [15, 23.07, 14.89, 2, 2, -4, 48], [21, 13.43, 13.18, 2, 2, -6, 33],
                  [12, 13.08, 11.9, 2, 2, -4, 30], [34, 2.18, 8.19, 2, 2, -10, 18], [35, 14.17, 11.7, 2, 2, -8, 33],
                  [26, 34.35, 18.11, 2, 2, -2, 67], [32, 39.41, 25.33, 2, 2, -1, 86], [37, 14.3, 12.35, 2, 2, -7, 36],
                  [43, 39.0, 24.38, 2, 2, -8, 87], [33, 44.61, 29.4, 2, 2, -1, 92], [28, 42.5, 26.84, 2, 2, -7, 96],
                  [35, 0.37, 4.0, 2, 2, -5, 8], [46, 4.26, 4.81, 2, 2, -3, 13], [34, 17.76, 15.72, 2, 2, -4, 43],
                  [43, 23.0, 17.09, 2, 2, -5, 51], [11, 49.64, 29.2, 2, 2, -9, 93], [31, 20.77, 18.38, 2, 2, -9, 57],
                  [22, 19.77, 13.07, 2, 2, -3, 46], [16, 22.94, 14.5, 2, 2, -2, 41], [18, 29.83, 18.63, 2, 2, -5, 61],
                  [34, -1.85, 3.73, 2, 2, -8, 4]]

values_couldfail = [[46, 39.89, 22.45, 2, 2, -1, 97], [27, 42.37, 27.01, 2, 2, -10, 85],
                    [42, 37.07, 23.48, 2, 2, -3, 76], [47, 29.13, 21.6, 2, 2, -1, 71],
                    [36, -0.03, 5.14, 2, 2, -8, 9], [37, -1.65, 1.76, 2, 2, -4, 2],
                    [12, 6.75, 12.21, 2, 2, -10, 30], [32, 38.53, 25.93, 2, 2, -5, 80],
                    [29, 7.83, 11.94, 2, 2, -9, 28], [23, 29.48, 21.92, 2, 2, -2, 69],
                    [30, 7.27, 4.55, 2, 2, -1, 16], [15, 3.8, 7.59, 2, 2, -7, 17],
                    [26, 40.0, 20.38, 2, 2, -8, 80], [20, 30.05, 25.75, 2, 2, -5, 85],
                    [41, 33.54, 21.77, 2, 2, -3, 73], [39, 40.97, 24.17, 2, 2, -1, 81],
                    [45, 16.58, 11.44, 2, 2, -4, 36], [18, 36.44, 17.36, 2, 2, -8, 65],
                    [37, 9.65, 10.88, 2, 2, -10, 28], [45, 14.84, 10.58, 2, 2, -6, 32],
                    [16, 34.81, 24.65, 2, 2, -6, 83], [28, 0.32, 5.22, 2, 2, -10, 12],
                    [20, 18.3, 16.8, 2, 2, -7, 49], [11, 5.0, 4.95, 2, 2, -3, 16],
                    [43, 27.53, 20.17, 2, 2, -5, 63], [46, 38.13, 21.82, 2, 2, -2, 73],
                    [25, 25.24, 14.2, 2, 2, -4, 57], [24, -4.33, 2.09, 2, 2, -8, -1],
                    [24, 11.38, 8.28, 2, 2, -1, 24], [23, 6.65, 7.97, 2, 2, -8, 22],
                    [13, 29.85, 17.03, 2, 2, -1, 58], [11, 41.91, 31.25, 2, 2, -8, 85],
                    [35, 28.6, 16.36, 2, 2, -3, 54], [30, 26.0, 15.71, 2, 2, -1, 57],
                    [17, 9.47, 7.19, 2, 2, -6, 23], [19, 20.95, 18.17, 2, 2, -8, 51],
                    [22, 34.68, 21.85, 2, 2, -3, 70], [41, 27.49, 20.90, 2, 2, -10, 67],
                    [43, 12.65, 12.77, 2, 2, -8, 32], [11, 32.55, 27.81, 2, 2, -3, 78]]

values_must_include = [[36, 8.81, 7.87, 2, 2, -5, 29, {1: 3, 2: 2, 3: 4, 4: 1}],
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

values_must_exclude = [[35, 2.31, 1.62, 2, 2, 0, 6, {4: 0}], [35, 3.46, 2.55, 2, 2, 0, 8, {3: 0, 2: 0}],
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


@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val", values_success)
def test_success(npart, m, sd, m_prec, sd_prec, min_val, max_val):
    """
    Validate that pysprite can find a solution when a solution exists
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)
    assert (s.find_possible_distribution()[0] == "Success")

@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val", values_couldfail)
def test_against_SPRITE(npart, m, sd, m_prec, sd_prec, min_val, max_val):
    """
    Validate that pysprite and the original SPRITE library come to the same conclusion.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)
    answer_pysprite = (s.find_possible_distribution()[0] == "Success")
    answer_psprite = (SPRITE(m, m_prec, sd, sd_prec, npart, min_val, max_val)[0] == "solution")
    assert (answer_psprite == answer_pysprite)

@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions", values_must_include)
def test_inclusions(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions):
    """
    Validate that pysprite can find distributions with must-include restrictions.
    """
    s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions)
    values = s.find_possible_distribution()[1]
    values_dist = {k: v for k, v in zip(*np.unique(values, return_counts=True))}
    match_count = [values_dist.get(k) == v for k, v in restrictions.items()]
    assert all(match_count)

@pytest.mark.parametrize("npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions", values_must_exclude)
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
