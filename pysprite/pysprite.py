import numpy as np
import math
from fractions import Fraction
from itertools import product
from warnings import warn, catch_warnings, simplefilter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def deviation(data, u):
    return (sum([(i - u) ** 2 for i in data]) / (len(data) - 1)) ** .5


def dict_to_array(dicti):
    """
    A utility function to convert dictionaries into arrays
    :param dicti: A dictionary of {value: n_occurrences}
    :return: An array of values
    """
    vals = [i for i in dicti.keys()]
    counts = [i for _, i in dicti.items()]
    return np.repeat(vals, counts)

def grim(n, mu, prec=2, n_items=1):
    """
    Test that a mean mu reported with a decimal precision prec is possible, given a number of observations n and a
     number of items n_items.
    :param n: The number of observations
    :param mu: The mean
    :param prec: The precision (i.e., number of decimal places) of the mean
    :param n_items: The number of scale items that were averaged. Default is 1.
    :return: True if the mean is possible, False otherwise.
    """
    if n*n_items >= 10**prec:
        warn("The effective number of data points is such that GRIM will always find a solution.")
    cval = np.round(mu * n * n_items, 0)
    valid = np.round(cval/n/n_items, prec) == np.round(mu, prec)
    return valid


class GrimSearch:
    def __init__(self, n, mus, precs, n_items, max_diff=None, varnames=None, cellnames=None):
        """
        A GrimSearch object, that can be used to find possible combinations of cell sizes.
        :param n: The total number of observations, across all cells
        :param mus: A J x K matrix of means, where
            J is the number of cells
            K is the number of variables.
        :param precs: A K-length vector specifying the decimal precision of each mean.
        :param n_items: A K-length vector specifying how many items were averaged to form the mean.
        :param max_diff: The maximum difference in number of observations between two cells. Can be used to narrow down
        the possible cell sizes to check.
        """
        self.N = n
        self.mus = np.array(mus)
        self.n_items = np.array(n_items)
        self.precs = precs
        self._j = self.mus.shape[0]
        self._k = self.mus.shape[1]

        if varnames is None:
            self.varnames = [i for i in range(0, self._k)]
        else:
            self.varnames = varnames
        if cellnames is None:
            self.cellnames = [i for i in range(0, self._j)]
        else:
            self.cellnames = cellnames

        if max_diff:
            avg_n_per_cell = int(np.round(self.N / self._j, 0))
            min_n_per_cell = max(1, avg_n_per_cell - int(np.round(max_diff / 2)))
            max_n_per_cell = min(self.N, avg_n_per_cell + int(np.round(max_diff / 2)))
        else:
            min_n_per_cell = 1
            max_n_per_cell = self.N
        self._n_range = list(np.arange(min_n_per_cell, max_n_per_cell))
        n_combs = len(self._n_range) ** self._j
        if n_combs > 200:
            warn(
                f"The number of candidate cell sizes is very large (N = {n_combs}). You might want to set the max_diff parameter to a smaller value.")
            self._test_set = None
        else:
            self._test_set = self._build_test_set()
        self._grim_results = None

    def _build_test_set(self):
        """
        Build the set of cell sizes combinations to test.
        :return: An N x J matrix describing the combinations and the size of each cell
        """
        test_set = np.array(list(filter(lambda x: np.sum(x) == self.N, list(product(*[self._n_range] * self._j)))))
        return test_set

    def _apply_grim_to_test_sets(self):
        if self._test_set is None:
            self._test_set = self._build_test_set()
        n_tests = self._test_set.shape[0]
        grim_results = np.empty((n_tests, self._j, self._k))
        with catch_warnings():
            simplefilter("ignore")
            for i, ns in enumerate(self._test_set):
                for j, (n, mu) in enumerate(zip(ns, self.mus)):
                    grim_results[i, j, :] = [grim(n, m, p, n_it) for m, p, n_it in zip(mu, self.precs, self.n_items)]
        return grim_results

    def find_solutions(self):
        """
        Find possible cell size combinations.
        :return: A list of possible cell size combinations
        """
        if self._grim_results is None:
            self._grim_results = self._apply_grim_to_test_sets()
        n_grim_passed = self._grim_results.sum(axis=(1, 2))
        solutions = self._test_set[n_grim_passed == (self._j * self._k), :]
        return solutions

    def get_n_valid_means(self):
        """
        Return the number of valid means, for each cell size combinations.
        :return:
        """
        if self._grim_results is None:
            self._grim_results = self._apply_grim_to_test_sets()
        n_valids = self._grim_results.sum(axis=(1, 2))
        index = ["-".join([str(i) for i in t]) for t in self._test_set]
        return pd.Series(n_valids, index=index)

    def summarize_grim_checks(self):
        """
        Return the number of valid means, for each cell size combinations.
        :return:
        """
        if self._grim_results is None:
            self._grim_results = self._apply_grim_to_test_sets()
        values = self._grim_results.reshape(-1)
        ns = np.tile(self._test_set, (1, self._k)).reshape(-1)
        sample = np.repeat(["-".join([str(i) for i in t]) for t in self._test_set], self._j * self._k)
        variable = np.tile(np.repeat(self.varnames, self._j), len(self._test_set))
        cell = np.tile(self.cellnames, self._k * len(self._test_set))
        return pd.DataFrame({"Combination": sample, "Cell": cell, "Cell N": ns, "Variable": variable, "Valid": values})

    def plot_grim_checks(self, threshold=None, plot_kwargs=None,
                         heatmap_kwargs=dict(square=False, linewidths=.5, cmap="Blues")):
        """
        :param threshold: Will exclude any sample size combination that gives less than this number of valid means
        :param plot_kwargs: The kwargs to pass to plt.subplots
        :param heatmap_kwargs: The kwargs to pass to sns.heatmap.
        :return:
        """
        if self._grim_results is None:
            self._grim_results = self._apply_grim_to_test_sets()
        if threshold:
            mask = self._grim_results.sum(axis=(1, 2)) >= threshold
            grim_results = self._grim_results[mask]
            test_set = self._test_set[mask]
        else:
            grim_results = self._grim_results
            test_set = self._test_set
        if plot_kwargs is None:
            plot_kwargs = {}
        if heatmap_kwargs is None:
            heatmap_kwargs = {}
        n_cells = self._j
        fig, axes = plt.subplots(n_cells, 1, **plot_kwargs)
        for i, ax in enumerate(axes):
            cell_sizes = test_set[:, i]
            solutions = grim_results[:, i, :]
            sns.heatmap(solutions.T, ax=ax, cbar=False, vmin=0, vmax=1, **heatmap_kwargs)
            ax.set_xticks(np.arange(0.5, len(cell_sizes) + 0.5))
            ax.set_xticklabels(cell_sizes)
            ax.set_yticklabels(self.varnames, rotation="horizontal")
            ax.set_title(self.cellnames[i])
        plt.tight_layout()
        return fig, ax


class Sprite:
    def __init__(self, n, mu, sd, mu_prec, sd_prec, min_val, max_val, restrictions=None, n_items=1):
        """
        Initialize a Sprite object, that can be used to generate candidate distributions.

        :param n: The number of observations
        :param mu: The mean value
        :param sd: The standard deviation
        :param mu_prec: The precision (i.e., number of decimal places) of the mean
        :param sd_prec: The precision (i.e., number of decimal places) of the standard deviation
        :param min_val: The minimum value of the scale
        :param max_val: The maximum value of the scale
        :param restrictions: A dictionary {value: counts} specifying the number of observations counts that take the value value
        :param n_items: The number of scale items that were averaged. Default is 1.
        """
        # Encode the restrictions
        if restrictions is not None:
            self._has_restrictions = True
            self.restrictions = self._validate_restrictions(restrictions, min_val, max_val)
            self.restricted_values = list(self.restrictions.keys())
            self._increase_var = self._increase_var_withcheck  # Validate candidate values against restrictions: slower
            self._decrease_var = self._decrease_var_withcheck
        else:
            self._has_restrictions = False
            self.restrictions = {}
            self.restricted_values = []
            self._increase_var = self._increase_var_nocheck  # No validation: faster
            self._decrease_var = self._decrease_var_nocheck

        self.n = n
        self.mu = mu
        self.sd = sd
        self.mu_prec = mu_prec
        self.sd_prec = sd_prec
        self.n_items = n_items


        if n_items == 1:
            self.granularity = 1
            self.scale = list(np.arange(min_val, max_val + 1))
        elif n_items == 2:
            self.granularity = 1 / n_items
            self.scale = list(np.arange(min_val, max_val + self.granularity, self.granularity))
        else:
            self.granularity = Fraction(1, n_items)  # Fractions to avoid precision errors. Slower but failsafe.
            self.scale = list(np.arange(min_val, max_val + self.granularity, self.granularity))

        if not self._grim_test_valid():
            raise ValueError("GRIM Failed: The SPRITE method cannot be applied.")

        self.data = self._init_data()

    def _validate_restrictions(self, restrictions, min_val, max_val):
        exclusions = [k for k, v in restrictions.items() if v == 0]
        out_of_scale = [k for k in restrictions.keys() if k < min_val or k > max_val]
        if min_val in exclusions:
            raise ValueError(
                "The minimum value of the scale cannot be 0-restricted. Adjust the boundaries of the scale instead."
            )
        if max_val in exclusions:
            raise ValueError(
                "The maximum value of the scale cannot be 0-restricted. Adjust the boundaries of the scale instead."
            )
        if len(out_of_scale) > 0:
            raise ValueError(
                "Some restrictions apply to values out of the scale. Please adjust the restrictions."
            )
        return restrictions

    def _init_data(self, init_method="maxvar"):
        """
        Initialize a random consistent with the mean and sample size.
        :param init_method: The initialization method. Can be one of 'random', 'maxvar', or 'minvar'.
        :return: The initialized dataset.
        """
        if init_method not in ['random', 'maxvar', 'minvar']:
            raise ValueError("The method {} was not understood. Must be one of 'minvar', 'maxvar' or 'random'".format(
                init_method))

        # Compute upper and lower bounds of the sum of values
        lower = self.mu - .5 / (10 ** self.mu_prec)
        upper = self.mu + .5 / (10 ** self.mu_prec)

        if lower < 0:
            l_bound = int(lower * self.n)
        else:
            l_bound = int(math.floor(lower * self.n))
        if upper < 0:
            u_bound = int(math.ceil(upper * self.n))
        else:
            u_bound = int(upper * self.n)

        if self._has_restrictions:
            data = self._init_restricted_data(l_bound, u_bound)
        else:
            if init_method == "minvar":
                data = self._init_minvar_data(l_bound, u_bound)
            elif init_method == "maxvar":
                data = self._init_maxvar_data(l_bound, u_bound)
            else:
                data = self._init_random_data(l_bound, u_bound)
        return data

    def _init_restricted_data(self, l_bound, u_bound):
        """
        Initialize a distribution with restrictions. This function is not deterministic.
        :param l_bound: The lower bound of the sum of values in the distribution
        :param u_bound: The upper bound of the sum of values in the distribution
        :return: A random distribution with restrictions.
        """
        n = self.n
        n_restricted = 0
        scale = self.scale
        maxscale = max(scale)
        minscale = min(scale)
        target_sum = np.random.choice(range(l_bound, u_bound + 1))
        total_left = target_sum
        dist = []
        for k, v in self.restrictions.items():  # Create a distribution in accordance with all the restrictions
            dist += [k] * v
            n_restricted += v
            total_left -= k * v
        if (n_restricted > n) or (total_left < 0):
            raise ValueError("The distribution is impossible given the restrictions")

        dist += [minscale] * (
                n - n_restricted)  # Padding the rest of the distribution with the minimum value of the scale

        for i in range(0, 10000):
            ix = np.random.choice(np.arange(n_restricted, n))  # Selected an unrestricted value
            val = dist[ix]
            candidate = val + self.granularity  # Increment this unrestricted value until another valid value is found
            while (candidate in self.restricted_values) and (candidate <= maxscale):
                candidate += self.granularity
            if sum(dist) + candidate - val <= target_sum:  # If it is not too large, add it.
                dist[ix] = candidate
            if sum(dist) == target_sum:
                return self._array_to_dict(dist)

        raise ValueError("Could not find a suitable distribution given the restrictions.")

    def _init_random_data(self, l_bound, u_bound):
        """
        Initialize a random distribution. This function is not deterministic.
        :param l_bound: The lower bound of the sum of values in the distribution
        :param u_bound: The upper bound of the sum of values in the distribution
        :return: A random distribution.
        """
        n = self.n
        scale = self.scale
        maxscale = max(scale)
        target_sum = np.random.choice(range(l_bound, u_bound + 1))  # Choose a possible sum at random.
        dist = n * [min(scale)]
        r = np.arange(n)
        while sum(dist) != target_sum:
            ix = np.random.choice(r)
            newval = dist[ix] + self.granularity
            if (newval) <= maxscale:
                dist[ix] = newval
        return self._array_to_dict(dist)

    def _init_minvar_data(self, l_bound, u_bound):
        """
        Initialize a distribution with a minimum amount of variance. This function is deterministic
        :param l_bound: The lower bound of the sum of values in the distribution
        :param u_bound: The upper bound of the sum of values in the distribution
        :return: The min-variance distribution
        """
        n = self.n
        if l_bound == u_bound:  # Only one total is possible
            local_mu = l_bound / float(n)
            dist = n * [int(local_mu)]  # Repeat the integer closest to the mean n times.
            if l_bound > 0:  # Adjust by adding and subtracting until the true mean is reached.
                while sum(dist) < l_bound:
                    dist.sort()
                    dist[0] += self.granularity
            else:
                while sum(dist) > l_bound:
                    dist.sort(reverse=True)
                    dist[0] -= self.granularity
            return self._array_to_dict(dist)
        else:  # Multiple totals are possible: for all possible values of the total, repeat procedure above
            minvar = 1000
            for i in range(l_bound, u_bound + 1):
                local_mu = i / float(n)
                dist = n * [int(local_mu)]
                if l_bound > 0:
                    while sum(dist) < i:
                        dist.sort()
                        dist[0] = dist[0] + self.granularity
                else:
                    while sum(dist) > i:
                        dist.sort(reverse=True)
                        dist[0] = dist[0] - self.granularity
                local_sd = deviation(dist, local_mu)
                if local_sd < minvar:  # Keep the result if it has less variance than another result.
                    minvar = local_sd
                    minvar_dist = dist
            return self._array_to_dict(minvar_dist)

    def _init_maxvar_data(self, l_bound, u_bound):
        """
        Initialize a distribution with a maximum amount of variance. This function is deterministic
        :param l_bound: The lower bound of the sum of values in the distribution
        :param u_bound: The upper bound of the sum of values in the distribution
        :return: The max-variance distribution
        """
        n = self.n
        scale = self.scale
        maxscale = max(scale)
        minscale = min(scale)

        if l_bound == u_bound:  # Only one total is possible
            local_mu = l_bound / float(n)
            if maxscale - local_mu < local_mu - minscale:  # If the mean is closer to the end of the scale...
                dist = [maxscale]
            else:  # If the mean is closer to the beginning of the scale...
                dist = [minscale]
            for i in range(n - 1):  # Target the mean by adding top and bottom scale values
                if np.mean(dist) <= local_mu:
                    dist.append(maxscale)
                else:
                    dist.append(minscale)
            dist.sort()
            diff = l_bound - sum(dist)
            if diff < 0:  # If the distribution does not add up the total...
                dist[-1] += diff  # Add the difference to the top of the scale if it is negative
            else:
                dist[0] += diff  # Or add it to the bottom if it is positive
            return self._array_to_dict(dist)

        else:  # Multiple totals are possible: for all possible values of the total, repeat procedure above
            maxvar = 0
            for i in range(l_bound, u_bound + 1):
                local_mu = i / float(n)
                if maxscale - local_mu < local_mu - minscale:
                    dist = [maxscale]
                else:
                    dist = [minscale]
                for _ in range(n - 1):
                    if np.mean(dist) <= local_mu:
                        dist.append(maxscale)
                    else:
                        dist.append(minscale)
                dist.sort()
                diff = i - sum(dist)
                if diff < 0:
                    dist[-1] = dist[-1] + diff
                else:
                    dist[0] = dist[0] + diff
                localvar = deviation(dist, local_mu)
                if localvar > maxvar:  # Keep the result if it has higher variance than another result.
                    maxvar = localvar
                    maxvar_dist = dist
            return self._array_to_dict(maxvar_dist)

    def _grim_test_valid(self):
        """
        Apply the GRIM test.
        :return: True if the mean is possible given the sample size, scale, and number of items, False otherwise.
        """
        if self.mu > self.scale[-1]:  # Mean greater than max value
            return False
        if self.mu < self.scale[0]:  # Mean smaller than min value
            return False
        if round(round(self.mu * self.n * self.n_items, 0) / self.n / self.n_items, self.mu_prec) == self.mu:  # GRIM
            return True
        else:
            return False

    def _array_to_dict(self, array):
        """
        A utility function to convert arrays into dictionaries
        :param array: An array of values
        :return: A dictionary of {value: n_occurences}
        """
        unique, counts = np.unique(array, return_counts=True)
        data = dict(zip(unique, counts))
        for el in self.scale:
            if el not in data.keys():
                data[el] = 0
        return data

    def find_possible_distribution(self, init_method="maxvar", max_iter=100000):
        """
        Find one possible distribution.
        :param init_method: The initialization method for the data.
        :return: (result, distribution, current_sd):
            result: The result of the procedure. Can be 'success' or 'failure'.
            distribution: The distribution that was found (if success) / that had the closest variance (if failure).
            current_sd: The SD of the distribution that was found (success) / that had the closest variance (failure).
        """

        target_sd = self.sd
        self.data = self._init_data(init_method)
        dist = dict_to_array(self.data)
        
        mu = sum(dist) / len(dist)

        for i in range(max_iter):
            current_sd = np.round(deviation(dict_to_array(self.data), mu), 2)
            if current_sd == target_sd:
                return ["Success", dict_to_array(self.data), current_sd]
            elif target_sd < current_sd:
                self._decrease_var()
            else:
                self._increase_var()
        return ["Failure", dict_to_array(self.data), current_sd]

    def find_possible_distributions(self, n_dists=10, init_method="maxvar", max_iter=100000):
        """

        :param n_dists: The number of distributions to return.
        :param init_method:  The initialization method for the data. 'Random' is recommended.
        :param max_iter: The maximum number of iterations (across all distributions searched).
        :return: (result, distribution, n_found):
            result: Result of the procedure: 'success' or 'failure' (if less than n_dists are found).
            distribution: A list of distributions that were found.
            n_found: The number of distributions found.
        """

        target_sd = self.sd
        self.data = self._init_data(init_method)
        dist = dict_to_array(self.data)

        mu = sum(dist) / len(dist)
        k = 0
        possible = []
        for i in range(max_iter):
            current_sd = round(deviation(dict_to_array(self.data), mu), 2)

            if current_sd == target_sd:
                k += 1
                possible.append(dict_to_array(self.data))
                if k == n_dists:
                    return ["Success", possible, k]
                else:
                    self.data = self._init_data(init_method)  # Re-initialize the data.
            elif target_sd < current_sd:
                self._decrease_var()
            else:
                self._increase_var()
        return ["Failure", possible, k]

    def _validate_values(self, to_increment, to_decrement):
        """
        Check that the values about to be incremented/decrement are not restricted values.
        :param to_increment: The value that will be incremented
        :param to_decrement: The value that will be decremented
        :return: True if the values are valid, else False.
        """
        rv = self.restricted_values
        valid_increment = (to_increment not in rv) & ((to_increment + self.granularity) not in rv)
        valid_decrement = (to_decrement not in rv) & ((to_decrement - self.granularity) not in rv)
        return valid_decrement & valid_increment

    def _increase_var_withcheck(self):
        """
        Increases the variance of the distribution (while keeping the mean constant) by pushing values to the extreme
        ends of the scale. This version checks that the new distribution does not violate the restrictions
        imposed on the data, and is therefore slower.
        :return: True if the variance could be successfully increased, False otherwise.
        """
        valid_first_values = (i for i in np.random.permutation(self.scale[1:-1]) if
                              self.data.get(i) != 0)  # Data points that are not at the extreme of the scale
        try:
            first_value = next(valid_first_values)
        except StopIteration:
            raise BaseException("Could not find a proper value")

        valid_second_values = (i for i in np.random.permutation(self.scale[1:-1]) if
                               self.data.get(i) != 0)  # Data points that are not at the extreme

        second_value = next(valid_second_values)
        while (first_value == second_value) & (self.data.get(first_value) < 2):
            try:
                second_value = next(valid_second_values)
            except StopIteration:
                raise BaseException("Maximum SD reached")

        if (first_value >= second_value) & self._validate_values(first_value, second_value):
            self._increment(first_value)
            self._decrement(second_value)
            return True
        elif (first_value < second_value) & self._validate_values(second_value, first_value):
            self._increment(second_value)
            self._decrement(first_value)
            return True
        return False

    def _increase_var_nocheck(self):
        """
        Increases the variance of the distribution (while keeping the mean constant) by pushing values to the extreme
        ends of the scale. This version does not check that the new distribution does not violate the restrictions
        imposed on the data, and is therefore faster
        :return: True
        """
        valid_first_values = (i for i in np.random.permutation(self.scale[1:-1]) if
                              self.data.get(i) != 0)
        try:
            first_value = next(valid_first_values)
        except StopIteration:
            raise BaseException("Could not find a proper value")

        valid_second_values = (i for i in np.random.permutation(self.scale[1:-1]) if
                               self.data.get(i) != 0)

        second_value = next(valid_second_values)
        while (first_value == second_value) & (self.data.get(first_value) < 2):
            try:
                second_value = next(valid_second_values)
            except StopIteration:
                raise BaseException("Maximum SD reached")

        if first_value >= second_value:
            self._increment(first_value)
            self._decrement(second_value)
        else:
            self._increment(second_value)
            self._decrement(first_value)
        return True

    def _decrease_var_withcheck(self):
        """
        Decreases the variance of the distribution (while keeping the mean constant) by pushing values toward the mean.
        This version checks that the new distribution does not violate the restrictions imposed on the data,
        and is therefore slower.
        :return: True if the variance could be successfully decreased, False otherwise.
        """
        if np.random.random() > .5:  # Start by grabbing a data point at the left of the scale.
            valid_first_value = (i for i in np.random.permutation(self.scale[:-2]) if
                                 self.data.get(i) != 0)  # List of valid data points not on the extreme right
            try:
                to_increment = next(valid_first_value)
            except StopIteration:
                raise BaseException("Could not find a proper value")

            valid_second_value = (i for i in np.random.permutation(self.scale[self.scale.index(to_increment) + 2:]) if
                                  self.data.get(i) != 0)  # List of valid data points greater than first

            try:
                to_decrement = next(valid_second_value)
            except StopIteration:
                return False  # No value greater than the first could be found.

        else:  # Same logic, but start by grabbing a data point at the right of the scale.
            valid_first_value = (i for i in np.random.permutation(self.scale[2:]) if
                                 self.data.get(i) != 0)
            try:
                to_decrement = next(valid_first_value)
            except StopIteration:
                raise BaseException("Could not find a proper value")

            valid_second_value = (i for i in np.random.permutation(
                self.scale[:self.scale.index(to_decrement) - 1]) if
                                  self.data.get(i) != 0)

            try:
                to_increment = next(valid_second_value)
            except StopIteration:
                return False  # No value smaller than the first could be found.

        if self._validate_values(to_increment, to_decrement):
            self._increment(to_increment)
            self._decrement(to_decrement)
            return True
        else:
            return False

    def _decrease_var_nocheck(self):
        """
        Decreases the variance of the distribution (while keeping the mean constant) by pushing values toward the mean.
        This version does not check that the new distribution does not violate the restrictions imposed on the data,
        and is therefore faster.
        :return: True if the variance could be successfully decreased, False otherwise.
        """
        if np.random.random() > .5:
            valid_first_value = (i for i in np.random.permutation(self.scale[:-2]) if
                                 self.data.get(i) != 0)
            try:
                will_increment = next(valid_first_value)
            except StopIteration:
                raise BaseException("Could not find a proper value")

            valid_second_value = (i for i in np.random.permutation(self.scale[self.scale.index(will_increment) + 2:]) if
                                  self.data.get(i) != 0)

            try:
                will_decrement = next(valid_second_value)
            except StopIteration:
                return False

        else:
            valid_first_value = (i for i in np.random.permutation(self.scale[2:]) if
                                 self.data.get(i) != 0)
            try:
                will_decrement = next(valid_first_value)
            except StopIteration:
                raise BaseException("Could not find a proper value")

            valid_second_value = (i for i in np.random.permutation(
                self.scale[:self.scale.index(will_decrement) - 1]) if
                                  self.data.get(i) != 0)

            try:
                will_increment = next(valid_second_value)
            except StopIteration:
                return False

        self._increment(will_increment)
        self._decrement(will_decrement)
        return True

    def _increment(self, value):
        """
        Increment the given value in the data
        :param value: The value to increment
        :return: None
        """
        self.data[value] -= self.granularity
        self.data[value + self.granularity] += self.granularity

    def _decrement(self, value):
        """
        Decrement the given value in the data
        :param value: The value to decrement
        :return: None
        """
        self.data[value] -= self.granularity
        self.data[value - self.granularity] += self.granularity


if __name__ == "__main__":

    npart, mu, sd, m_prec, sd_prec, min_val, max_val, n_items, method = [40, 4.85, 2.73, 2, 2, 0, 9, 1, 'random']
    npart, mu, sd, m_prec, sd_prec, min_val, max_val, n_items, method = [38, 4.85, 2.73, 2, 2, 0, 9, 3, 'random']
    s = Sprite(npart, mu, sd, m_prec, sd_prec, min_val, max_val, n_items=n_items)
    results = s.find_possible_distribution(init_method=method)

    print(results)