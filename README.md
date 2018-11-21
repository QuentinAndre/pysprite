pysprite: A Python 3 Re-Implementation of Heathers et al. 'Sample Parameter Reconstruction via Iterative TEchniques (SPRITE)'
===========================================================================

# I. Acknowledgments

The [SPRITE procedure](https://peerj.com/preprints/26968/), and the original code on which this library is based,
were developed by [James Heathers](http://jamesheathers.com/), [Jordan Anaya](http://www.omnesres.com/), 
[Tim van der Zee](http://www.timvanderzee.com/) and [Nicholas JL Brown](http://steamtraen.blogspot.com/).

# II. Manifest

The original code was only available for Python 2.7 (which is being shifted out of existence), and was not very... 
pythonic, which made it difficult for researchers to extend the codebase and add new features. 

The current library re-implements the features of the original code in Python 3. It features:
 * A class-oriented, easily-extensible codebase, 
 * Some features that were only available in other implementations of SPRITE (e.g., the ability to specify restrictions 
 in a more flexible way). 
 * A test suite to facilitate development.


# III. Usage

## 1. Initialization
In pysprite, a Sprite object must first be initialized as follow:

```python
npart = 20
m = 3.02
sd = 2.14
m_prec = 2
sd_prec = 2
min_val = 1
max_val = 7
s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val)
```

As in the original library, the initialization will fail if:
* The mean is smaller (greater) than the minimum (maximum) of the scale
* The mean is not possible given the precision and sample size (i.e., the GRIM test fails)

Unlike the original library however, pysprite will not return anything at this stage. 

## 2. Find a possible distribution
To find a possible distribution, the `find_possible_distribution()` method must be called after initialization.

The following parameters can be used:
* `init_method`: The initialization method of the distributions. One of `minvar`, `maxvar`, or `random`. 
* `max_iter`: The maximum number of iterations to run across all distributions.

````python
result = s.find_possible_distribution()
print(result)

result = ['Success', np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 4, 5, 5, 5, 5, 5, 5, 5]), 2.14]
````

As in the original library, `result` is always a tuple of `(outcome, dist, sd)` where:
* `outcome`: 'Success' or 'Failure', indicating whether a suitable distribution was found.
* `dist`: A suitable distribution (or the closest thing to a suitable distribution if SPRITE failed).
* `sd`: The standard deviation of the suitable distribution (or of the closest distribution if SPRITE failed).


## 3. Get a list of possible distributions
The `find_possible_distributions()` (with an s) method is available to sample several suitable distributions.

The following parameters can be used:
* `n_dists`: The number of distributions to find.
* `init_method`: The initialization method of the distributions. One of `minvar`, `maxvar`, or `random`. 
* `max_iter`: The maximum number of iterations to run across all distributions.

````python
possible_dists = s.find_possible_distributions()
print(possible_dists)

possible_dists = ['Success', [np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 3, 3, 3, 5, 5]), ...], 10]
````

With this method, `possible_dists` is a tuple of `(outcome, dists, k)` where:
* `outcome`: 'Success' or 'Failure', indicating whether the appropriate number of distribution was found.
* `dists`: A list of suitable distributions (or an empty list of none was found).
* `k`: The number of suitable distributions found.

## 4. Advanced Usage: Specifying Restrictions
In pysprite, restrictions are a dictionary of `{values: counts}` that must be present in the candidate distribution.

For instance, we want to specify that participants answered on a scale from 0 to 8, but that none of them answered `4`,
and that two of them answered `3`:

```python
npart = 20
m = 3.25
sd = 2.05
m_prec = 2
sd_prec = 2
min_val = 1
max_val = 7
restrictions = {3: 2, 4: 0}
s = Sprite(npart, m, sd, m_prec, sd_prec, min_val, max_val, restrictions=restrictions)
```

With restrictions, the initialization will fail if:
* The mean is smaller (greater) than the minimum (maximum) of the scale
* The mean is not possible given the precision, sample size, and restrictions (i.e., the GRIM test fails)
* The restrictions are invalid (e.g., they are outside the scale, or one of the bounds of the scale is restricted to
have zero values).

When restrictions are specified, the `init_method` argument `find_possible_distribution()` and 
`find_possible_distributions()` has no effect. The distribution will always be initialized at random, such that it
conforms to the restrictions.

# IV. Development

## 1. Up-and-coming features

* Support for composite scale scores

## 2. Running tests

````cmd
pip install pytest
pytest
````


