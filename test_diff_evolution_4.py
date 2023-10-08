import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_4():
    from diff_evolution import differential_evolution
    SEED = 21
    random.seed(SEED)
    np.random.seed(SEED)

    def rastrigin(array, A=10):
        return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))
  
    def griewank(array):
        term_1 = (array[0] ** 2 + array[1] ** 2) / 2
        term_2 = np.cos(array[0]/ np.sqrt(2)) * np.cos(array[1]/ np.sqrt(2))
        return 1 + term_1 - term_2
  
    def rosenbrock(array):
        return (1 - array[0]) ** 2 + 100 * (array[1] - array[0] ** 2) ** 2

    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 1.290061391046038e-10
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] == 1.343844239443115e-06
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 9.950103123657073e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] == 0.00018277054575399632
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] == 3.552713678800501e-15
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] == 1.0658141036401503e-14
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 1.0365393876554663e-08
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] == 9.535293283846613e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 2.2518667819326765e-08
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] == 4.785992864242417e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 0.00014878031395859637
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] == 0.003355577154181333
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] == 1.0889067425523535e-12
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] == 1.7763568394002505e-14
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 4.950996768116056e-08
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] == 1.9072627743810244e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 5.195389007894846e-09
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] == 3.084298043631861e-07
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 2.3878628940821045e-06
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] == 0.0012417578836654286
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] == 7.460698725481052e-14
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 4.870859271477457e-10
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] == 4.033337043907181e-06
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] == 8.091793901598976e-09
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] == 2.020540636671342e-05
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] == 4.937244557545739e-05
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] == 0.003326951104462239
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] == 1.2256862191861728e-13
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] == 0.9949590570932898
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] == 0.0
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] == 9.403803957752643e-09
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] == 1.2667463540338986e-06
    assert list(differential_evolution(rastrigin, np.array([[-20, 20], [-20, 20]]), init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] == 0.0
