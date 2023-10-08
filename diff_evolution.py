import numpy as np
import random
from scipy.stats import qmc
import math


def differential_evolution(fobj, bounds, mutation_coefficient=0.5,
                           crossover_coefficient=0.5, population_size=50, iterations=50,
                           init_setting='random', mutation_setting='rand1',
                           selection_setting='current', p_min=0.1, p_max=0.2):
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    bounds = np.array(bounds)
    dimensions = len(bounds)

    if init_setting == 'LatinHypercube':
        population = qmc.LatinHypercube(d=dimensions, seed=SEED)
        assert population.__class__ == qmc.LatinHypercube
        population = population.random(n=population_size)
    elif init_setting == 'Halton':
        population = qmc.Halton(d=dimensions, seed=SEED)
        assert population.__class__ == qmc.Halton
        population = population.random(n=population_size)
    elif init_setting == 'Sobol':
        population = qmc.Sobol(d=dimensions, seed=SEED)
        assert population.__class__ == qmc.Sobol
        population = population.random(n=population_size)
    else:
        population = np.random.rand(population_size, dimensions)
    min_bound, max_bound = bounds.T
    diff = np.fabs(min_bound - max_bound)
    population_denorm = min_bound + population * diff
    fitness = np.asarray([fobj(ind) for ind in population_denorm])
    best_idx = np.argmin(fitness)
    best = population_denorm[best_idx]

    for iteration in range(iterations):
        for population_index in range(population_size):
            idxs = np.setdiff1d(np.arange(population_size), [best_idx, population_index], assume_unique=True)
            if mutation_setting == 'rand2':
                a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]
                assert 'e' in locals(), "Данный ассерт проверяет, что вы точно написали формулу"
                assert 'd' in locals(), "Данный ассерт проверяет, что вы точно написали формулу"
                mutant = np.clip(a + mutation_coefficient * (b - c) + mutation_coefficient * (d - e), 0, 1)
            elif mutation_setting == 'best1':
                index_of_best1 = np.setdiff1d(np.argsort(fitness), [best_idx, population_index], assume_unique=True)[0]
                idxs = np.setdiff1d(idxs, [index_of_best1], assume_unique=True)
                assert index_of_best1 not in idxs, "Данный ассерт проверяет, что вы для выбора b и c вы не будете использовать выбранный, чтобы не повторятmся"
                assert index_of_best1 != population_index, "Данный ассерт проверяет, что вы не взяли индекс нынешнего индивида"
                assert index_of_best1 != best_idx, "Данный ассерт проверяет, что вы не взяли индекс самого лучшего индивида"
                if iteration == 0:
                    for idx in idxs: assert np.array_equal(population[index_of_best1], population[
                        idx]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                    assert np.array_equal(population[index_of_best1], population[
                        population_index]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                    assert np.array_equal(population[index_of_best1], population[
                        best_idx]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                b, c = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(population[index_of_best1] + mutation_coefficient * (b - c), 0, 1)
            elif mutation_setting == 'rand_to_p_best1':
                p = np.random.uniform(p_min, p_max)  # не удалять
                num_top_individuals = int(p * len(idxs))
                againind = np.setdiff1d(np.argsort(fitness), [population_index, best_idx],
                                        assume_unique=True)[:num_top_individuals]
                index_of_rand_to_p_best1 = np.random.choice(againind, replace=False)
                idxs = np.setdiff1d(idxs, [index_of_rand_to_p_best1], assume_unique=True)

                assert 'a' not in locals()
                assert index_of_rand_to_p_best1 not in idxs, "Данный ассерт проверяет, что вы для выбора b и c вы не будете использовать выбранный, чтобы не повторятmся"
                assert index_of_rand_to_p_best1 != population_index, "Данный ассерт проверяет, что вы не взяли индекс нынешнего индивида"
                assert index_of_rand_to_p_best1 != best_idx, "Данный ассерт проверяет, что вы не взяли индекс самого лучшего индивида"
                if iteration == 0:
                    for idx in idxs: assert np.array_equal(population[index_of_rand_to_p_best1], population[
                        idx]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                    assert np.array_equal(population[index_of_rand_to_p_best1], population[
                        population_index]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                    assert np.array_equal(population[index_of_rand_to_p_best1], population[
                        best_idx]) is False, "Данный ассерт проверяет правильность выбранного индекса"
                b, c = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(population[index_of_rand_to_p_best1] + mutation_coefficient * (b - c), 0, 1)
            else:
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_coefficient * (b - c), 0, 1)

            # Оператор кроссовера
            cross_points = np.random.rand(dimensions) < crossover_coefficient
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # Рекомбинация (замена мутантными значениями)
            trial = np.where(cross_points, mutant, population[population_index])
            trial_denorm = min_bound + trial * diff
            # Оценка потомка
            result_of_evolution = fobj(trial_denorm)
            # Селекция
            if selection_setting == 'worst':
                selection_index = np.argmax(fitness)
            elif selection_setting == 'random_among_worst':
                worse_indices = np.array(np.where(result_of_evolution< fitness))[0]
                if len(worse_indices)!= 0:
                    selection_index = np.random.choice(worse_indices, replace = False)
                else:
                    selection_index = population_index
            elif selection_setting == 'random_selection':
                selection_index = np.random.choice(idxs)

            else:
                selection_index = population_index
            if result_of_evolution < fitness[selection_index]:
                fitness[selection_index] = result_of_evolution
                population[selection_index] = trial
                if result_of_evolution < fitness[best_idx]:
                    best_idx = selection_index
                    best = trial_denorm
        yield best, fitness[best_idx]
