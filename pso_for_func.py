import numpy as np
from functions import (rosenbrock_func, easom_func, eggholder_func,
                       beale_func, rastrigin_func, sphere_func, bohachevsky_func, ackley_func)
import datetime
import openpyxl
import pandas as pd
from tabulate import tabulate


'''
time
number of iterations
euclidean_distance
'''


def time_of_function(function):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = function(*args, **kwargs)
        used_time = datetime.datetime.now() - start_time
        # print('Used time:', used_time)
        # return result, str(used_time)
        return result, used_time.total_seconds()
    return wrapper


class Particle:
    def __init__(self, dimension, interval):
        self.position = np.random.uniform(interval[0], interval[1], dimension)
        # self.position = np.random.rand(dimension)
        self.velocity = 0.1 * np.random.rand(dimension)
        self.best_fitness = float('inf')
        self.best_position = self.position


@time_of_function
def pso(func, dimension, number_particles, max_iter, c1, c2, w_min, w_max, interval):
    number_of_calling_func = 0

    particles = [Particle(dimension, interval) for _ in range(number_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for i in range(max_iter):
        for particle in particles:
            number_of_calling_func += 1
            if func(*particle.position) < particle.best_fitness:
                particle.best_fitness = func(*particle.position)
                number_of_calling_func += 1
                particle.best_position = particle.position

            if particle.best_fitness < global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position

        for particle in particles:
            w = w_max - ((w_max - w_min) / max_iter) * i
            r1, r2 = np.random.rand(dimension), np.random.rand(dimension)
            particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + \
                                c2 * r2 * (global_best_position - particle.position)
            # particle.position = particle.position + particle.velocity
            new_position = particle.position + particle.velocity
            particle.position = np.clip(new_position, interval[0], interval[1])

    return [global_best_position, global_best_fitness, number_of_calling_func]


num_particles = 40  # size of population
DIM = 2
iterations = 300
c1 = 1.5  # cognitive parameter
c2 = 1.5  # social parameter

w_min = 0.4
w_max = 0.9


interval = [-512, 512]
minimum_position = np.array([0, 0])
objective_func = eggholder_func


results = []
for i in range(5):
    (position, fitness, calls), exec_time = pso(
        objective_func, DIM, num_particles, iterations, c1, c2, w_min, w_max, interval
    )

    results.append({
        "№ запуску": i+1,
        "Мінімум функ.": round(fitness, 6),
        "Час виконання (с)": round(exec_time, 3),
        "К-сть викликів функ.": calls
    })

# Формування таблиці
df = pd.DataFrame(results)
df.loc["Середнє"] = df.mean(numeric_only=True)
df.loc["Мінімум"] = df.min(numeric_only=True)
df.loc["Максимум"] = df.max(numeric_only=True)

name = objective_func.__name__.upper()
print(tabulate(df, headers='keys', tablefmt='latex'))

df.to_excel(f"func_test/pso_{name}_results.xlsx", index=False)
print(df)

