import numpy as np


def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        # print(f"Function '{func.__name__}' has been called {wrapper.calls} times.")
        return func(*args, **kwargs), wrapper.calls

    wrapper.calls = 0
    return wrapper


def sphere_func(x, y):
    """
    f∗ = 0
    x∗ = (0, 0)
    interval = [-inf, inf]
    """
    return x ** 2 + y ** 2


# Rosenbrock’s function
def rosenbrock_func(x, y):
    """
    f∗ = 0
    x∗ = (1, 1)
    interval = [-5, 5]
    """
    result = (x - 1) ** 2 + 100 * (y - x ** 2) ** 2
    return result


def easom_func(x, y):
    """
    f∗ = -1
    x∗ = (3.14, 3.14)
    interval = [-100, 100]
    """
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


def eggholder_func(x, y):
    """
    f∗ = -959.6407
    x∗ = (512, 404.2319)
    interval = [-512, 512]
    """
    return (-(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47))))
            - x * np.sin(np.sqrt(np.abs(x - (y + 47)))))


def beale_func(x, y):
    """
    f∗ = 0
    x∗ = (3, 0.5)
    interval = [-4.5, 4.5]
    """
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def rastrigin_func(x, y):
    """
    f∗ = 0
    x∗ = (0, 0)
    interval = [-5.12, 5.12]
    """
    return 10 * 2 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


# Bohachevsky Function
def bohachevsky_func(x, y):
    """
    f∗ = 0
    x∗ = (0, 0)
    interval = [-100, 100]
    """
    return x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7


def ackley_func(x, y):
    """
    f∗ = 0
    x∗ = (0, 0)
    interval = [-5, 5]
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = (x**2 + y**2) / 2
    cos_comp = (np.cos(c * x) + np.cos(c * y)) / 2
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(cos_comp) + a + np.e