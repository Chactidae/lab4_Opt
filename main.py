import math
import scipy
import scipy.optimize as optimize
import numpy as np
from typing import Callable, List

import testData
from testData import *


def grad(func, xcur, eps) -> np.array:
    return optimize.approx_fprime(xcur, func, eps ** 2)


def koshi(func: Callable[..., float], x0: List[float], eps = 0.0001):
    x = np.array(x0)
    t = 0
    gr = grad(func, x, eps)
    a = 0.
    flag = False
    while any([abs(gr[i]) > eps for i in range(len(gr))]):
        t = t + 1
        gr = grad(func, x, eps)
        a = optimize.minimize_scalar(lambda koef: func(*[x + koef * gr])).x
        x += a * gr

        for i in range(len(gr)):
            if abs(gr[i]) <= eps:
                flag = True
    print(str(x))
    print(str(t))
    return x


# CONDITIONAL METHODS OF MANY-DIMENSIONAL OPTIMIZATION
def penalty_method(x0, objective_func, alpha_start, beta, eps, rest_eq, rest_not_eq):
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x):
        H = 0
        for i in rest_eq:
            H += pow(abs(i(x)), 2)
        for i in rest_not_eq:
            H += pow(max(0, i(x)), 2)
        return objective_func(x) + alpha * H
    x_cur = np.array(x0)
    x_new = koshi(lambda x: getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x), x_cur, eps)
    while ((x_cur - x_new)**2).sum() > eps:
        alpha *= beta
        x_cur = x_new
        x_new = koshi(lambda x: getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x), x_cur, eps)
        if alpha > 100000:
            break
    print("alpha in the end = ", alpha)
    return x_new


def barrier_method(x0, objective_func, alpha_start, beta, eps, rest_not_eq):
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_not_eq, x):
        H = sum(1 / (0.000000001 + max(0, -i(x))**2) for i in rest_not_eq)
        return objective_func(x) + alpha * H
    x_cur = np.array(x0)
    x_new = None
    atLeastOnePointFound = False
    while not (atLeastOnePointFound and (((x_cur - x_new) ** 2).sum() < eps ** 2)):
        x_temp = koshi(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                                 rest_not_eq, x), x_cur, eps)
        isInside = not any(neq(x_temp) > eps for neq in rest_not_eq)
        if (isInside):
            if not atLeastOnePointFound:
                atLeastOnePointFound = True
            else:
                x_cur = x_new
            x_new = x_temp

        alpha *= beta
        if alpha < 0.00001:
            break
    print("alpha in the end = ", alpha)
    return x_new


print("""Choose task:
1. 
2.
3.
4.
5.
6.
7.
8.
9.
10.
11.
12.
13.
14.
15.
16.
17.
""")

print("Выбрать тестовые или учебные задачи[1/2]: ")

choose_part = int(input())

function_number = int(input("Выберите функцию: "))
if 0 <= function_number <= 17:
    if choose_part == 2 and 1 <= function_number <= 17:
        function = task[function_number]
        dimensions_num, restrictions_of_equality, restrictions_of_non_equality = rest[function_number]
    elif function_number <= 3 and choose_part == 1:
        function = test_f[function_number]
        dimensions_num, restrictions_of_equality, restrictions_of_non_equality = rest_test[function_number]

    if len(restrictions_of_equality) == 0:
        print('Methods:\n1. Penalty\n2. Barrier')
        method = int(input("Выберите метод: "))
    else:
        method = 1

    # INPUT penalty parameter, Beta, Eps
    alpha_start = 1
    beta = 2
    eps = 0.001
    print("alpha_start = 1, beta = 2, nu = 1/2, eps = 0.0001")
    start_point = [0]*dimensions_num
    for count in range(0, dimensions_num):
        start_point[count] = float(input("x0: "))

    result = []
    if method == 1:
        result = penalty_method(start_point, function, alpha_start, beta, eps,
                                restrictions_of_equality, restrictions_of_non_equality)
    else:
        result = barrier_method(start_point, function, alpha_start, 1 / beta, eps, restrictions_of_non_equality)

    print("x*: ", result, "f(x*) = ", function(result))



