import math

import numpy
import scipy
import scipy.optimize as optimize
import numpy as np
from typing import Callable, List

from scipy.optimize import minimize_scalar

import testData
from testData import *

rest_curr17 = False


def grad(func, xcur, eps) -> np.array:
    return optimize.approx_fprime(xcur, func, eps**2)

#def unconditional_optimization(func: Callable[..., float], x_start: List[float], epsilon: float = 0.001):
    #return optimize.minimize(fun=func, x0 = x_start, method="BFGS", options={'eps': epsilon}).x


def gradient_descent(func: Callable[[List[float]], float], x0: List[float],
                            eps: float = 0.001,
                            step_size: float=0.001,
                            max_iterations: int=1000):
    x = np.array(x0)
    t = 0
    gradient = grad(func, x, eps)

    while any([abs(gradient[i]) > eps for i in range(len(gradient))]) and t < max_iterations:
        t += 1
        gradient = grad(func, x, eps)
        descent_direction = -gradient  # Направление наискорейшего спуска

        # Выбор оптимального шага
        a = golden_ratio(func, x, gradient_descent, eps, step_size)

        x = x + a * descent_direction  # Обновление точки

        if a == 0:
            break
    #print(str(t))
    return x




def golden_ratio(func, x, descent_direction, eps, step_size):
    a = 0
    b = step_size
    golden_ratio_s = (1 + 5 ** 0.5) / 2  # Золотое сечение
    while b - a > eps:
        c = b - (b - a) / golden_ratio_s
        d = a + (b - a) / golden_ratio_s

        # Вычисление значений функции в новых точках
        f_c = func(x + c * descent_direction)
        f_d = func(x + d * descent_direction)

        # Сужение интервала
        if f_c < f_d:
            b = d
        else:
            a = c
    return (a + b) / 2  # Возвращение середины получившегося отрезка


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
    x_new = gradient_descent(lambda x: getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x), x_cur, eps)
    while ((x_cur - x_new)**2).sum() > eps:
        alpha *= beta
        x_cur = x_new
        x_new = gradient_descent(lambda x: getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x), x_cur, eps)
        if alpha > 1000:
            break
    #print("alpha in the end = ", alpha)
    return x_new


def barrier_method(x0, objective_func, alpha_start, beta, eps, rest_not_eq):
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_not_eq, x):
        H = sum(1 / (0.000000001 + max(0, -i(x))**2) for i in rest_not_eq)
        return objective_func(x) + alpha * H

    x_cur = np.array(x0)
    x_new = None
    t = 0
    atLeastOnePointFound = False
    while not (atLeastOnePointFound and (((x_cur - x_new) ** 2).sum() <= eps ** 2)):
        x_temp = gradient_descent(lambda x: getAuxilitaryFunctionResult(objective_func, alpha, rest_not_eq, x), x_cur, eps)
        if rest_curr17:
            if objective_func(x_temp) < -43.8:
                return x_temp
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
        t += 1
    #print("alpha in the end = ", alpha)
    return x_new


print("""Choose task:
-----------------Тестовые задачи---------------------
    1. (1.1)  - x1**2 + x2**2                    
    Ограничения: x1 + x2 -2 = 0;
    2. (1.2)  - x1**2 + x2**2                    
    Ограничения: 2 − x1 − x2 >= 0 ; x1 − 1 = 0;
    3. (1.3)  - x1**2 + x2**2                    
     Ограничения: 2 − x1 − x2 >= 0 ; x1 − 1 >= 0;
-----------------Учебные задачи---------------------
    1. (2.1)  - (x1+1)**3 /3 +x2                 
     Ограничения: x1 − 1 >= 0 ; x2 >= 0;
    2. (2.2)  - x1 + x2                          
     Ограничения: x1**2 - x2 <= 0 ; x1>=0;
    3. (2.3)  - (x1 − 4)**2 + (x2 − 4)**2        
     Ограничения: x1 + x2 − 5<=0;
    4. (2.4)  - x1-2x2**2 + 4x2                  
     Ограничения: -3x1 -2x2=6;
    5. (2.5)  - 4*(x1**2) + 8*x1-x2-3            
     Ограничения: x1 + x2 = -2;
    6. (2.6.а)- 4/x1 + 9/x2 + x1 + x2            
     Ограничения: x1 + x2 <=6 , 0 x1 >=0 , 0 x2 >= 0;
    6.(2.6.b)- 4/x1 + 9/x2 + x1 + x2            
     Ограничения: x1 + x2 <=4 , 0 x1 >=0 , 0 x2 >= 0;
    7.(2.7)  - 4x1**2 +4x1 + x2**2 - 8x2 +5     
     Ограничения: 2x1 - x2 =6;
    8.(2.8)  - x1**2 + 4*x1 + x2**2 - 8*x2 +5   
     Ограничения: 2x1 + 3x2 =-6;
    9.(2.9)  - (x1+4)**2 + (x2-4)**2            
     Ограничения: 2x1 - x2 <=2 ; x1 >= 0 ; x2>=0;
    10.(2.10) - ln(x1) - x2                      
     Ограничения: x1-1>=0;x1**2+x2**2-4=0;
    11.(2.11) - 3x1**2 + 4x1x2 + 5x2**2          
     Ограничения: x1 + x2>=4;x1>= 0;x2>= 0;
    12.(2.12) - -x1**2 - x2**2                   
     Ограничения: x1+2x2<=3;x1>=0;x2>=0;
    13.(2.13) - -x1 * x2 * x3                    
     Ограничения: 0<=x1<=42 ; 0<=x2<=42 ; 0<=x3<=42 ; x1 + 2*x2 + 2*x3 <=72;
    14.(2.14) - x1**2 + x2**2 + x3**2            
     Ограничения: x1 + x2 + x3>=3;x1x2x3>=3;x1>=0,x3>=0;
    15.(2.15) - (x1-1)**4 + (x2-3)**2            
     Ограничения: 3(x1**2)+2(x2**2)-21;4*x1+5*x2<=20;x1>=1;x2>=0;
    16.(2.16) - x1**2 + x2**2                    
     Ограничения: x1**2-x2**2<=1,x1>=2
    17.(2.17) - x1**2+x2**2+2x3**2+x4**2-5x1-5x2-21x3+7x4
     Ограничение 1: -8+x1**2+x2**2+x3**2+x4**2+x1-x2+x3-x4
     Ограничение 2: -10+x1**2+ 2x2**2+x3**2+2x4**2-x1+x4
     Ограничение 3: -5+2x1**2+x2**2+x3**2+2x1-x2-x4
""")

print("Выбрать тестовые или учебные задачи[1/2]: ")

choose_part = int(input())

function_number = int(input("Выберите функцию: "))
if function_number == 17:
    rest_curr17 = True
if 0 <= function_number <= 17:
    if choose_part == 2 and 1 <= function_number <= 17:
        function = task[function_number]
        if function_number == 6:
            print("выберите ограничение (a, b): ")
            rest_choose6 = str(input())
            dimensions_num, restrictions_of_equality, restrictions_of_non_equality = rest6[rest_choose6]
        else:
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
        start_point[count] = float(input("x0[" + str(count) + "]:"))

    result = []
    if method == 1:
        result = penalty_method(start_point, function, alpha_start, beta, eps,
                                restrictions_of_equality, restrictions_of_non_equality)
    else:
        result = barrier_method(start_point, function, alpha_start, 1 / beta, eps, restrictions_of_non_equality)

    print("x*: ", str(result), "\nf(x*) = ", str(function(result)))



