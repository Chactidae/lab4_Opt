import numpy as np
# тестовые и учебные функции





def test_func(x):
    return x[0] ** 2 + x[1] ** 2


def task_1(x):
    return (1 / 3) * ((x[0] + 1) ** 3) + x[1]


def task_2(x):
    return x[0] ** 2 + x[1] ** 2


def task_3(x):
    return (x[0] - 4) ** 2 + (x[1] - 4) ** 2


def task_4(x):
    return -x[0] + 2 * x[1] ** 2 - 4 * x[1]


def task_5(x):
    return 4 * x[0] ** 2 + 8 * x[0] - x[1] - 3


def task_6(x):
    return (4 / x[0]) + (9 / x[1]) + x[0] + x[1]


def task_7(x):
    return 4 * x[0] ** 2 + 4 * x[0] + x[1] ** 2 - 8 * x[1] + 5


def task_8(x):
    return (8 * x[0] ** 2) - 4 * x[0] + x[1] ** 2 - 12 * x[1] + 7


def task_9(x):
    return (x[0] + 4) ** 2 + (x[1] - 4) ** 2


def task_10(x):
    return np.log10(x[0]) - x[1]


def task_11(x):
    return 3 * x[0] ** 2 + 4 * x[0] * x[1] + 5 * x[1] ** 2


def task_12(x):
    return (-x[0] ** 2) - x[1] ** 2


def task_13(x):
    return (-x[0]) * x[1] * x[2]


def task_14(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def task_15(x):
    return (x[0] - 1) ** 4 + (x[1] - 3) ** 2


def task_16(x):
    return x[0] ** 2 + x[1] ** 2


def task_17(x):
    return x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]

# ограничения для задач


#-----------------------------------------------------------------
# ОГРАНИЧЕНИЯ ДЛЯ ТЕСТОВЫХ ЗАДАЧ

def rest_test1():
    return [2,
           [lambda x: x[0] + x[1] - 2], []]


def rest_test2():
    return [2,
           [lambda x: x[0] - 1], [lambda x: x[0] + x[1] - 2]]


def rest_test3():
    return [2,
           [], [lambda x: 1 - x[0], lambda x: x[0] + x[1] - 2]]


#-----------------------------------------------------------------

def rest_1():
    return [2,
            [], [lambda x: 1 - x[0], lambda x: -x[1]]]


def rest_2():
    return [2,
            [], [lambda x: x[0] ** 2 - x[1], lambda x: -x[0]]]

def rest_3():
    return[2,
           [],
           [lambda x: x[0]+x[1]-5]]

def rest_4():
    return [2,
           [lambda x: -3*x[0]-2*x[1]-6], []]

def rest_5():
    return [2,
            [lambda x: x[0] + x[1] + 2], []]

#2.6
def rest_6_a():
    return[2,
           [],
           [lambda x: x[0]+x[1]-6,
            lambda x: -x[0],
            lambda x: -x[1]]]


def rest_6_b():
    return[2,
           [],
           [lambda x: x[0]+x[1]-4,
            lambda x: -x[0],
            lambda x: -x[1]]]


def rest_7():
    return[2,
           [lambda x: 2*x[0]-x[1]-6], []]


def rest_8():
    return[2,
           [lambda x: 2*x[0]+3*x[1]+6],[]]

def rest_9():
    return [2,
            [],
            [lambda x: 2 * x[0] - x[1] - 2, lambda x: -x[0], lambda x: -x[1]]]

def rest_10():
    return[2,
           [lambda x: x[0]**2+x[1]**2-4],
           [lambda x: 1-x[0]]]


def rest_11():
    return[2,
           [],
           [lambda x: 4-x[0]-x[1],
            lambda x: -x[0],
            lambda x: -x[1]]]


def rest_12():
    return[2,
           [],
           [lambda x: x[0]+2*x[1]-3,
            lambda x: -x[0],
            lambda x: -x[1]]]


def rest_13():
    return [3,
            [],
            [lambda x: -x[0], lambda x: x[0] - 42, lambda x: -x[1], lambda x: x[1] - 42,
             lambda x: -x[2], lambda x: x[2] - 42, lambda x: x[0] + 2 * x[1] + 2 * x[2] - 72]]

def rest_14():
    return[3,
           [],
           [lambda x: 3-x[0]-x[1]-x[2],
            lambda x: 3-x[0]*x[1]*x[2],
            lambda x: -x[0],lambda x: -x[1],lambda x: -x[2]]]


def rest_15():
    return[2,
           [],
           [lambda x: 3*(x[0]**2)+2*(x[1]**2)-21,
            lambda x: 4*x[0]+5*x[1]-20,lambda x: -x[0],lambda x: -x[1]]]


def rest_16():
    return[2,
          [],
          [lambda x: x[0]**2-x[1]**2-1,lambda x: 2-x[0]]]


def rest_17():
    return[4,
           [],
           [lambda x: -8+x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[0]-x[1]+x[2]-x[3],
            lambda x: -10+x[0]**2 + 2*(x[1]**2)+x[2]**2+2*(x[3]**2)-x[0]+x[3],
            lambda x: -5+2*(x[0]**2)+x[1]**2+x[2]**2+2*x[0]-x[1]-x[3]
            ]]




test_f = {
    1: test_func,
    2: test_func,
    3: test_func,
}

rest_test = {
    1: rest_test1(),
    2: rest_test2(),
    3: rest_test3(),
}

task = {
    0: test_func,
    1: task_1,
    2: task_2,
    3: task_3,
    4: task_4,
    5: task_5,
    6: task_6,
    7: task_7,
    8: task_8,
    9: task_9,
    10: task_10,
    11: task_11,
    12: task_12,
    13: task_13,
    14: task_14,
    15: task_15,
    16: task_16,
    17: task_17,
}

rest_6_choose = {
    "a": rest_6_a,
    "b": rest_6_b
}


def choose_rest6(n):
    return rest_6_choose[n]

rest = {
    1: rest_1(),
    2: rest_2(),
    3: rest_3(),
    4: rest_4(),
    5: rest_5(),
    7: rest_7(),
    8: rest_8(),
    9: rest_9(),
    10: rest_10(),
    11: rest_11(),
    12: rest_12(),
    13: rest_13(),
    14: rest_14(),
    15: rest_15(),
    16: rest_16(),
    17: rest_17(),
}

rest6 = {
    "a": rest_6_a(),
    "b": rest_6_b()
}