"""
Triggers take indicators and give back true or false,
depending on whether their conditions are fulfilled.
"""


def cross_up(indicator_a, indicator_b):

    """
    did "a" cross over "b" (so was lower, now is higher)?
    :param indicator_a:
    :param indicator_b:
    :return: boolean indicating whether a cross happened
             index of the cross
    """

    def cu(a, b):
        if type(a) is not list:
            if type(b) is not list:
                return False, None
            a = [a for _ in range(len(b))]

        if type(b) is not list:
            if type(a) is not list:
                return False, None
            b = [b for _ in range(len(a))]

        return a[-2] <= b[-2] and a[-1] > b[-1]

    return cu(indicator_a.value_history[-2:], indicator_b.value_history[-2:])


def cross_down(indicator_a, indicator_b):
    return cross_up(indicator_b, indicator_a)


def cross(indicator_a, indicator_b):
    """
    :param indicator_a: signal a, can be skalar or list
    :param indicator_b: signal b, can be skalar or list
    :return: boolean indicating if cross occured;
             the last index where a[index] == b[index] or a[index] == b for scalar b and vice versa
    """
    up = cross_up(indicator_a, indicator_b)
    down = cross_down(indicator_a, indicator_b)

    if up or down:
        return True
    else:
        return False, None


def rising(indicator_a, window=2):
    """
    :param indicator_a:
    :param window: rise is given in respect to the last n values of a, where n = window
    :return: true if rising, false otherwise;
             slope of a in respect to window
    """
    return indicator_a.value_history[-window] < indicator_a.value_history[-1]


def falling(indicator_a, window=2):
    """

    :param indicator_a:
    :param window: fall is given in respect to the last n values of a, where n = window
    :return: true if falling, false otherwise;
             slope of a if falling
    """
    return indicator_a.value_history[-window] > indicator_a.value_history[-1]

