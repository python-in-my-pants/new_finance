
def cross_up(a, b):

    """
    TODO still wrong,must have been lower sometime and just recently crossed up
    """

    """
    did "a" cross over "b" (so was lower, now is higher)?
    :param a:
    :param b:
    :return: boolean indicating wether a cross happened
             index of the cross
    """
    if type(a) is not list:
        if type(b) is not list:
            return False, None
        a = [a for _ in range(len(b))]

    if type(b) is not list:
        if type(a) is not list:
            return False, None
        b = [b for _ in range(len(a))]

    return a[-2] <= b[-2] and a[-1] > b[-1]


def cross_down(a, b):
    return cross_up(b, a)


def cross(a, b):
    """
    :param a: signal a, can be skalar or list
    :param b: signal b, can be skalar or list
    :return: boolean indicating if cross occured;
             the last index where a[index] == b[index] or a[index] == b for scalar b and vice versa
    """
    up = cross_up(a, b)
    down = cross_down(a, b)

    if up or down:
        return True
    else:
        return False, None


def rising(a, window=2):
    """

    :param a:
    :param window: rise is given in respect to the last n values of a, where n = window
    :return: true if rising, false otherwise;
             slope of a in respect to window
    """
    if a.instanceof(list):
        is_rising = a[-window] < a[-1]
        return is_rising,
    else:
        return False, None


def falling(a, window=2):
    """

    :param a:
    :param window: fall is given in respect to the last n values of a, where n = window
    :return: true if falling, false otherwise;
             slope of a if falling
    """
    tmp = rising(a, window)
    return not tmp[0], tmp[1]

