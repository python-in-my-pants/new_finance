import time


def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        print("{} finished in {}".format(func.__name__, seconds_to_timestamp(time.time()-start)))
        return retval
    return inner


def flatten(t):
    return [item for sublist in t for item in sublist]


def avg(l):
    return sum(l)/len(l)


def seconds_to_timestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)
