import time
from numpy import median
import sys


def interactive():

    print("Interactive mode on! Type 'exit' to return to normal mode.")
    while True:
        try:
            i = input(">>> ")
            if i != "exit":
                sys.stdout.write(">>> ")
                exec(i)
            else:
                return
        except Exception as e:
            print(e)


def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        print("\n{} finished in {}".format(func.__name__, seconds_to_timestamp(time.time()-start)))
        return retval
    return inner


def median_old(inseq):
    seq = sorted(inseq)
    if len(seq) < 3:
        return seq[0]
    return seq[int(len(seq)/2)]  # +1 for next element, -1 for index


def my_median(seq):
    return median(seq)


def weigted_median_index(weights):

    w_sum = sum(weights)

    s = 0
    for i in range(len(weights)-1):

        if s + weights[i+1] > w_sum/2 or i == len(weights)-2:
            return i
        else:
            s += weights[i]

    return 0


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
