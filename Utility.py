import time
from numpy import median
import sys


def ppdict(d, i=0):
    indent = "\t"*i
    s = ""
    longest_key_len = max([len(key) for key in d.keys()])
    for key, val in d.items():
        s += f'{indent}{(longest_key_len-len(key))*" "}{key}:\t{val}\n'
    return s


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


class StopWatch:

    def __init__(self, s):
        self._start_time = time.time()
        self._rounds = [self._start_time]
        print(f"Starting StopWatch {s}")

    def take_time(self, s):
        t = time.time()
        t_from_start = t - self._start_time
        t_from_last = t - self._rounds[-1]
        self._rounds.append(t)
        #print(f'{s} finished in {t_from_last:.8f} ({t_from_start:.8f} since start)')


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
