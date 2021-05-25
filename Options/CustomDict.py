from bisect import bisect_left


class CustomDict(dict):

    def __getitem__(self, item):

        values = list(self.keys())
        pos = bisect_left(values, item)

        #if pos == 0:
        #    return values[0]

        if pos == 0 or pos >= len(values):
            raise KeyError

        #if pos == len(values):
        #    return values[-1]

        before = values[pos - 1]
        after = values[pos]
        if after - item < item - before:
            return list(self.values())[pos]
        else:
            if item - before > after - before:
                # if item is very big and thus very wide out of range, better not return anything
                raise KeyError
            return list(self.values())[pos-1]
