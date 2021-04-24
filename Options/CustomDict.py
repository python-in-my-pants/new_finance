class CustomDict(dict):

    def __getitem__(self, item):
        for i, index in enumerate(self.keys()):
            if index >= item:
                return list(self.values())[i]
        raise KeyError(f'No value greater or equal to {item} found in OrderedMap')
