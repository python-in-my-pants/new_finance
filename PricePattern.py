class PricePattern:

    def __init__(self, cause, effect, interpolation=False):
        """
        cause seq is followed by effect seq

        :param cause:  numeric sequence
        :param effect: numeric sequence
        """
        if interpolation:
            print("Interpolation is not supported yet!")
            interpolation = False
        self.interpolation = interpolation
        self.cause = PriceSequence(cause, interpolation=interpolation)
        self.effect = PriceSequence(effect, interpolation=interpolation)
        self.length = self.cause.length + self.effect.length

    def fits(self, prices):

        if self.interpolation:
            # TODO
            #  set splitting point in relation to relation of lengths of cause and effect
            pass
        else:
            if len(prices) != self.length:
                print("Wrong number of prices provided to 'fits' in PricePattern")
                return False, False
            else:
                if self.cause.fits(prices[:self.cause.length]):
                    if self.effect.fits(prices[-self.effect.length:]):
                        return True, True
                    return True, False
                return False, False

    def check_seq(self, seq, significance=0.8, min_occs=10, p=False):

        if len(seq) < self.length:
            raise Exception("Sequence to check is too short!")

        causes = 0
        following_effects = 0

        for i in range(len(seq)-self.length+1):
            cause_found, effect_found = self.fits(seq[i:self.length])
            if cause_found:
                causes += 1
            if effect_found:
                following_effects += 1

        sigificant = (following_effects / causes >= significance if causes else False) and following_effects > min_occs

        if p and sigificant:
            spots = len(seq) - self.length + 1
            perc_effs = 100 * following_effects / causes if causes else 0
            print(f'Checked {str(spots)} spots in price history\n'
                  f'Found pattern {str(following_effects)} times\n'
                  f'which equals {str(perc_effs)}% '
                  f'{">=" if sigificant else "<"} '
                  f'{str(int(significance*100))}%')

        return sigificant, (following_effects/causes if causes else 0), following_effects

    def __str__(self):
        return "Cause: " + str(self.cause) + "\nEffect: " + str(self.effect)


class PriceSequence:

    def __init__(self, data, interpolation=False):
        self.interpolation = interpolation
        self.prices = data
        self.length = len(data)
        # half of the min gradient
        self.tolerance = abs(min([data[i+1]-data[i] for i in range(len(data)-1)])/2)

    def fits(self, other_data) -> bool:

        if self.interpolation:
            if len(other_data) == self.length:
                for i, element in enumerate(self.prices):
                    if not (element - self.tolerance <= other_data[i] <= element + self.tolerance):
                        return False
                return True
            elif len(other_data) < self.length:
                # TODO
                #  interpolate and get #self.length points from other_data
                pass
            else:
                # TODO
                #  interpolate and get #self.length points from other_data
                pass
        else:
            if len(other_data) != self.length:
                return False
            else:
                for i, element in enumerate(self.prices):
                    if not (element - self.tolerance <= other_data[i] <= element + self.tolerance):
                        return False
                return True

    def __str__(self):
        return f'(Length: {self.length}, Prices: {self.prices})'
