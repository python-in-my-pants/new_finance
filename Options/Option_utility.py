import numpy as np
from datetime import datetime, date, timedelta
import time


def round_cut(x, n=0):
    if x == 0:
        return 0
    return x / abs(x) * int(abs(x) * 10 ** n - 0.5) / 10 ** n


def get_timestamp():
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# <editor-fold desc="Requirements">

"""
def high(x):  # x 0 to 100, high if > 50; returns scale from 0 to 1 indicating highness
    return 0 if x <= 50 else (x / 50 if x > 0 else 0) - 1


def low(x):  # x 0 to 100
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)


def high_ivr(ivr):
    # todo adjust to overall market situation (give result in relation to VIX IV)
    return 0 if ivr <= 50 else (ivr / 50 if ivr > 0 else 0) - 1


def low_ivr(x):  # x 0 to 100
    # todo adjust to overall market situation (give result in relation to VIX IV)
    return 0 if x >= 50 else 1 - (x / 50 if x > 0 else 0)"""


def high_ivr(ivr):
    return (ivr - 50) / 50


def neutral_ivr(ivr):
    return neutral((ivr / 50) - 1)


def low_ivr(ivr):
    return -high_ivr(ivr)


def low_rsi(rsi):
    # todo which one is better?
    """if rsi <= 30:
        return rsi/30 - 1
    if rsi <= 70:
        return rsi
    else:
        return rsi/30 - (7/3)"""
    return -rsi / 50 + 1


def high_rsi(rsi):
    return -low_rsi(rsi)


def neutral(outlook):
    if outlook <= -0.5:
        return outlook
    if outlook <= 0:
        return 4 * outlook + 1
    if outlook <= 0.5:
        return -4 * outlook + 1
    else:
        return -outlook


def bullish(outlook):
    return outlook


def bearish(outlook):
    return -outlook


def extreme(outlook):
    if outlook <= 0:
        return -2 * outlook - 1
    else:
        return 2 * outlook - 1


def calendar_spread_check(front_m_vol, back_m_vol):
    return front_m_vol >= back_m_vol * 1.2


def get_closest_date(expiries, dte):
    """
    :param expiries: list of strings in format year-mon-day
    :param dte: days to expiration
    :return: closest date from expiries to dte
    """
    best_exp = None
    best_diff = 10000
    for expiration in expiries:
        diff = abs(dte - date_str_to_dte(expiration))
        if diff < best_diff:
            best_diff = abs(dte - date_str_to_dte(expiration))
            best_exp = expiration
    return best_exp


def date_str_to_dte(d: str):
    """
    :param d: as string YYYY-MM-DD
    :return:
    """
    return abs((datetime.now().date() - str_to_date(d)).days)


def datetime_to_dte(d: datetime):
    """
    :param d: as string datetime obj
    :return:
    """
    return abs((datetime.now().date() - d.date()).days)


def date_to_dte(d: date):
    """
    :param d: as string datetime obj
    :return:
    """
    return abs((datetime.now().date() - d).days)


def dte_to_date(dte: int):
    return datetime.now() + timedelta(days=dte)


def str_to_date(expiration: str):
    return datetime.strptime(expiration, "%Y-%m-%d").date()


def date_to_european_str(d: str):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y")


def datetime_to_european_str(d: datetime):
    return d.strftime("%d.%m.%Y")


def date_to_opt_format(d: str):
    return str_to_date(d).strftime("%b %d %y")


def opt_date_str_to_date(opt_date_str: str):
    # Jul 02 21
    return datetime.strptime(opt_date_str, "%b %d %y")


def datetime_to_str(d):
    return d.strftime("%Y-%m-%d")


def get_delta_option_strike(chain, delta):  # chain must contain deltas (obviously onii-chan <3)
    return chain.at["strike", abs_min_closest_index(list(chain["delta"]), delta)]


def abs_min_closest_index(a, v):
    """
    :param a: list of values
    :param v: target value
    :return: index of object in list that is closest to target value
    """
    return min(range(len(a)), key=lambda i: abs(abs(a[i]) - v))


def get_atm_strike(chain, _ask):
    return chain["strike"][abs_min_closest_index(list(chain["strike"]), _ask)]


def get_bid_ask_spread(chain, u_ask):
    """

    :param chain:
    :return: returns relative and absolute spread
    """

    # atm_index = chain.options["gamma"].argmax()
    atm_index = abs_min_closest_index(list(chain.options["strike"]), u_ask)

    b = chain.options.loc[atm_index, "bid"]
    a = chain.options.loc[atm_index, "ask"]

    if a == b == 0:
        return 0, 0

    return 1 - b / a, a - b


def get_extrinsic_at_dte(dte_now, extr_now, dte_then):
    # dte now -> extr now

    def decay_func(y):  # TODO add in piecewise functions
        return np.piecewise(y, [], [])

    # decay func(then) * factor
    # = decay func(then) * (extr now / f_dte_now)
    # = decay func(then) * (extr now / decay_func(dte_now)

    return decay_func(dte_then) * extr_now / decay_func(dte_now)


class SnappingCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """

    def __init__(self, ax, line):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.x, self.y = line.get_data()
        self._last_index = None
        # text location in axes coords
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


# </editor-fold>
