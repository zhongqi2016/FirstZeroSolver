import math
import numpy as np


class Interval:

    def __init__(self, x):
        self.x = x.copy()

    def __repr__(self):
        return "[" + str(round(self.x[0], 3)) + ", " + str(round(self.x[1], 3)) + "]"

    def __round__(self, n=3):
        return Interval([np.round(self.x[0], 3), np.round(self.x[1], 3)])

    def include_zero(self) -> bool:
        if (self.x[0] > 0 and self.x[1] > 0) or (self.x[0] < 0 and self.x[1] < 0):
            return False
        else:
            return True

    def mid(self):
        return 0.5 * (self.x[0] + self.x[1])

    def width(self):
        return self.x[1] - self.x[0]

    def scale(self, factor):
        m = 0.5 * (self.x[0] + self.x[1])
        r = 0.5 * (self.x[1] - self.x[0])
        self.x[0] = m - factor * r
        self.x[1] = m + factor * r

    def isIn(self, other):
        return (self.x[0] >= other.x[0]) and (self.x[1] <= other.x[1])

    def isNoIntersec(self, other):
        return (self.x[0] > other.x[1]) or (self.x[1] < other.x[0])

    def intersec(self, other):
        if self.x[0] > self.x[1]:
            raise ValueError(other.x[0], other.x[1], "results in wrong bounds:", self.x[0], self.x[1])
        return Interval([max(self.x[0], other.x[0]), min(self.x[1], other.x[1])])

    def __getitem__(self, item):
        return self.x[item]

    def __setitem__(self, key, value):
        self.x.__setitem__(key, value)

    def __neg__(self):
        ninterval = Interval(self.x)
        ninterval.x[0] = - self.x[1]
        ninterval.x[1] = - self.x[0]
        return ninterval

    def __add__(self, other):
        ointerval = valueToInterval(other)
        ninterval = Interval(self.x)
        # print(self, other)
        ninterval.x[0] = self.x[0] + ointerval.x[0]
        ninterval.x[1] = self.x[1] + ointerval.x[1]
        return ninterval

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        ointerval = valueToInterval(other)
        ninterval = Interval(self.x)
        ninterval.x[0] = self.x[0] - ointerval.x[1]
        ninterval.x[1] = self.x[1] - ointerval.x[0]
        return ninterval

    def __rsub__(self, other):
        ointerval = valueToInterval(other)
        return ointerval.__sub__(self)

    def __pow__(self, other):
        ninterval = Interval(self.x)
        u = self.x[0] ** other
        v = self.x[1] ** other
        if other == 0:
            ninterval.x[0] = 1
            ninterval.x[1] = 1
        elif other % 2 == 0:
            ninterval.x[1] = max(u, v)
            if self.x[0] <= 0 and self.x[1] >= 0:
                ninterval.x[0] = 0
            else:
                ninterval.x[0] = min(u, v)
        else:
            ninterval.x[0] = u
            ninterval.x[1] = v
        return ninterval

    def __mul__(self, other):
        ointerval = valueToInterval(other)
        v = [self.x[0] * ointerval.x[0], self.x[0] * ointerval.x[1], self.x[1] * ointerval.x[0],
             self.x[1] * ointerval.x[1]]
        b = [min(v), max(v)]
        return Interval(b)

    def __truediv__(self, other):
        ointerval = valueToInterval(other)
        # v = [self.x[0] / ointerval.x[0], self.x[0] / ointerval.x[1], self.x[1] / ointerval.x[0],
        #      self.x[1] / ointerval.x[1]]
        # f = [min(v), max(v)]
        # return Interval(f)
        if not (valueToInterval(0).isIn(ointerval)):
            v = [self.x[0] / ointerval.x[0], self.x[0] / ointerval.x[1], self.x[1] / ointerval.x[0],
                 self.x[1] / ointerval.x[1]]
            f = [min(v), max(v)]
            return Interval(f)
        else:
            # print(self, ointerval)
            c = []
            a1, a2 = self.x[0], self.x[1]
            b1, b2 = ointerval.x[0], ointerval.x[1]
            if b2 == 0:
                c = self * Interval([-np.inf, 1 / b1])
            elif b1 == 0:
                c = self * Interval([1 / b2, np.inf])
            elif b1 < 0 and b2 > 0:
                # c = [self*Interval([-np.inf, 1 / b1]), self*Interval([1 / b2, np.inf])]
                c = Interval([-np.inf, np.inf])
            # if b2 == 0 and a2 <=0:
            #     c = [a2/b1, np.inf]
            # elif a2 < 0 and b1 < 0 and b2 > 0:
            #     c = [[-np.inf, a2/b2], [a2/b1, np.inf]]
            # elif a2 <= 0 and b1 == 0:
            #     c = [-np.inf, a2/b2]
            # elif a1 >= 0 and b2 == 0:
            #     c = [-np.inf, a1/b1]
            # elif a1 > 0 and b1 < 0 and b2 > 0:
            #     c = [[-np.inf, a1/b1], [a1/b2, np.inf]]
            # elif a1 >= 0 and b1 == 0:
            #     c = [a1/b2, np.inf]
            # elif a1 < 0 and a2 > 0:
            #     c = [-np.inf, np.inf]
            # print(c)
            # print(type(c))
            return c

    def __floordiv__(self, other):
        ointerval = valueToInterval(other)
        v = [self.x[0] // ointerval.x[0], self.x[0] // ointerval.x[1], self.x[1] // ointerval.x[0],
             self.x[1] // ointerval.x[1]]
        b = [min(v), max(v)]
        return Interval(b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        ointerval = valueToInterval(other)
        return ointerval.__truediv__(self)

    def __abs__(self):
        if self.x[1] < 0:
            return Interval([-self.x[1], -self.x[0]])
        elif self.x[0] < 0:
            if -self.x[0] > self.x[1]:
                return Interval([self.x[1], -self.x[0]])
            else:
                return Interval([-self.x[0], self.x[1]])
        else:
            return Interval([self.x[0], self.x[1]])


def sin(x_input):
    if isinstance(x_input, Interval):
        return _sin(x_input.x)
    else:
        return math.sin(x_input)


def cos(x_input):
    if isinstance(x_input, Interval):
        return _cos(x_input.x)
    else:
        return math.cos(x_input)


def exp(x_input):
    if isinstance(x_input, Interval):
        return _exp(x_input.x)
    else:
        return math.exp(x_input)


def abs(x_input):
    if isinstance(x_input, Interval):
        return _abs(x_input.x)
    else:
        if x_input < 0:
            return -x_input
        else:
            return x_input


def log(x_input):
    if isinstance(x_input, Interval):
        return _ln(x_input.x)
    else:
        return math.log(x_input)


def sqrt(x_input):
    if isinstance(x_input, Interval):
        return _sqrt(x_input.x)
    else:
        return math.sqrt(x_input)


def valueToInterval(expr):
    if not isinstance(expr, Interval):
        etmp = Interval([expr, expr])
        return etmp
    else:
        return expr
    # if isinstance(expr, int):
    #     etmp = Interval([expr, expr])
    # elif isinstance(expr, float):
    #     etmp = Interval([expr, expr])
    # else:
    #     etmp = expr
    # return etmp


def _sin(x):
    # if isinstance(x, (int, np.integer)):
    #     return math.sin(x)
    # elif isinstance(x, (float, np.float)):
    #     return math.sin(x)
    # else:
    y = [math.sin(x[0]), math.sin(x[1])]
    pi2 = 2 * math.pi
    pi05 = math.pi / 2
    if math.ceil((x[0] - pi05) / pi2) <= math.floor((x[1] - pi05) / pi2):
        b = 1
    else:
        b = max(y)

    if math.ceil((x[0] + pi05) / pi2) <= math.floor((x[1] + pi05) / pi2):
        a = -1
    else:
        a = min(y)
    return Interval([a, b])


def _cos(x):
    # if isinstance(x, (int, np.integer)):
    #     return math.cos(x)
    # elif isinstance(x, (float, np.float)):
    #     return math.cos(x)
    # else:
    y = [math.cos(x[0]), math.cos(x[1])]
    pi2 = 2 * math.pi
    if math.ceil(x[0] / pi2) <= math.floor(x[1] / pi2):
        b = 1
    else:
        b = max(y)
    if math.ceil((x[0] - math.pi) / pi2) <= math.floor((x[1] - math.pi) / pi2):
        a = -1
    else:
        a = min(y)
    return Interval([a, b])


def _exp(x):
    return Interval([math.exp(x[0]), math.exp(x[1])])


def _abs(x):
    if x[1] < 0:
        return Interval([-x[1], -x[0]])
    elif x[0] < 0:
        if -x[0] > x[1]:
            return Interval([x[1], -x[0]])
        else:
            return Interval([-x[0], x[1]])
    else:
        return Interval([x[0], x[1]])


def _log(x, base):
    if base > 1:
        return Interval([math.log(x[0], base), math.log(x[1], base)])
    else:
        return Interval([math.log(x[1], base), math.log(x[0], base)])


def _sqrt(x):
    return Interval([math.sqrt(x[0]), math.sqrt(x[1])])


def _ln(x):
    return Interval([math.log(x[0]), math.log(x[1])])
