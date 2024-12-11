"""
Algrithm from Casado, L.G., Garcia, I.F., Sergeyev, Y.D.: Interval branch and bound algorithm for finding
the first-zero-crossing-point in one-dimensional functions. Reliable Computing 6(2), 179–191 (2000)
"""

import sub as sb
import interval as ival


class CasData:
    """
    The subproblem data
    """

    def __init__(self, ival, split_point):
        """
        The constructor
        Args:
            ival: the subproblem's interval
            split_point: the point to split interval
        """
        self.ival = ival
        self.split_point = split_point


class CasProcessor:
    """
    The processor by using algorithm of Casado, L.G., Garcia, I.F., Sergeyev, Y.D.
    """

    def __init__(self, rec_v, rec_x, problem, eps):
        """
        Initializes processor
        Args:
            res_list: list of results
            rec_v: record value
            rec_x: record point
            problem: problem to solve
            eps: tolerance
            running: the process is work or not
        """
        self.res_list = []
        self.rec_v = rec_v
        self.rec_x = rec_x
        self.problem = problem
        self.eps = eps
        self.running = True
        # self.di = problem.df(ival.Interval([problem.a, problem.b]))

    def updateSplitAndBounds(self, sub):
        """
            update the split point and lowerbound and upperbound of the subinterval
            :param sub: current subinterval
        """
        bounds = self.problem.objective(sub.data.ival)
        sub.bound[1] = bounds.x[1]
        sub.bound[0] = bounds.x[0]
        widthX = sub.data.ival[1] - sub.data.ival[0]
        widthF = sub.bound[1] - sub.bound[0]
        if widthF == 0:
            sub.data.split_point = (sub.data.ival[0] + sub.data.ival[0]) / 2
            return
        ratio = sub.bound[1] / widthF
        if ratio <= 0.33:
            beta = 0.33 * widthX
        elif ratio <= 0.66:
            beta = ratio * widthX
        else:
            obj = self.problem.objective
            fx2 = obj(sub.data.ival[1])
            if (sub.bound[1] - fx2) <= (fx2 - sub.bound[0]):
                beta = 0.33 * widthX
            else:
                beta = 0.66 * widthX

        sub.data.split_point = beta + sub.data.ival[0]

    def fzcp_process(self, sub):
        """
            Process of branching
        """
        lst = []
        obj = self.problem.objective
        if sub.bound[0] <= 0 <= sub.bound[1] and sub.data.ival[0] < self.rec_x:
            if sub.data.ival.x[1] - sub.data.ival.x[0] < self.eps:
                self.res_list.append(sub.data.split_point)
                self.running = False
                return lst
            else:
                sub_1 = sb.Sub(sub.level + 1, [0, 0],
                               CasData(ival.Interval([sub.data.ival.x[0], sub.data.split_point]), None))
                self.updateSplitAndBounds(sub_1)
                sub_2 = sb.Sub(sub.level + 1, [0, 0],
                               CasData(ival.Interval([sub.data.split_point, sub.data.ival.x[1]]), None))
                self.updateSplitAndBounds(sub_2)
                lst.append(sub_2)
                if obj(sub_1.data.ival[1]) <= 0 and sub_1.data.ival[1] < self.rec_x:
                    self.rec_x = sub_1.data.ival[1]
                lst.append(sub_1)
        return lst
