import psqe_bounds as psqe
import psl_bounds as psl
import copy
from enum import Enum, auto

import interval as ival


class Estimator(Enum):
    """
        LINEAR: Piece-wise Linear Estimator of Piyavksii;
        QUADRATIC: Piece-wise Smoothy Quadratic Estimator
    """
    PSL = 1
    PSQE = 2


class Division(Enum):
    """
    Bisection: Bisection method;
    Falsi: Regula falsi or method of false position
    """
    Bisection = auto()
    Falsi = auto()


class ProcData:
    """
    The subproblem data used in algorithm
    """

    def __init__(self, sub_interval: ival.Interval, fa: float, fb: float):
        """
        The constructor
        Args:
            sub_interval: the subproblem's interval
        """
        self.sub_interval = sub_interval
        self.fa = fa
        self.fb = fb


class ProcessorNew:

    def __init__(self, rec_v, rec_x, problem, eps, global_lipint=False, use_symm_lipint=False,
                 estimator=Estimator.PSQE, reduction=True, div=Division.Bisection, alp=0.7):
        """
        Initializes processor
        Args:
            rec_v: record value
            rec_x: record point
            problem: problem to solve
            eps: tolerance
            global_lipint: if True use global Lipschitz constant computed for the whole interval
            use_symm_lipint: if True use [-L,L], where L = max(|a|,|b|)
            estimator:estimator==LINEAR -> Algorithm Piyavksii, estimator==QUADRATIC -> Piece-wise Smoothy Quadratic Estimator
        """
        self.res_list = []
        self.use_symm_lipint = use_symm_lipint
        self.global_lipint = global_lipint
        self.rec_v = rec_v
        self.rec_x = rec_x
        self.problem = problem
        self.eps = eps
        self.estimator = estimator
        self.reduction = reduction
        self.running = True
        self.div = div
        self.alp = alp
        if self.estimator == Estimator.PSQE:
            self.lip = problem.ddf(ival.Interval([problem.a, problem.b]))
        elif self.estimator == Estimator.PSL:
            self.lip = problem.df(ival.Interval([problem.a, problem.b]))
        else:
            raise ValueError("Unknown estimator")
        if self.use_symm_lipint:
            L = max(-self.lip.x[0], self.lip.x[1])
            self.lip = ival.Interval([-L, L])

    def update_lipschitz(self, sub_interval: ival.Interval):
        if self.estimator == Estimator.PSL:
            lip = self.problem.df(sub_interval)
            if self.use_symm_lipint:
                L = max(-lip.x[0], lip.x[1])
                lip = ival.Interval([-L, L])
        else:
            lip = self.problem.ddf(sub_interval)
            if self.use_symm_lipint:
                L = max(-lip.x[0], lip.x[1])
                lip = ival.Interval([-L, L])
        self.lip = lip

    def compute_bounds(self, data: ProcData, under: bool):
        a = data.sub_interval.x[0]
        b = data.sub_interval.x[1]
        if self.estimator == Estimator.PSL:
            return psl.PSL_Bounds(a, b, self.lip.x[0], self.lip.x[1], data.fa, data.fb, under)
        else:
            return psqe.PSQE_Bounds(a=a, b=b, alp=self.lip.x[0], bet=self.lip.x[1], fa=data.fa, fb=data.fb,
                                    dfa=self.problem.df(a), dfb=self.problem.df(b), under=under)

    def get_split(self, data: ProcData, lam=0.9):
        mid = data.sub_interval.x[0] + (data.sub_interval.x[1] - data.sub_interval.x[0]) / 2
        if self.div == Division.Bisection or data.fb >= 0:
            return mid
        elif self.div == Division.Falsi:
            return (1 - lam) * mid + lam * (data.sub_interval.x[0] * data.fb - data.sub_interval.x[1] * data.fa) / (
                        data.fb - data.fa)
        else:
            raise ValueError("Unknown division method")

    def fzcp_process(self, data: ProcData):
        """
        Process of branching
        :param data: the selected subinterval data
        :return: subintervals
        """
        sub_interval = data.sub_interval
        lst = []
        obj = self.problem.objective
        if self.rec_x - sub_interval.x[0] <= self.eps:
            self.res_list.append((ival.Interval([sub_interval.x[0], self.rec_x]), 'certain'))
            self.running = False
            return lst
        width_of_interval = sub_interval.x[1] - sub_interval.x[0]
        if not self.global_lipint:
            self.update_lipschitz(sub_interval)
        lower_estimator = self.compute_bounds(data, under=True)
        left_end = lower_estimator.get_left_end()

        if left_end is not None:
            assert (left_end >= sub_interval.x[0])
            if width_of_interval <= self.eps:
                self.res_list.append((ival.Interval([sub_interval.x[0], sub_interval.x[1]]), 'uncertain'))
                return lst
            if self.reduction:
                if sub_interval.x[1] < self.rec_x:
                    right_end = lower_estimator.get_right_end_under_bound()
                else:
                    upper_estimator = self.compute_bounds(data, under=False)
                    right_end = upper_estimator.get_right_end_upper_bound()
                    if right_end >= sub_interval.x[1]:
                        right_end = sub_interval.x[1]
                    else:
                        self.rec_x = right_end
            else:
                left_end = sub_interval.x[0]
                right_end = sub_interval.x[1]

            if self.reduction:
                data.fa = obj(left_end)
                data.fb = obj(right_end)
                data.sub_interval.x[0] = left_end
                data.sub_interval.x[1] = right_end
            if (right_end - left_end) / width_of_interval > self.alp:
                split_point = self.get_split(data)
                f_split = obj(split_point)
                if f_split <= 0:
                    self.rec_x = split_point
                else:
                    data2 = ProcData(sub_interval=ival.Interval([split_point, right_end]), fa=f_split, fb=data.fb)
                    lst.append(data2)
                data.sub_interval.x[1] = split_point
                data.fb = f_split
                lst.append(data)
            else:
                lst.append(data)
        return lst
