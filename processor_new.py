from os.path import split

import psqe_bounds as psqe
import psl_bounds as psl
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
    Piyavskii = auto()
    Falsi = auto()
    FalsiTan = auto()
    Casado = auto()
    Baumann = auto()
    FalsiLipschitz = auto()


class ProcData:
    """
    The subproblem data used in algorithm
    """

    def __init__(self, sub_interval: ival.Interval, fa: float, fb: float, lip, update_lip=True):
        """
        The constructor
        Args:
            sub_interval: the subproblem's interval
        """
        self.sub_interval = sub_interval
        self.fa = fa
        self.fb = fb
        self.lb = None
        self.ub = None
        self.lip = lip
        self.update_lip = update_lip


class ProcessorNew:

    def __init__(self, rec_v, rec_x, problem, eps, global_lipint=False, use_symm_lipint=False,
                 estimator=Estimator.PSQE, reduction=True, adap_lip=False, div=Division.Bisection, alp=0.7, rho=36):
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
        self.rho = rho
        self.adap_lip = adap_lip
        # if self.estimator == Estimator.PSQE:
        #     self.lip = problem.ddf(ival.Interval([problem.a, problem.b]))
        # elif self.estimator == Estimator.PSL:
        #     self.lip = problem.df(ival.Interval([problem.a, problem.b]))
        # else:
        #     raise ValueError("Unknown estimator")
        # if self.use_symm_lipint:
        #     L = max(-self.lip.x[0], self.lip.x[1])
        #     self.lip = ival.Interval([-L, L])

    def update_lipschitz(self, data: ProcData):
        if self.estimator == Estimator.PSL:
            lip = self.problem.df(data.sub_interval)
            if self.use_symm_lipint:
                L = max(-lip.x[0], lip.x[1])
                lip = ival.Interval([-L, L])
        else:
            lip = self.problem.ddf(data.sub_interval)
            if self.use_symm_lipint:
                L = max(-lip.x[0], lip.x[1])
                lip = ival.Interval([-L, L])
        data.lip = lip

    def compute_bounds(self, data: ProcData, under: bool):
        a = data.sub_interval.x[0]
        b = data.sub_interval.x[1]
        if self.estimator == Estimator.PSL:
            return psl.PSL_Bounds(a, b, data.lip.x[0], data.lip.x[1], data.fa, data.fb, under)
        else:
            return psqe.PSQE_Bounds(a=a, b=b, alp=data.lip.x[0], bet=data.lip.x[1], fa=data.fa, fb=data.fb,
                                    dfa=self.problem.df(a), dfb=self.problem.df(b), under=under)

    def get_split(self, data: ProcData, lam=0.8):
        mid = data.sub_interval.x[0] + (data.sub_interval.x[1] - data.sub_interval.x[0]) / 2
        match self.div:
            case Division.Bisection:
                return mid
            case Division.Falsi:
                if data.fb >= 0:
                    return mid
                return (1 - lam) * mid + lam * (data.sub_interval.x[0] * data.fb - data.sub_interval.x[1] * data.fa) / (
                        data.fb - data.fa)
            # case Division.Piyavskii:
            #
            case _:
                raise ValueError("Unknown division method")

    def get_split_casado(self, data: ProcData, upper_bound: float, lower_bound: float):
        widthX = data.sub_interval.x[1] - data.sub_interval.x[0]
        widthF = upper_bound - lower_bound
        if widthF == 0:
            return data.sub_interval.x[0] + widthX / 2
        ratio = upper_bound / widthF
        if ratio <= 0.33:
            beta = 0.33 * widthX
        elif ratio <= 0.66:
            beta = ratio * widthX
        else:
            if (upper_bound - data.fb) <= (data.fb - lower_bound):
                beta = 0.33 * widthX
            else:
                beta = 0.66 * widthX
        return beta + data.sub_interval.x[0]

    def fzcp_process(self, data: ProcData):
        """
        Process of branching
        :param data: the selected subinterval data
        :return: subintervals
        """
        # sub_interval = data.sub_interval
        x1_interval = data.sub_interval.x[0]
        x2_interval = data.sub_interval.x[1]
        lst = []
        obj = self.problem.objective
        if self.rec_x - x1_interval <= self.eps:
            self.res_list.append((ival.Interval([x1_interval, self.rec_x]), 'certain'))
            self.running = False
            return lst
        width_of_interval = x2_interval - x1_interval
        if not self.global_lipint and data.update_lip == True:
            self.update_lipschitz(data)
        lower_estimator = self.compute_bounds(data, under=True)
        left_end = lower_estimator.get_left_end()
        if left_end is not None:
            assert (left_end >= x1_interval)
            if width_of_interval <= self.eps:
                self.res_list.append((ival.Interval([x1_interval, x2_interval]), 'uncertain'))
                return lst
            if self.reduction:
                if x2_interval < self.rec_x:
                    right_end = lower_estimator.get_right_end_under_bound()
                else:
                    over_estimator = self.compute_bounds(data, under=False)
                    right_end = over_estimator.get_right_end_upper_bound()
                    if right_end >= x2_interval:
                        right_end = x2_interval
                    else:
                        self.rec_x = right_end
            else:
                left_end = x1_interval
                right_end = x2_interval

            if data.lip.x[1] < 0:
                new_interval = x1_interval - self.problem.objective(x1_interval) / data.lip
                new_interval=data.sub_interval.intersect(new_interval)
                # print(right_end-left_end,new_interval.width())

            if self.reduction:
                data.fa = obj(left_end)
                data.fb = obj(right_end)
                data.sub_interval.x[0] = left_end
                data.sub_interval.x[1] = right_end

            data.update_lip = True
            if (right_end - left_end) / width_of_interval > self.alp:
                # split_point = self.get_split(data)
                mid = left_end + (right_end - left_end) / 2
                lam = 0.8
                data.update_lip = True
                match self.div:
                    case Division.Bisection:
                        split_point = mid
                    case Division.Falsi:
                        if data.fb >= 0:
                            split_point = mid
                        else:
                            split_point = (1 - lam) * mid + lam * (
                                    left_end * data.fb - right_end * data.fa) / (
                                                  data.fb - data.fa)
                    case Division.FalsiTan:
                        if data.fb >= 0:
                            split_point = mid
                        else:
                            c_falsi = (left_end * data.fb - right_end * data.fa) / (
                                    data.fb - data.fa)
                            if data.fa < data.fb:
                                dfa = self.problem.df(left_end)
                                if dfa == 0:
                                    c_tan = mid
                                else:
                                    c_tan = left_end - data.fa / dfa
                            else:
                                dfb = self.problem.df(right_end)
                                if dfb == 0:
                                    c_tan = mid
                                else:
                                    c_tan = right_end - data.fb / dfb
                            if abs(c_tan - mid) < abs(c_falsi - mid):
                                split_point = (1 - lam) * mid + lam * c_tan
                            else:
                                split_point = (1 - lam) * mid + lam * c_falsi
                    case Division.Piyavskii:
                        assert self.estimator == Estimator.PSL
                        if self.reduction:
                            lower_estimator = self.compute_bounds(data, under=False)
                        split_point = lower_estimator.c
                    case Division.Casado:
                        if self.reduction:
                            self.compute_bounds(data, under=True)
                        _, lower_bound = lower_estimator.lower_bound_and_point()
                        over_estimator = self.compute_bounds(data, under=False)
                        _, upper_bound = over_estimator.lower_bound_and_point()
                        split_point = self.get_split_casado(data, upper_bound, lower_bound)
                    case Division.Baumann:
                        assert self.estimator == Estimator.PSL
                        if data.lip.x[0] > 0 or data.lip.x[1] < 0:
                            split_point = mid
                        else:
                            split_point = (1 - lam) * mid + lam * (
                                    (data.lip.x[1] * data.sub_interval.x[0] - data.lip.x[0] * data.sub_interval.x[
                                        1]) / (data.lip.x[1] - data.lip.x[0]))
                    case Division.FalsiLipschitz:
                        if data.fb >= 0:
                            split_point = mid
                        else:
                            lam = self.rho / (data.lip.width() + self.rho)
                            split_point = (1 - lam) * mid + lam * (
                                    left_end * data.fb - right_end * data.fa) / (data.fb - data.fa)

                    case _:
                        raise ValueError("Unknown division method")

                f_split = obj(split_point)
                if f_split <= 0:
                    self.rec_x = split_point
                else:
                    data2 = ProcData(sub_interval=ival.Interval([split_point, right_end]), fa=f_split, fb=data.fb,
                                     lip=data.lip)
                    lst.append(data2)
                data.sub_interval.x[1] = split_point
                data.fb = f_split
                lst.append(data)
            else:
                if self.adap_lip:
                    data.update_lip = False
                lst.append(data)
        return lst
