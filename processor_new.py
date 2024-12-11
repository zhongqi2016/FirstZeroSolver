import math

from interval import Interval
import psqe_bounds as psqe
import psl_bounds as psl
import sys
import copy

sys.path.append("..")
import interval as ival


class ProcData:
    """
    The subproblem data used in algorithm
    """

    def __init__(self, sub_interval: ival.Interval, lip: ival.Interval, counter: int):
        """
        The constructor
        Args:
            sub_interval: the subproblem's interval
            split_point: the point to split interval
        """
        self.sub_interval = sub_interval
        self.lip = lip
        self.counter = counter


class ProcessorNew:

    def __init__(self, rec_v, rec_x, problem, eps, global_lipint=False, use_symm_lipint=False, estimator=2,
                 reduction=1, period_comp_lip=0):
        """
        Initializes processor
        Args:
            rec_v: record value
            rec_x: record point
            problem: problem to solve
            eps: tolerance
            global_lipint: if True use global Lipschitz constant computed for the whole interval
            use_symm_lipint: if True use [-L,L], where L = max(|a|,|b|)
            estimator:estimator==1 -> Algorithm Piyavksii, estimator==2 -> PSQE
        """
        self.res_list = []
        self.use_symm_lipint = use_symm_lipint
        self.global_lipint = global_lipint
        self.rec_v = rec_v
        self.rec_x = rec_x
        self.problem = problem
        self.eps = eps
        self.ddi = problem.ddf(ival.Interval([problem.a, problem.b]))
        self.di = problem.df(ival.Interval([problem.a, problem.b]))
        self.estimator = estimator
        self.reduction = reduction
        self.running = True
        self.period_comp_lip = period_comp_lip

    def update_lipschitz(self, data: ProcData):
        """
        Args:
            data: data of sub_interval
        """
        if self.estimator == 1:
            data.lip = self.problem.df(data.sub_interval)
            if self.use_symm_lipint:
                L = max(-data.lip.x[0], data.lip.x[1])
                data.lip = ival.Interval([-L, L])
        else:
            data.lip = self.problem.ddf(data.sub_interval)
            if self.use_symm_lipint:
                L = max(-data.lip.x[0], data.lip.x[1])
                data.lip = ival.Interval([-L, L])

    def compute_bounds(self, data: ProcData, under: bool):
        a = data.sub_interval.x[0]
        b = data.sub_interval.x[1]
        if self.estimator == 1:
            return psl.PSL_Bounds(a, b, data.lip.x[0], data.lip.x[1], self.problem.objective(a),
                                  self.problem.objective(b), under)
        else:
            return psqe.PSQE_Bounds(a=a, b=b, alp=data.lip.x[0], bet=data.lip.x[1], fa=self.problem.objective(a),
                                    fb=self.problem.objective(b),
                                    dfa=self.problem.df(a), dfb=self.problem.df(b), under=under)

    def fzcp_process(self, data: ProcData):
        """
        Process of branching
        :param data: the selected subinterval data
        :return: subintervals
        """
        sub_interval = data.sub_interval
        lst = []
        obj = self.problem.objective
        if sub_interval.x[1] > self.rec_x:
            return lst
        if self.rec_x - sub_interval.x[0] <= self.eps:
            self.res_list.append((Interval([sub_interval.x[0], self.rec_x]), 'certain'))
            self.running = False
            return lst
        width_of_interval = sub_interval.x[1] - sub_interval.x[0]

        if not self.global_lipint:
            if data.counter < self.period_comp_lip:
                data.counter += 1
            else:
                self.update_lipschitz(data)
                self.period_comp_lip = int(128 / math.sqrt(data.lip.x[1] - data.lip.x[0]))
                data.counter = 0
        lower_estimator = self.compute_bounds(data, under=True)
        left_end = lower_estimator.get_left_end()

        if left_end is not None:
            if self.reduction > 0:
                if self.reduction == 1:
                    if sub_interval.x[1] < self.rec_x:
                        right_end = lower_estimator.get_right_end_under_bound()
                    elif sub_interval.x[1] == self.rec_x:
                        upper_estimator = self.compute_bounds(data, under=False)
                        right_end = upper_estimator.get_right_end_upper_bound()
                        if right_end > sub_interval.x[1]:
                            right_end = sub_interval.x[1]
                        self.rec_x = right_end
                        if right_end - left_end < self.eps:
                            self.res_list.append((Interval([left_end, right_end]), 'certain'))
                            self.running = False
                            return lst
                else:
                    right_end = None
            elif self.reduction == 0:
                left_end = sub_interval.x[0]
                right_end = sub_interval.x[1]
            split_point = left_end + (right_end - left_end) / 2
            if right_end - left_end < self.eps:
                self.res_list.append((Interval([left_end, right_end]), 'uncertain'))
            else:
                new_width = right_end - left_end
                if new_width / width_of_interval > 0.7:
                    sub_1 = ival.Interval([left_end, split_point])
                    if obj(sub_1.x[1]) <= 0:
                        self.rec_x = sub_1.x[1]
                    else:
                        data2 = ProcData(sub_interval=ival.Interval([split_point, right_end]),
                                         lip=copy.deepcopy(data.lip), counter=data.counter)
                        lst.append(data2)

                    data.sub_interval = sub_1
                    lst.append(data)
                else:
                    data.sub_interval.x[0] = left_end
                    data.sub_interval.x[1] = right_end
                    lst.append(data)
        return lst
