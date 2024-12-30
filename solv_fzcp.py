import math
import sys
# from sortedcontainers import SortedKeyList
from collections import namedtuple
import processor_new as newproc
import processor_Casado as casproc

import interval as ival
import bnb as bnb
import sub as sub

TestResult = namedtuple('TestResult', ['nsteps', 'first_crossing_zero_point'])


def get_initial_recval(prob, known_record):
    if known_record:
        return prob.min_f
    else:
        return float('inf')


def cas(prob, max_steps=sys.maxsize, epsilon=1e-2, known_record=False):
    epsilon = epsilon
    psp = casproc.CasProcessor(rec_v=get_initial_recval(prob, known_record), rec_x=prob.b, problem=prob,
                               eps=epsilon)
    sl = []
    subp = sub.Sub(0, [0, 0], casproc.CasData(ival.Interval([prob.a, prob.b]), 0))
    psp.updateSplitAndBounds(subp)
    sl.append(subp)
    cnt = max_steps
    steps = bnb.bnb_fzcp(sl, cnt, psp)
    return TestResult(nsteps=steps, first_crossing_zero_point=psp.res_list)


def new_method(prob, symm=True, max_steps=sys.maxsize, epsilon=1e-2, global_lipschitz_interval=False,
               known_record=False, estimator=2, reduction=1):
    psp = newproc.ProcessorNew(rec_v=get_initial_recval(prob, known_record), rec_x=prob.b, problem=prob,
                               eps=epsilon, global_lipint=global_lipschitz_interval, use_symm_lipint=symm,
                               estimator=estimator, reduction=reduction)
    sl = []
    interval = ival.Interval([prob.a, prob.b])
    data = newproc.ProcData(sub_interval=interval, lip=ival.Interval([0, 0]))
    psp.update_lipschitz(data)
    sl.append(data)
    cnt = max_steps
    steps = bnb.bnb_fzcp(sl, cnt, psp)
    return TestResult(nsteps=steps, first_crossing_zero_point=psp.res_list)
