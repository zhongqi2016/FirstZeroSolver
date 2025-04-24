import interval as ival
import uvarprob as uvpr
import numpy as np
import psqe_bounds as pq
import psl_bounds as psl
import matplotlib.pyplot as plt
from processor_new import Estimator


def update_lipschitz(a, b, estimator: Estimator, sym: bool, df, ddf):
    """
    Args:
        a: left end of current interval
        b: right end of current interval
        estimator: type of estimator(PSL or PSQE)
        sym: symmetry of Lipschitz interval
        df: derivative of objective function
        ddf: second derivative of objective function
    """
    current_interval = ival.Interval([a, b])
    if estimator == Estimator.PSL:
        lip = df(current_interval)
    else:
        lip = ddf(current_interval)
    lip1, lip2 = lip.x[0], lip.x[1]
    if sym:
        L = max(-lip1, lip2)
        lip1, lip2 = -L, L
    return lip1, lip2


def draw_rect(ax, x1, x2, y1, y2, color):
    if color == 'none':
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none', linewidth=0.5)
    else:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color=color, alpha=0.3)
    ax.add_patch(rect)


def plot_proc(estimator: Estimator, sym: bool, problem: uvpr.UniVarProblem, num_iterations: int, reduction: bool,
              eps: float, global_lip=False):
    """
    plot the process of branching and reduction
    yellow rectangles: eliminated by reduction
    blue rectangles: eliminated by under bound
    green rectangles: eliminated by record point
    :param estimator: type of estimator(PSL or PSQE)
    :param sym: symmetry of Lipschitz interval
    :param problem: uni-variate problem to solve
    :param num_iterations: maximal number of iterations to solve the problem
    :param reduction: reduction or not
    :param eps: tolerance
    :param global_lip: use the global Lipschitz approach or not
    """
    step = (problem.b - problem.a) / 1000.
    ta = np.arange(problem.a, problem.b, step)
    num_points = len(ta)
    lip1, lip2 = update_lipschitz(problem.a, problem.b, estimator, sym, problem.df, problem.ddf)
    work_list = [(problem.a, problem.b, 0, 0)]
    fig, ax = plt.subplots()
    i = 0
    record_x = problem.b + 1
    while len(work_list) != 0:
        if i >= num_iterations:
            break
        x1, x2, lb, ub = work_list.pop()
        ole = x1
        ore = x2
        if x2 - x1 < eps:
            continue
        if x1 >= record_x:
            draw_rect(ax, ole, ore, lb, ub, 'green')
            draw_rect(ax, ole, ore, lb, ub, 'none')
            continue
        if not global_lip:
            lip1, lip2 = update_lipschitz(x1, x2, estimator, sym, problem.df, problem.ddf)
        if estimator == Estimator.PSL:
            under_estimator = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), True)
            over_estimator = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), False)
        else:
            under_estimator = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2),
                                             problem.df(x1), problem.df(x2), True)
            over_estimator = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2),
                                            problem.df(x1), problem.df(x2), False)
        x1 = under_estimator.get_left_end()
        check_upper_bound = False
        if x1:
            if reduction:
                if problem.objective(x2) < 0:
                    check_upper_bound = True
                    x2 = over_estimator.get_right_end_upper_bound()
                else:
                    x2 = under_estimator.get_right_end_under_bound()

        f1 = under_estimator.estimator
        f2 = over_estimator.nestimator

        ub = problem.objective(ole)
        lb = ub
        ft_under = np.empty(num_points)
        ft_over = np.empty(num_points)
        for j in range(num_points):
            if ole <= ta[j] <= ore:
                ft_under[j] = f1(ta[j]) - 0.05
                lb = min(lb, ft_under[j])
                if check_upper_bound:
                    ft_over[j] = f2(ta[j]) + 0.05
                    ub = max(ub, ft_over[j], problem.objective(ta[j]))
                else:
                    ft_over[j] = None
                    ub = max(ub, problem.objective(ta[j]))
            else:
                ft_under[j] = None
                ft_over[j] = None
        ax.plot(ta, ft_under, 'b-')
        ax.plot(ta, ft_over, 'g-')
        if x1:
            if reduction:
                if (x2 - x1) / (ore - ole) < 0.7:
                    work_list.append((x1, x2, lb, ub))
                else:
                    mid = x1 + (x2 - x1) / 2
                    if problem.objective(mid) > 0:
                        work_list.append((mid, x2, lb, ub))
                    else:
                        record_x = mid
                        draw_rect(ax, mid, x2, lb, ub, 'green')
                        draw_rect(ax, mid, x2, lb, ub, 'none')
                    work_list.append((x1, mid, lb, ub))
                draw_rect(ax, ole, x1, lb, ub, 'yellow')
                draw_rect(ax, x2, ore, lb, ub, 'yellow')
                draw_rect(ax, x1, x2, lb, ub, 'none')
                draw_rect(ax, ole, ore, lb, ub, 'none')
            else:
                mid = ole + (ore - ole) / 2
                if problem.objective(mid) > 0:
                    work_list.append((mid, ore, lb, ub))
                else:
                    record_x = mid
                    draw_rect(ax, mid, ore, lb, ub, 'green')
                    draw_rect(ax, mid, ore, lb, ub, 'none')
                work_list.append((ole, mid, lb, ub))
        else:
            draw_rect(ax, ole, ore, lb, ub, 'blue')
            draw_rect(ax, ole, ore, lb, ub, 'none')
        i += 1

    fta = np.empty(num_points)
    for j in range(num_points):
        fta[j] = problem.objective(ta[j])
    ax.plot(ta, fta, 'r-')

    ax.axhline(y=0, linestyle='--', color='black')
    # plt.savefig('./process_reduction.png', dpi=500)
    plt.show()


def plot_proc_by_step(estimator: Estimator, sym: bool, problem: uvpr.UniVarProblem, num_iterations: int,
                      reduction: bool, eps: float, global_lip=False, linewidth=1):
    """
    plot the process of branching and reduction
    yellow rectangles: eliminated by reduction
    blue rectangles: eliminated by under bound
    green rectangles: eliminated by record point
    :param estimator: type of estimator(PSL or PSQE)
    :param sym: symmetry of Lipschitz interval
    :param problem: uni-variate problem to solve
    :param num_iterations: maximal number of iterations to solve the problem
    :param reduction: reduction or not
    :param eps: tolerance
    :param global_lip: use the global Lipschitz approach or not
    """
    step = (problem.b - problem.a) / 1000.
    ta = np.arange(problem.a, problem.b, step)
    num_points = len(ta)
    lip1, lip2 = update_lipschitz(problem.a, problem.b, estimator, sym, problem.df, problem.ddf)
    work_list = [(problem.a, problem.b, 0, 0)]
    # fig, ax = plt.subplots()
    cols = 2
    fig, axes = plt.subplots((num_iterations + cols - 1) // cols, cols, figsize=(6, 6))
    fta = np.empty(num_points)
    for j in range(num_points):
        fta[j] = problem.objective(ta[j])

    i = 0
    record_x = problem.b + 1
    rect_list = []
    lb_edge = 0
    ub_edge = 0
    while len(work_list) != 0:
        if i >= num_iterations:
            break
        x1, x2, lb, ub = work_list.pop()
        ole = x1
        ore = x2
        if x2 - x1 < eps:
            continue

        ax = axes[i // cols, i % cols]
        ax.plot(ta, fta, 'r-', linewidth=linewidth)
        ax.axhline(y=0, linestyle='--', color='black', linewidth=linewidth)
        if x1 >= record_x:
            rect_list.append((ole, ore, lb, ub, 'green'))
            rect_list.append((ole, ore, lb, ub, 'none'))
            continue
        if not global_lip:
            lip1, lip2 = update_lipschitz(x1, x2, estimator, sym, problem.df, problem.ddf)
        if estimator == Estimator.PSL:
            under_estimator = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), True)
            over_estimator = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), False)
        else:
            under_estimator = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2),
                                             problem.df(x1), problem.df(x2), True)
            over_estimator = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2),
                                            problem.df(x1), problem.df(x2), False)
        x1 = under_estimator.get_left_end()
        check_upper_bound = False
        if x1:
            if reduction:
                if problem.objective(x2) < 0:
                    check_upper_bound = True
                    x2 = over_estimator.get_right_end_upper_bound()
                else:
                    x2 = under_estimator.get_right_end_under_bound()

        f1 = under_estimator.estimator
        f2 = over_estimator.nestimator

        ub = problem.objective(ole)
        lb = ub
        ft_under = np.empty(num_points)
        ft_over = np.empty(num_points)
        for j in range(num_points):
            if ole <= ta[j] <= ore:
                ft_under[j] = f1(ta[j]) - 0.05
                lb = min(lb, ft_under[j])
                if check_upper_bound:
                    ft_over[j] = f2(ta[j]) + 0.05
                    ub = max(ub, ft_over[j], problem.objective(ta[j]))
                else:
                    ft_over[j] = None
                    ub = max(ub, problem.objective(ta[j]))
            else:
                ft_under[j] = None
                ft_over[j] = None
        lb_edge = min(lb_edge, lb)
        ub_edge = max(ub_edge, ub)
        ax.plot(ta, ft_under, 'b-', linewidth=linewidth)
        ax.plot(ta, ft_over, 'g-', linewidth=linewidth)
        for down, up, left, right in work_list:
            draw_rect(ax, down, up, left, right, 'none')
        if x1:
            if reduction:
                if (x2 - x1) / (ore - ole) < 0.7:
                    work_list.append((x1, x2, lb, ub))
                else:
                    mid = x1 + (x2 - x1) / 2
                    if problem.objective(mid) > 0:
                        work_list.append((mid, x2, lb, ub))
                    else:
                        record_x = mid
                        rect_list.append((mid, x2, lb, ub, 'green'))
                        rect_list.append((mid, x2, lb, ub, 'none'))
                    work_list.append((x1, mid, lb, ub))
                rect_list.append((ole, x1, lb, ub, 'yellow'))
                rect_list.append((x2, ore, lb, ub, 'yellow'))
                rect_list.append((ole, x1, lb, ub, 'none'))
                rect_list.append((x2, ore, lb, ub, 'none'))
                # rect_list.append((x1, x2, lb, ub, 'none'))
                # rect_list.append((ole, ore, lb, ub, 'none'))
            else:
                mid = ole + (ore - ole) / 2
                if problem.objective(mid) > 0:
                    work_list.append((mid, ore, lb, ub))
                else:
                    record_x = mid
                    rect_list.append((mid, ore, lb, ub, 'green'))
                    rect_list.append((mid, ore, lb, ub, 'none'))
                work_list.append((ole, mid, lb, ub))
        else:
            rect_list.append((ole, ore, lb, ub, 'blue'))
            rect_list.append((ole, ore, lb, ub, 'none'))
        draw_rect(ax, ole, ore, lb, ub, 'none')
        for down, up, left, right, color in rect_list:
            draw_rect(ax, down, up, left, right, color)
        i += 1

    d = 1
    for i in range(num_iterations):
        ax = axes[i // cols, i % cols]
        ax.set_ylim(lb_edge - d, ub_edge + d)
    # plt.savefig('./process.png', dpi=500)
    # plt.show()
    return plt
