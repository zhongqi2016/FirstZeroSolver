import interval as ival
import uvarprob as uvpr
import numpy as np
import psqe_bounds as pq
import psl_bounds as psl
import matplotlib.pyplot as plt


def update_lipschitz(a, b, estimator: int, sym: bool, df, ddf):
    """
    Args:
        data: data of sub_interval
    """
    current_interval = ival.Interval([a, b])
    if estimator == 1:
        lip = df(current_interval)
    else:
        lip = ddf(current_interval)
    lip1, lip2 = lip.x[0], lip.x[1]
    if sym:
        L = max(-lip1, lip2)
        lip1, lip2 = -L, L
    return lip1, lip2


def draw_rect(ax, x1, x2, y1, y2, color):
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color=color, alpha=0.3)
    ax.add_patch(rect)


def draw_edge(ax, x1, x2, y1, y2):
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none', linewidth=1)
    ax.add_patch(rect)


def plot_proc(estimator: int, sym: bool, problem: uvpr.UniVarProblem, num_iterations: int, reduction: bool, eps,
              global_lip=False):
    step = (problem.b - problem.a) / 1000.
    ta = np.arange(problem.a, problem.b, step)
    num_points = len(ta)
    le = problem.a
    re = problem.b
    lip1, lip2 = update_lipschitz(le, re, estimator, sym, problem.df, problem.ddf)
    work_list = []
    work_list.append((problem.a, problem.b, 0, 0))
    fig, ax = plt.subplots()
    i = 0
    record_x = re + 1
    while len(work_list) != 0:
        if i >= num_iterations - 1:
            break
        x1, x2, lb, ub = work_list.pop()
        ole = x1
        ore = x2
        if x2 - x1 < eps:
            continue
        if x1 >= record_x:
            draw_rect(ax, ole, ore, lb, ub, 'green')
            draw_edge(ax, ole, ore, lb, ub)
            continue
        if not global_lip:
            lip1, lip2 = update_lipschitz(x1, x2, estimator, sym, problem.df, problem.ddf)
        if estimator == 1:
            estim_int = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), True)
            estim_int_ob = psl.PSL_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), False)
        else:
            estim_int = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2), problem.df(x1),
                                       problem.df(x2), True)
            estim_int_ob = pq.PSQE_Bounds(x1, x2, lip1, lip2, problem.objective(x1), problem.objective(x2),
                                          problem.df(x1), problem.df(x2), False)
        x1 = estim_int.get_left_end()
        check_upper_bound = False
        if x1:
            if reduction:
                if problem.objective(x2) < 0:
                    check_upper_bound = True
                    x2 = estim_int_ob.get_right_end_upper_bound()
                else:
                    x2 = estim_int.get_right_end_under_bound()

        f1 = estim_int.estimator
        f2 = estim_int_ob.nestimator

        ub = problem.objective(ole)
        lb = ub
        fta1 = np.empty(num_points)
        fta2 = np.empty(num_points)
        for j in range(num_points):
            if ole <= ta[j] <= ore:
                fta1[j] = f1(ta[j]) - 0.05
                lb = min(lb, fta1[j])
                if check_upper_bound:
                    fta2[j] = f2(ta[j]) + 0.05
                    ub = max(ub, fta2[j], problem.objective(ta[j]))
                else:
                    fta2[j] = None
                    ub = max(ub, problem.objective(ta[j]))
            else:
                fta1[j] = None
                fta2[j] = None
        ax.plot(ta, fta1, 'b-')
        ax.plot(ta, fta2, 'g-')
        if x1:
            if (x2 - x1) / (ore - ole) < 0.7:
                work_list.append((x1, x2, lb, ub))
            else:
                mid = x1 + (x2 - x1) / 2
                if problem.objective(mid) > 0:
                    work_list.append((mid, x2, lb, ub))
                else:
                    record_x = mid
                    draw_rect(ax, mid, x2, lb, ub, 'green')
                    draw_edge(ax, mid, x2, lb, ub)
                work_list.append((x1, mid, lb, ub))
            draw_rect(ax, ole, x1, lb, ub, 'yellow')
            draw_rect(ax, x2, ore, lb, ub, 'yellow')
            draw_edge(ax, x1, x2, lb, ub)
            draw_edge(ax, ole, ore, lb, ub)
        else:
            draw_rect(ax, ole, ore, lb, ub, 'blue')
            draw_edge(ax, ole, ore, lb, ub)
        i += 1

    fta = np.empty(num_points)
    for j in range(num_points):
        fta[j] = problem.objective(ta[j])
    ax.plot(ta, fta, 'r-')

    ax.axhline(y=0, linestyle='--', color='black')
    # plt.savefig('./process_reduction.png', dpi=500)
    plt.show()
