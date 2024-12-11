import matplotlib.pyplot as plt
import numpy as np

max_num_series = 7

default_graphic_tips = {'function': 'b-', 'legend_colors': ['r', 'g', 'brown', 'orange', 'b', 'b', 'b', 'b'],
                        'legend_markers': ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
                        'legend_sizes': [1, 1, 1, 1, 1, 1, 1, 1]}


def plot_problem(prob,  tips=default_graphic_tips, npoints=1000, legend=2):
    """
       Plots a problem
       
       Parameters
       ----------
       prob : UniVarProblem
           The problem to plot
       npoints : int
           number of points used to draw a plot
       """

    #     colors = ['r-', 'b-', 'g-', 'y-', 'm-', 'c-']
    step = (prob.b - prob.a) / npoints
    ta = np.arange(prob.a, prob.b + step, step)
    num_points = len(ta)
    fta = np.empty([num_points])

    for i in range(num_points):
        fta[i] = prob.objective(ta[i])
    lb = np.amin(fta)
    ub = np.amax(fta)
    d = (ub - lb) * 0.1
    plt.plot(ta, fta, tips['function'])
    plt.ylim([lb - d - legend, ub + d])
#     plt.show()


def plot_points(x, y, num_series, tips=default_graphic_tips):
    if num_series > max_num_series:
        raise Exception('num_series is too large')
    plt.scatter(x, y, c=tips['legend_colors'][num_series], linewidths=1, marker=tips['legend_markers'][num_series],
                s=tips['legend_sizes'][num_series])
#     plt.scatter(x, y, c = tips['legend_colors'][num_series], marker = tips['legend_markers'][num_series], s = 2)
#     plt.scatter(x, y, c = tips['legend_colors'][num_series], marker = '+', s = 5)
#     plt.scatter(x, y, c = 'r', marker=legend_markers[num_ser, linewidths = 1, s = 10)
