import sys
import time

sys.path.append("..")
import pandas as pd
import solv_fzcp as sfzcp
import uvarprob as uvpr
test_file = "../tst3.csv"

# Maximal number of steps
max_steps = 1e6
# Tolerance
epsilon = 1e-2
# If True - global Lipschitz constant is used
global_lipschitz_interval = True
# If True - the record value is taken from the test database
known_record = False
# How many points to skip in vizualization - regular step
skip = 1000
# The size of the legend in plots
legend_size = 2


def log_point(x, points_list):
    points_list.append(x)


def read_problems(fname):
    data = pd.read_csv(fname, index_col='name', comment='#')
    return data


points_db = {}
psl_lipint_points_list = []
psl_lip_points_list = []
psqe_lipint_points_list = []
psqe_lip_points_list = []
bnb2_lipint_points_list = []
bnb2_lip_points_list = []


def fun():
    print('test.Index,PC_N,PI_N,QC_N,QI_N,PC_R,PI_R,QC_R,QI_R')
    eps = 1e-5
    for test in df.itertuples():
        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']))
        T1 = time.perf_counter()
        PC_N = sfzcp.new_method(prob, symm=True, epsilon=eps, global_lipschitz_interval=True, estimator=1,
                                reduction=True).nsteps
        T2 = time.perf_counter()
        time_PC_N = T2 - T1
        print(PC_N, time_PC_N)
        # T1 = time.perf_counter()
        # PI_N = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=True, estimator=1,
        #                         reduction=False).nsteps
        # T2 = time.perf_counter()
        # time_PI_N = T2 - T1
        # T1 = time.perf_counter()
        # QC_N = sfzcp.new_method(prob, symm=True, epsilon=eps, global_lipschitz_interval=True, estimator=2,
        #                         reduction=False).nsteps
        # T2 = time.perf_counter()
        # time_QC_N = T2 - T1
        # T1 = time.perf_counter()
        # QI_N = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=True, estimator=2,
        #                         reduction=False).nsteps
        # T2 = time.perf_counter()
        # time_QI_N = T2 - T1
        #
        # T1 = time.perf_counter()
        # PC_R = sfzcp.new_method(prob, symm=True, epsilon=eps, global_lipschitz_interval=True, estimator=1,
        #                         reduction=True).nsteps
        # T2 = time.perf_counter()
        # time_PC_R = T2 - T1
        # T1 = time.perf_counter()
        # PI_R = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=True, estimator=1,
        #                         reduction=True).nsteps
        # T2 = time.perf_counter()
        # time_PI_R = T2 - T1
        # T1 = time.perf_counter()
        # QC_R = sfzcp.new_method(prob, symm=True, epsilon=eps, global_lipschitz_interval=True, estimator=2,
        #                         reduction=True).nsteps
        # T2 = time.perf_counter()
        # time_QC_R = T2 - T1
        # T1 = time.perf_counter()
        # QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=True, estimator=2,
        #                         reduction=True).nsteps
        # T2 = time.perf_counter()
        # time_QI_R = T2 - T1

        # print('%s & %d & %d & %d & %d & %d & %d & %d & %d \\\\' % (
        #     test.Index, PC_N, PI_N, QC_N, QI_N, PC_R, PI_R, QC_R, QI_R))
        # print('%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (
        #     test.Index, time_PC_N, time_PI_N, time_QC_N, time_QI_N, time_PC_R, time_PI_R, time_QC_R, time_QI_R))


if __name__ == "__main__":
    df = read_problems(test_file)
    fun()
