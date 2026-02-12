import time
import numpy as np
import pandas as pd
import solv_fzcp as sfzcp
import uvarprob as uvpr
import interval as ival
from processor_new import Estimator, Division


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


def getMax(matrix):
    max_values = []
    if isinstance(matrix[0][0], ival.Interval):
        for j in range(len(matrix[0])):
            max_value = matrix[0][j].width()
            for i in range(len(matrix)):
                if isinstance(matrix[i][j], str):
                    continue
                if matrix[i][j].width() > max_value:
                    max_value = matrix[i][j].width()
            max_values.append(max_value)
    else:
        for j in range(len(matrix[0])):
            max_value = matrix[0][j]
            for i in range(len(matrix)):
                if matrix[i][j] > max_value:
                    max_value = matrix[i][j]
            max_values.append(max_value)
    return max_values


def getMaxRatio(matrix):
    max_values = []
    # end = len(matrix[0]) - 1
    end = 0
    # if isinstance(matrix[0][0], ival.Interval):
    #     for j in range(len(matrix[0])):
    #         max_value = 0
    #         for i in range(len(matrix)):
    #             if isinstance(matrix[i][j], str) or isinstance(matrix[i][end], str):
    #                 continue
    #             if matrix[i][j].width() / matrix[i][end].width() > max_value:
    #                 max_value = matrix[i][j].width() / matrix[i][end].width()
    #         max_values.append(max_value)
    # else:
    for j in range(len(matrix[0])):
        max_value = matrix[0][end] / matrix[0][j]
        for i in range(len(matrix)):
            if matrix[i][j] == 0:
                max_value = 0
                break
            if matrix[i][end] / matrix[i][j] > max_value:
                max_value = matrix[i][end] / matrix[i][j]
        max_values.append(max_value)
    return max_values[1:]


def getMin(matrix):
    min_values = []
    if isinstance(matrix[0][0], ival.Interval):
        for j in range(len(matrix[0])):
            min_value = matrix[0][j].width()
            for i in range(len(matrix)):
                if isinstance(matrix[i][j], str):
                    continue
                if matrix[i][j].width() < min_value:
                    min_value = matrix[i][j].width()
            min_values.append(min_value)
    else:
        for j in range(len(matrix[0])):
            min_value = matrix[0][j]
            for i in range(len(matrix)):
                if matrix[i][j] < min_value:
                    min_value = matrix[i][j]
            min_values.append(min_value)
    return min_values


def getMinRatio(matrix):
    min_values = []
    # end = len(matrix[0]) - 1
    end = 0
    # if isinstance(matrix[0][0], ival.Interval):
    #     for j in range(len(matrix[0])):
    #         if isinstance(matrix[0][j], str) or isinstance(matrix[0][end], str):
    #             continue
    #         min_value = matrix[0][j].width() / matrix[0][end].width()
    #         for i in range(len(matrix)):
    #             if isinstance(matrix[i][j], str) or isinstance(matrix[i][end], str):
    #                 continue
    #             if matrix[i][j].width() / matrix[i][end].width() < min_value:
    #                 min_value = matrix[i][j].width() / matrix[i][end].width()
    #         min_values.append(min_value)
    # else:
    for j in range(len(matrix[0])):
        min_value = matrix[0][end] / matrix[0][j]
        for i in range(len(matrix)):
            if matrix[i][j] == 0:
                min_value = 0
                break
            if matrix[i][end] / matrix[i][j] < min_value:
                min_value = matrix[i][end] / matrix[i][j]
        min_values.append(min_value)
    return min_values[1:]


def getAvg(matrix):
    avg_values = []
    for j in range(len(matrix[0])):
        sum_value = 0
        count = 0
        for i in range(len(matrix)):
            if isinstance(matrix[i][j], ival.Interval):
                sum_value += matrix[i][j].width()
            elif isinstance(matrix[i][j], str):
                continue
            else:
                sum_value += matrix[i][j]
            count += 1
        avg_value = sum_value / count
        avg_values.append(avg_value)
    return avg_values


def getNumLessIBB(matrix):
    nums_list = []
    for j in range(len(matrix[0])):

        if j == 0:
            nums_list.append(0)
        else:
            counter = 0
            for i in range(len(matrix)):
                if matrix[i][j] < matrix[i][0]:
                    counter += 1
            nums_list.append(counter)
    return nums_list


def getAvgRatio(matrix):
    avg_values = []
    # end = len(matrix[0]) - 1
    end = 0
    # if isinstance(matrix[0][0], ival.Interval):
    #     for j in range(len(matrix[0])):
    #         sum_value = 0
    #         count = 0
    #         for i in range(len(matrix)):
    #             if isinstance(matrix[i][j], str) or isinstance(matrix[i][end], str):
    #                 continue
    #             count += 1
    #             sum_value += matrix[i][j].width() / matrix[i][end].width()
    #         avg_value = sum_value / count
    #         avg_values.append(avg_value)
    # else:
    for j in range(len(matrix[0])):
        sum_value = 0
        for i in range(len(matrix)):
            if matrix[i][j] == 0:
                sum_value = 0
                break
            sum_value += matrix[i][end] / matrix[i][j]
        avg_value = sum_value / len(matrix)
        avg_values.append(avg_value)
    return avg_values[1:]


def print_row_f3(row, index, bf=True):
    print('%s' % index, end=' ')
    for idx, roxi in enumerate(row):
        if bf and idx > 0 and roxi < row[0]:
            print(r"& \textbf{", end='')
            print('%.3f' % roxi, end='')
            print(r'}', end=' ')
        else:
            print('& %.3f' % roxi, end=' ')
    print('\\\\')


def format_number(x, precision=5, threshold=1e-4):
    if abs(x) < threshold:
        return f"{x:.2e}"
    else:
        return f"{x:.{precision}f}"


def print_row_f5(row, index):
    print('%s' % index, end=' ')
    for idx, roxi in enumerate(row):
        if idx > 0 and roxi < row[0]:
            print(r"& \textbf{", end='')
            print('%s}' % format_number(roxi, 5), end=' ')
        else:
            print('& %s' % format_number(roxi, 5), end=' ')
    print(r'\\')


def print_row_int(row, index):
    print('%s' % index, end=' ')
    for idx, roxi in enumerate(row):
        if idx > 0 and roxi < row[0]:
            print(r"& \textbf{", end='')
            print('& %d' % roxi, end='')
            print(r'}', end=' ')
        else:
            print('& %d' % roxi, end=' ')
    print('\\\\')


def print_simple_row_int(row, index):
    print('%s' % index, end=' ')
    for idx, roxi in enumerate(row):
        print('& %d' % roxi, end=' ')
    print('\\\\')


def print_row_f1(row, index):
    print('%s' % index, end=' ')
    for roxi in row:
        print('& %.1f' % roxi, end=' ')

    print('\\\\')


def print_com_time(head_line, com_time, precision=3):
    if isinstance(head_line, str):
        print(head_line)
    for idx, time_row in enumerate(com_time):
        id1 = idx + 1
        print('%d' % id1, end=' ')
        for idx2, time in enumerate(time_row):
            if idx2 > 0 and time < time_row[0]:
                print(r"& \textbf{", end='')
                print(f"{time:.{precision}f}", end='')
                print(r"}", end=' ')
            else:
                print(f"& {time:.{precision}f}", end=' ')

        print('\\\\')


def print_com_res(head_line, iterations, precision=5):
    if isinstance(head_line, str):
        print(head_line)

    for idx, it_row in enumerate(iterations):
        id1 = idx + 1
        print('%d' % id1, end=' ')
        for idx2, it in enumerate(it_row):
            if isinstance(it, (int, np.integer)) or isinstance(it, (float, np.floating)):
                if idx2 > 0 and it < it_row[0]:
                    print(r"& \textbf{", end='')
                    if isinstance(it, (int, np.integer)):
                        print("{}".format(it), end='')
                    else:
                        print("%s" % format_number(it, precision), end='')
                    print(r"}", end=' ')
                else:
                    if isinstance(it, (int, np.integer)):
                        print("& {}".format(it), end=' ')
                    else:
                        print("& %s" % format_number(it, precision), end=' ')
            else:
                print("& {}".format(it), end=' ')
        print('\\\\')


def test_casado(df, eps, repeat):
    # print('test.Index,PC_N,PI_N,QC_N,QI_N,PC_R,PI_R,QC_R,QI_R')
    it_list = it_list = np.zeros((len(list(df.itertuples())), 3), dtype=int)
    time_list = []

    for test in df.itertuples():
        it_list_row = [None] * 3
        time_list_row = [0.] * 3
        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']), True)
        for num in range(0, repeat):
            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps).nsteps
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[0] = Cas
            time_list_row[0] += time_Cas

            T1 = time.perf_counter()
            QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=False,
                                    estimator=Estimator.PSQE,
                                    reduction=True).nsteps
            T2 = time.perf_counter()
            time_QI_R = (T2 - T1)
            it_list_row[1] = QI_R
            time_list_row[1] += time_QI_R

            T1 = time.perf_counter()
            QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps, global_lipschitz_interval=True,
                                    estimator=Estimator.PSQE,
                                    reduction=True).nsteps
            T2 = time.perf_counter()
            time_QI_R = (T2 - T1)
            it_list_row[2] = QI_R
            time_list_row[2] += time_QI_R

        it_list.append(it_list_row)
        time_list.append(time_list_row)

        # print('%s & %d & %d & %d & %d & %d & %d & %d & %d \\\\' %(test.Index,PC_N,PI_N,QC_N,QI_N,PC_R,PI_R,QC_R,QI_R))
        # print('%s & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (
        # test.Index, time_PC_N, time_PI_N, time_QC_N, time_QI_N, time_PC_R, time_PI_R, time_QC_R, time_QI_R))
    index = 0
    print('Index & IBB & local Lip & global Lip \\')
    for time_row in time_list:
        index = index + 1
        print('%s & %.6f & %.6f & %.6f \\\\' % (index,
                                                time_row[0], time_row[1], time_row[2]))
    min_list = getMinRatio(time_list)
    max_list = getMaxRatio(time_list)
    avg_list = getAvgRatio(time_list)
    print_row_f1(min_list, 'Min')
    print_row_f1(max_list, 'Max')
    print_row_f1(avg_list, 'Average')

    index = 0
    print('Index & IBB & local Lip & global Lip \\')
    for it_row in it_list:
        index = index + 1
        print('%s & %d & %d & %d \\\\' % (index,
                                          it_row[0], it_row[1], it_row[2]))
    min_list = getMinRatio(it_list)
    max_list = getMaxRatio(it_list)
    avg_list = getAvgRatio(it_list)
    print_row_f1(min_list, 'Min')
    print_row_f1(max_list, 'Max')
    print_row_f1(avg_list, 'Average')


def get_full_interval(intervals):
    if len(intervals) == 0:
        return r'No root'
    else:
        sum_width_Q = 0
        if isinstance(intervals[0], tuple):
            for interval in intervals:
                sum_width_Q += interval[0].width()
        else:
            for interval in intervals:
                sum_width_Q += interval.width()
        return sum_width_Q


def test_last(df, eps, repeat, global_lip=True, div=Division.Bisection, alp=0.7):
    # print('test.Index,PC_N,PI_N,QC_N,QI_N,PC_R,PI_R,QC_R,QI_R')
    time_list = np.zeros((len(list(df.itertuples())), 9), dtype=float)
    it_list = np.zeros((len(list(df.itertuples())), 9), dtype=int)
    full_interval_list = np.zeros((len(list(df.itertuples())), 9), dtype=float)
    i = 0
    for test in df.itertuples():
        it_list_row = np.zeros(9, dtype=int)
        time_list_row = np.zeros(9, dtype=float)
        full_interval_row = np.zeros(9, dtype=ival.Interval)

        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']), True)
        for num in range(0, repeat):
            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps * (test.b - test.a))
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[0] = Cas.nsteps
            full_interval_row[0] = get_full_interval(Cas.first_crossing_zero_point)
            time_list_row[0] += time_Cas / repeat * 1000

            T1 = time.perf_counter()
            PC_N = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                    reduction=False, div=div, alp=alp)
            T2 = time.perf_counter()
            time_PC_N = (T2 - T1)
            it_list_row[1] = PC_N.nsteps
            full_interval_row[1] = get_full_interval(PC_N.first_crossing_zero_point)
            time_list_row[1] += time_PC_N / repeat * 1000

            T1 = time.perf_counter()
            PI_N = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                    reduction=False, div=div, alp=alp)
            T2 = time.perf_counter()
            time_PI_N = (T2 - T1)
            it_list_row[2] = PI_N.nsteps
            full_interval_row[2] = get_full_interval(PI_N.first_crossing_zero_point)
            time_list_row[2] += time_PI_N / repeat * 1000

            T1 = time.perf_counter()
            QC_N = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                    reduction=False, div=div, alp=alp)
            T2 = time.perf_counter()
            time_QC_N = (T2 - T1)
            it_list_row[3] = QC_N.nsteps
            full_interval_row[3] = get_full_interval(QC_N.first_crossing_zero_point)
            time_list_row[3] += time_QC_N / repeat * 1000

            T1 = time.perf_counter()
            QI_N = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                    reduction=False, div=div, alp=alp)
            T2 = time.perf_counter()
            time_QI_N = (T2 - T1)
            it_list_row[4] = QI_N.nsteps
            full_interval_row[4] = get_full_interval(QI_N.first_crossing_zero_point)
            time_list_row[4] += time_QI_N / repeat * 1000

            T1 = time.perf_counter()
            PC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                    reduction=True, div=div, alp=alp)
            T2 = time.perf_counter()
            time_PC_R = (T2 - T1)
            it_list_row[5] = PC_R.nsteps
            full_interval_row[5] = get_full_interval(PC_R.first_crossing_zero_point)
            time_list_row[5] += time_PC_R / repeat * 1000

            T1 = time.perf_counter()
            PI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                    reduction=True, div=div, alp=alp)
            T2 = time.perf_counter()
            time_PI_R = (T2 - T1)
            it_list_row[6] = PI_R.nsteps
            full_interval_row[6] = get_full_interval(PI_R.first_crossing_zero_point)
            time_list_row[6] += time_PI_R / repeat * 1000

            T1 = time.perf_counter()
            QC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                    reduction=True, div=div, alp=alp)
            T2 = time.perf_counter()
            time_QC_R = (T2 - T1)
            it_list_row[7] = QC_R.nsteps
            full_interval_row[7] = get_full_interval(QC_R.first_crossing_zero_point)
            time_list_row[7] += time_QC_R / repeat * 1000

            T1 = time.perf_counter()
            QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                    global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                    reduction=True, div=div, alp=alp)
            T2 = time.perf_counter()
            time_QI_R = (T2 - T1)
            it_list_row[8] = QI_R.nsteps
            full_interval_row[8] = get_full_interval(QI_R.first_crossing_zero_point)
            time_list_row[8] += time_QI_R / repeat * 1000

        it_list[i] = it_list_row
        full_interval_list[i] = full_interval_row
        time_list[i] += time_list_row
        i = i + 1
        # time_list.append(time_list_row)
    # it_list.append(it_list_row)

    # print('%s & %d & %d & %d & %d & %d & %d & %d & %d \\\\' %(test.Index,PC_N,PI_N,QC_N,QI_N,PC_R,PI_R,QC_R,QI_R))
    # print('%s & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f \\\\' % (
    # test.Index, time_PC_N, time_PI_N, time_QC_N, time_QI_N, time_PC_R, time_PI_R, time_QC_R, time_QI_R))

    index = 0
    print('test.Index,IBB,PL_N,PI_N,QL_N,QI_N,PL_R,PI_R,QL_R,QI_R')
    for time_row in time_list:
        index = index + 1
        print('%d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\' % (index,
                                                                                          time_row[0],
                                                                                          time_row[1],
                                                                                          time_row[2],
                                                                                          time_row[3],
                                                                                          time_row[4],
                                                                                          time_row[5],
                                                                                          time_row[6],
                                                                                          time_row[7],
                                                                                          time_row[8]))
    min = getMin(time_list)
    max = getMax(time_list)
    avg = getAvg(time_list)
    print_row_f3(min, 'Min')
    print_row_f3(max, 'Max')
    print_row_f3(avg, 'Average')

    min_list = getMinRatio(time_list)
    max_list = getMaxRatio(time_list)
    avg_list = getAvgRatio(time_list)
    print_row_f1(min_list, 'Min')
    print_row_f1(max_list, 'Max')
    print_row_f1(avg_list, 'Average')

    index = 0
    print('test.Index,IBB,PL_N,PI_N,QL_N,QI_N,PL_R,PI_R,QL_R,QI_R')
    for it_row in it_list:
        index = index + 1
        print('%d & %d & %d & %d & %d & %d & %d & %d & %d & %d \\\\' % (index,
                                                                        it_row[0], it_row[1],
                                                                        it_row[2], it_row[3],
                                                                        it_row[4], it_row[5],
                                                                        it_row[6], it_row[7], it_row[8]))
    min_it = getMin(it_list)
    max_it = getMax(it_list)
    avg_it = getAvg(it_list)
    print_row_int(min_it, 'Min')
    print_row_int(max_it, 'Max')
    print_row_f1(avg_it, 'Average')

    min_list = getMinRatio(it_list)
    max_list = getMaxRatio(it_list)
    avg_list = getAvgRatio(it_list)
    print_row_f1(min_list, 'Min')
    print_row_f1(max_list, 'Max')
    print_row_f1(avg_list, 'Average')

    index = 0
    headline = 'test.Index,IBB,PL_N,PI_N,QL_N,QI_N,PL_R,PI_R,QL_R,QI_R'
    print_com_res(headline, full_interval_list, 5)
    avg_interval_list = getAvg(full_interval_list)
    print_row_f5(avg_interval_list, 'Average')


def test_linear(df, eps, repeat, alp=0.5):
    num_variant = 9
    time_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=float)
    it_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=int)
    full_interval_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=ival.Interval)

    i = 0
    for test in df.itertuples():
        it_list_row = np.zeros(num_variant, dtype=int)
        time_list_row = np.zeros(num_variant, dtype=float)
        full_interval_row = np.zeros(num_variant, dtype=ival.Interval)

        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']), True)
        for num in range(0, repeat):
            j = 0

            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps * (test.b - test.a))
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[j] = Cas.nsteps
            full_interval_row[j] = get_full_interval(Cas.first_crossing_zero_point)
            time_list_row[j] += time_Cas / repeat * 1000
            j += 1
            for global_lip in [True, False]:
                T1 = time.perf_counter()
                PC_N = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=False, alp=alp)
                T2 = time.perf_counter()
                time_PC_N = (T2 - T1)
                it_list_row[j] = PC_N.nsteps
                full_interval_row[j] = get_full_interval(PC_N.first_crossing_zero_point)
                time_list_row[j] += time_PC_N / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                PI_N = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=False, alp=alp)
                T2 = time.perf_counter()
                time_PI_N = (T2 - T1)
                it_list_row[j] = PI_N.nsteps
                full_interval_row[j] = get_full_interval(PI_N.first_crossing_zero_point)
                time_list_row[j] += time_PI_N / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                PC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_PC_R = (T2 - T1)
                it_list_row[j] = PC_R.nsteps
                full_interval_row[j] = get_full_interval(PC_R.first_crossing_zero_point)
                time_list_row[j] += time_PC_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                PI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_PI_R = (T2 - T1)
                it_list_row[j] = PI_R.nsteps
                full_interval_row[j] = get_full_interval(PI_R.first_crossing_zero_point)
                time_list_row[j] += time_PI_R / repeat * 1000
                j += 1

        it_list[i] = it_list_row
        full_interval_list[i] = full_interval_row
        time_list[i] += time_list_row
        i = i + 1

    headline = 'test.Index,IBB,PL_N,PI_N,PL_R,PI_R'
    print_com_time(headline, time_list)

    min = getMin(time_list)
    max = getMax(time_list)
    avg = getAvg(time_list)
    print_row_f3(min, 'Min')
    print_row_f3(max, 'Max')
    print_row_f3(avg, 'Average')

    time_less_IBB = getNumLessIBB(time_list)
    print_simple_row_int(time_less_IBB, 'Wins')

    min_list = getMinRatio(time_list)
    max_list = getMaxRatio(time_list)
    avg_list = getAvgRatio(time_list)
    print_row_f3(min_list, 'Min')
    print_row_f3(max_list, 'Max')
    print_row_f3(avg_list, 'Average')

    print_com_res(headline, it_list)

    min_it = getMin(it_list)
    max_it = getMax(it_list)
    avg_it = getAvg(it_list)
    print_row_int(min_it, 'Min')
    print_row_int(max_it, 'Max')
    print_row_f1(avg_it, 'Average')

    it_less_IBB = getNumLessIBB(it_list)
    print_simple_row_int(it_less_IBB, 'Wins')

    min_list = getMinRatio(it_list)
    max_list = getMaxRatio(it_list)
    avg_list = getAvgRatio(it_list)
    print_row_f3(min_list, 'Min')
    print_row_f3(max_list, 'Max')
    print_row_f3(avg_list, 'Average')

    print_com_res(headline, full_interval_list, 5)
    # min_interval_list = getMin(full_interval_list)
    # max_interval_list = getMax(full_interval_list)
    avg_interval_list = getAvg(full_interval_list)
    # print_row_f5(min_interval_list, 'Min')
    # print_row_f5(max_interval_list, 'Max')
    print_row_f5(avg_interval_list, 'Average')
    accuracy_less_IBB = getNumLessIBB(full_interval_list)
    print_simple_row_int(accuracy_less_IBB, 'Wins')


def test_division(df, eps, repeat, alp=0.7, reduction=True, rho_g=36, rho_l=36):
    # divisions = [Division.Bisection, Division.Piyavskii, Division.Casado, Division.FalsiLipschitz]
    divisions = [Division.Bisection, Division.Piyavskii, Division.Casado]
    num_variant = len(divisions) * 2 + 1
    time_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=float)
    it_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=int)
    full_interval_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=ival.Interval)

    i = 0
    for test in df.itertuples():
        it_list_row = np.zeros(num_variant, dtype=int)
        time_list_row = np.zeros(num_variant, dtype=float)
        full_interval_row = np.zeros(num_variant, dtype=ival.Interval)
        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']),
                                  True)
        for num in range(0, repeat):
            j = 0

            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps * (test.b - test.a))
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[j] = Cas.nsteps
            full_interval_row[j] = get_full_interval(Cas.first_crossing_zero_point)
            time_list_row[j] += time_Cas / repeat * 1000
            j += 1
            for glob in [True, False]:
                if glob:
                    rho = rho_g
                else:
                    rho = rho_l
                for division in divisions:
                    T1 = time.perf_counter()
                    res = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                           global_lipschitz_interval=glob, estimator=Estimator.PSL,
                                           div=division, alp=alp, reduction=reduction, rho=rho)
                    T2 = time.perf_counter()
                    time_res = (T2 - T1)
                    it_list_row[j] = res.nsteps
                    full_interval_row[j] = get_full_interval(res.first_crossing_zero_point)
                    time_list_row[j] += time_res / repeat * 1000
                    j += 1

        it_list[i] = it_list_row
        full_interval_list[i] = full_interval_row
        time_list[i] += time_list_row
        i = i + 1

    index = 0
    # if reduction:
    #     headline = '№, IBB, Bisection, Casado, Baumann, Falsi, FalsiTan'
    # else:
    headline = '№, Bisection, Piyavskii, Casado'
    print_com_time(headline, time_list)

    min = getMin(time_list)
    max = getMax(time_list)
    avg = getAvg(time_list)
    # print_row_f3(min, 'Min')
    # print_row_f3(max, 'Max')
    print_row_f3(avg, 'Average')

    time_less_IBB = getNumLessIBB(time_list)
    print_simple_row_int(time_less_IBB, 'Wins')

    # min_list = getMinRatio(time_list)
    # max_list = getMaxRatio(time_list)
    # avg_list = getAvgRatio(time_list)
    # print_row_f3(min_list, 'Min')
    # print_row_f3(max_list, 'Max')
    # print_row_f3(avg_list, 'Average')

    print_com_res(headline, it_list)
    min_it = getMin(it_list)
    max_it = getMax(it_list)
    avg_it = getAvg(it_list)
    # print_row_int(min_it, 'Min')
    # print_row_int(max_it, 'Max')
    print_row_f1(avg_it, 'Average')

    it_less_IBB = getNumLessIBB(it_list)
    print_simple_row_int(it_less_IBB, 'Wins')

    # min_list = getMinRatio(it_list)
    # max_list = getMaxRatio(it_list)
    # avg_list = getAvgRatio(it_list)
    # print_row_f3(min_list, 'Min')
    # print_row_f3(max_list, 'Max')
    # print_row_f3(avg_list, 'Average')


def test_quadratic(df, eps, repeat, alp=0.5):
    num_variant = 9
    time_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=float)
    it_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=int)
    full_interval_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=ival.Interval)

    i = 0
    for test in df.itertuples():
        it_list_row = np.zeros(num_variant, dtype=int)
        time_list_row = np.zeros(num_variant, dtype=float)
        full_interval_row = np.zeros(num_variant, dtype=ival.Interval)

        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']), True)
        for num in range(0, repeat):
            j = 0

            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps * (test.b - test.a))
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[j] = Cas.nsteps
            full_interval_row[j] = get_full_interval(Cas.first_crossing_zero_point)
            time_list_row[j] += time_Cas / repeat * 1000
            j += 1
            for global_lip in [True, False]:
                T1 = time.perf_counter()
                PC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_PC_R = (T2 - T1)
                it_list_row[j] = PC_R.nsteps
                full_interval_row[j] = get_full_interval(PC_R.first_crossing_zero_point)
                time_list_row[j] += time_PC_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                PI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSL,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_PI_R = (T2 - T1)
                it_list_row[j] = PI_R.nsteps
                full_interval_row[j] = get_full_interval(PI_R.first_crossing_zero_point)
                time_list_row[j] += time_PI_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                QC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_QC_R = (T2 - T1)
                it_list_row[j] = QC_R.nsteps
                full_interval_row[j] = get_full_interval(QC_R.first_crossing_zero_point)
                time_list_row[j] += time_QC_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=global_lip, estimator=Estimator.PSQE,
                                        reduction=True, alp=alp)
                T2 = time.perf_counter()
                time_QI_R = (T2 - T1)
                it_list_row[j] = QI_R.nsteps
                full_interval_row[j] = get_full_interval(QI_R.first_crossing_zero_point)
                time_list_row[j] += time_QI_R / repeat * 1000
                j += 1

        it_list[i] = it_list_row
        full_interval_list[i] = full_interval_row
        time_list[i] += time_list_row
        i = i + 1

    headline = 'test.Index, IBB, PL_R, PI_R,QL_R,QI_R'
    print_com_time(headline, time_list)

    min = getMin(time_list)
    max = getMax(time_list)
    avg = getAvg(time_list)
    print_row_f3(min, 'Min')
    print_row_f3(max, 'Max')
    print_row_f3(avg, 'Average')

    time_less_IBB = getNumLessIBB(time_list)
    print_simple_row_int(time_less_IBB, 'Wins')

    min_list = getMinRatio(time_list)
    max_list = getMaxRatio(time_list)
    avg_list = getAvgRatio(time_list)
    print_row_f3(min_list, 'Min', bf=False)
    print_row_f3(max_list, 'Max', bf=False)
    print_row_f3(avg_list, 'Average', bf=False)

    print_com_res(headline, it_list)

    min_it = getMin(it_list)
    max_it = getMax(it_list)
    avg_it = getAvg(it_list)
    print_row_int(min_it, 'Min')
    print_row_int(max_it, 'Max')
    print_row_f1(avg_it, 'Average')

    it_less_IBB = getNumLessIBB(it_list)
    print_simple_row_int(it_less_IBB, 'Wins')

    min_list = getMinRatio(it_list)
    max_list = getMaxRatio(it_list)
    avg_list = getAvgRatio(it_list)
    print_row_f3(min_list, 'Min', bf=False)
    print_row_f3(max_list, 'Max', bf=False)
    print_row_f3(avg_list, 'Average', bf=False)

    print_com_res(headline, full_interval_list, 5)
    # min_interval_list = getMin(full_interval_list)
    # max_interval_list = getMax(full_interval_list)
    avg_interval_list = getAvg(full_interval_list)
    # print_row_f5(min_interval_list, 'Min')
    # print_row_f5(max_interval_list, 'Max')
    print_row_f5(avg_interval_list, 'Average')
    accuracy_less_IBB = getNumLessIBB(full_interval_list)
    print_simple_row_int(accuracy_less_IBB, 'Wins')


def test_adap_lip(df, eps, repeat, alp=0.5):
    num_variant = 9
    time_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=float)
    it_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=int)
    full_interval_list = np.zeros((len(list(df.itertuples())), num_variant), dtype=ival.Interval)

    i = 0
    for test in df.itertuples():
        it_list_row = np.zeros(num_variant, dtype=int)
        time_list_row = np.zeros(num_variant, dtype=float)
        full_interval_row = np.zeros(num_variant, dtype=ival.Interval)

        points_db[test.Index] = {'bnb2_pslint_points_list': []}
        prob = uvpr.UniVarProblem(test.Index, test.objective, test.a, test.b, test.min_f, test.min_x,
                                  lambda x: log_point(x, points_db[test.Index]['bnb2_pslint_points_list']), True)
        for num in range(0, repeat):
            j = 0

            T1 = time.perf_counter()
            Cas = sfzcp.cas(prob=prob, epsilon=eps * (test.b - test.a))
            T2 = time.perf_counter()
            time_Cas = (T2 - T1)
            it_list_row[j] = Cas.nsteps
            full_interval_row[j] = get_full_interval(Cas.first_crossing_zero_point)
            time_list_row[j] += time_Cas / repeat * 1000
            j += 1
            for adap_lip in [False, True]:
                T1 = time.perf_counter()
                PC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=False, estimator=Estimator.PSL,
                                        reduction=True,adap_lip=adap_lip, alp=alp)
                T2 = time.perf_counter()
                time_PC_R = (T2 - T1)
                it_list_row[j] = PC_R.nsteps
                full_interval_row[j] = get_full_interval(PC_R.first_crossing_zero_point)
                time_list_row[j] += time_PC_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                PI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=False, estimator=Estimator.PSL,
                                        reduction=True,adap_lip=adap_lip, alp=alp)
                T2 = time.perf_counter()
                time_PI_R = (T2 - T1)
                it_list_row[j] = PI_R.nsteps
                full_interval_row[j] = get_full_interval(PI_R.first_crossing_zero_point)
                time_list_row[j] += time_PI_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                QC_R = sfzcp.new_method(prob, symm=True, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=False, estimator=Estimator.PSQE,
                                        reduction=True,adap_lip=adap_lip, alp=alp)
                T2 = time.perf_counter()
                time_QC_R = (T2 - T1)
                it_list_row[j] = QC_R.nsteps
                full_interval_row[j] = get_full_interval(QC_R.first_crossing_zero_point)
                time_list_row[j] += time_QC_R / repeat * 1000
                j += 1

                T1 = time.perf_counter()
                QI_R = sfzcp.new_method(prob, symm=False, epsilon=eps * (test.b - test.a),
                                        global_lipschitz_interval=False, estimator=Estimator.PSQE,
                                        reduction=True,adap_lip=adap_lip, alp=alp)
                T2 = time.perf_counter()
                time_QI_R = (T2 - T1)
                it_list_row[j] = QI_R.nsteps
                full_interval_row[j] = get_full_interval(QI_R.first_crossing_zero_point)
                time_list_row[j] += time_QI_R / repeat * 1000
                j += 1

        it_list[i] = it_list_row
        full_interval_list[i] = full_interval_row
        time_list[i] += time_list_row
        i = i + 1

    headline = 'test.Index, IBB, PL_R, PI_R,QL_R,QI_R'
    print_com_time(headline, time_list)

    min = getMin(time_list)
    max = getMax(time_list)
    avg = getAvg(time_list)
    print_row_f3(min, 'Min')
    print_row_f3(max, 'Max')
    print_row_f3(avg, 'Average')

    time_less_IBB = getNumLessIBB(time_list)
    print_simple_row_int(time_less_IBB, 'Wins')

    min_list = getMinRatio(time_list)
    max_list = getMaxRatio(time_list)
    avg_list = getAvgRatio(time_list)
    print_row_f3(min_list, 'Min', bf=False)
    print_row_f3(max_list, 'Max', bf=False)
    print_row_f3(avg_list, 'Average', bf=False)

    print_com_res(headline, it_list)

    min_it = getMin(it_list)
    max_it = getMax(it_list)
    avg_it = getAvg(it_list)
    print_row_int(min_it, 'Min')
    print_row_int(max_it, 'Max')
    print_row_f1(avg_it, 'Average')

    it_less_IBB = getNumLessIBB(it_list)
    print_simple_row_int(it_less_IBB, 'Wins')

    min_list = getMinRatio(it_list)
    max_list = getMaxRatio(it_list)
    avg_list = getAvgRatio(it_list)
    print_row_f3(min_list, 'Min', bf=False)
    print_row_f3(max_list, 'Max', bf=False)
    print_row_f3(avg_list, 'Average', bf=False)

    print_com_res(headline, full_interval_list, 5)
    # min_interval_list = getMin(full_interval_list)
    # max_interval_list = getMax(full_interval_list)
    avg_interval_list = getAvg(full_interval_list)
    # print_row_f5(min_interval_list, 'Min')
    # print_row_f5(max_interval_list, 'Max')
    print_row_f5(avg_interval_list, 'Average')
    accuracy_less_IBB = getNumLessIBB(full_interval_list)
    print_simple_row_int(accuracy_less_IBB, 'Wins')
