"""
The piece-wise smooth quadratic estimators from Posypkin, M.A., Sergeyev, Y.D.: Efficient smooth minorants for global
optimization of univariate functions with the first derivative satisfying the interval lipschitz condition.
Journal of Global Optimization, 1–29 (2022)
"""


class PSQE_Bounds:
    """
    Piecewise quadratic underestimator
    """

    def __init__(self, a: float, b: float, alp: float, bet: float, fa: float, fb: float, dfa: float, dfb: float,
                 under: bool):
        """
        The smooth piecewise quadratic estimator constiructor
        Args:
            a: left interval end
            b: right interval end
            alp: lower end of the Lipschitzian interval for derivative
            bet: upper end of the Lipschitzian interval for derivative
            f: objective
            df: objective's derivative
            under: True - compute the under estimator, False - compute the over estimator
        """
        self.a = a
        self.b = b
        self.under = under
        if under:
            self.alp = alp
            self.bet = bet
            self.fa = fa
            self.fb = fb
            self.dfa = dfa
            self.dfb = dfb
        else:
            self.alp = -bet
            self.bet = -alp
            self.fa = -fa
            self.fb = -fb
            self.dfa = -dfa
            self.dfb = -dfb

        delt = (self.dfb - self.dfa - self.alp * (self.b - self.a)) / (self.bet - self.alp)
        self.c = (self.alp * (b - a) + self.dfa - self.dfb) / (2 * (self.bet - self.alp)) + (
                self.fa - self.fb + b * self.dfb - a * self.dfa + self.alp * (a ** 2 - b ** 2) / 2) / (
                         self.dfb - self.dfa - self.alp * (b - a))
        self.d = self.c + delt

        self.qc = self.fa + self.dfa * (self.c - self.a) + self.alp / 2 * (self.c - self.a) ** 2
        self.qd = self.fb + self.dfb * (self.d - self.b) + self.alp / 2 * (self.d - self.b) ** 2

        if self.d > self.b:
            print('error d>b')
            print('a=', self.a, 'b=', self.b, 'c=', self.c, 'delt=', delt, 'fa=', self.fa, 'fb=', self.fb, 'alp=',
                  self.alp, 'beta=', self.bet)

    def __repr__(self):
        return "Estimator " + "a = " + str(self.a) + ", b = " + str(self.b) + ", c = " + str(self.c) + ", d = " + str(
            self.d) + ", alp = " + str(self.alp) + ", bet = " + str(self.bet) + ", fa = " + str(
            self.fa) + ", fb = " + str(self.fb) + ", dfa = " + str(self.dfa) + ", dfb = " + str(self.dfb)

    def q1(self, x: float):
        return self.fa + self.dfa * (x - self.a) + self.alp / 2 * (x - self.a) ** 2

    def q2(self, x: float):
        return self.qc + (self.dfa + self.alp * (self.c - self.a)) * (x - self.c) + self.bet / 2 * (x - self.c) ** 2

    def q3(self, x: float):
        return self.fb + self.dfb * (x - self.b) + 0.5 * self.alp * (x - self.b) ** 2

    def estimator(self, x: float):
        """
        The piecewise quadratic underestimator
        Args:
            x: argument

        Returns: underestimator's value
        """
        assert self.a <= x <= self.b
        if x <= self.c:
            return self.q1(x)
        elif x < self.d:
            return self.q2(x)
        else:
            return self.q3(x)

    def nestimator(self, x):
        return -self.estimator(x)

    def dq1(self, x: float):
        return self.dfa + self.alp * (x - self.a)

    def dq2(self, x: float):
        return self.dfa + self.alp * (self.c - self.a) + self.bet * (x - self.c)

    def dq3(self, x: float):
        return self.dfb + self.alp * (x - self.b)

    def estimators_derivative(self, x):
        """
        The piecewise linear underestimator's derivative
        Args:
            x: argument

        Returns: underestimator's derivative value
        """
        if x < self.c:
            res = self.dq1(x)
        elif x < self.d:
            res = self.dq2(x)
        else:
            res = self.dq3(x)
        # if self.under:
        return res
        # else:
        #     return -res

    def lower_bound_and_point(self):
        """
        Returns: Tuple (point where it is achieved, lower bound on interval [a,b])
        """
        x_list = [self.a, self.c, self.d, self.b]
        df_list = [self.estimators_derivative(x) for x in x_list]
        check_list = [self.a, self.b]
        record_x = None
        record_v = None
        ln = len(x_list)
        for i in range(ln - 1):
            x = self.find_argmin(x_list[i], df_list[i], x_list[i + 1], df_list[i + 1])
            if not (x is None):
                check_list.append(x)
        # print(check_list)
        for x in check_list:
            v = self.estimator(x)
            if record_v is None or v < record_v:
                record_v = v
                record_x = x
        if not self.under:
            record_v = -record_v
        return record_x, record_v

    def record_and_point(self):
        """
        Returns: Tuple (point c where the best value of objective is achieved (a or b), f(c))
        """
        return (self.a, self.fa) if self.fa <= self.fb else (self.b, self.fb)

    def find_argmin(self, x1, df1, x2, df2):
        if df1 == 0 and df2 == 0:
            xs = 0.5 * (x1 + x2)
        elif df1 <= 0 <= df2:
            xs = x1 + (-df1) * (x2 - x1) / (df2 - df1)
        else:
            xs = None
        return xs

    def under_est_der_le_0(self, num1, num2):
        """
        if(\phi'(num1) * \phi'(num2)<0) return true, else return false.
        """
        est_der_num1 = self.estimators_derivative(num1)
        est_der_num2 = self.estimators_derivative(num2)
        if est_der_num1 < 0 and est_der_num2 > 0:
            return True
        else:
            return False

    def upper_est_der_le_0(self, num1, num2):
        """
        if(\phi'(num1) * \phi'(num2)<0) return true, else return false.
        """
        est_der_num1 = self.estimators_derivative(num1)
        est_der_num2 = self.estimators_derivative(num2)
        if est_der_num1 > 0 and est_der_num2 < 0:
            return True
        else:
            return False

    def delta_first(self):
        return self.dfa ** 2 - 2 * self.alp * self.fa

    def delta_second(self):
        c = self.qc
        b = self.dfa + self.alp * (self.c - self.a)
        return b ** 2 - 2 * self.bet * c

    def delta_third(self):
        return self.dfb ** 2 - 2 * self.alp * self.fb

    def root_first_left(self, d1):
        return self.a + (-self.dfa - d1 ** 0.5) / self.alp

    def root_second_left(self, d2):
        return self.c + (-self.dfa - self.alp * (self.c - self.a) - d2 ** 0.5) / self.bet

    def root_third_left(self, d3):
        return self.b + (-self.dfb - d3 ** 0.5) / self.alp

    def root_first_right(self, d1):
        return self.a + (-self.dfa + d1 ** 0.5) / self.alp

    def root_second_right(self, d2):
        return self.c + (-self.dfa - self.alp * (self.c - self.a) + d2 ** 0.5) / self.bet

    def root_third_right(self, d3):
        return self.b + (-self.dfb + d3 ** 0.5) / self.alp

    def get_left_end(self):
        d1 = self.delta_first()
        if self.qc <= 0:
            if self.alp == 0:
                return self.a - self.fa / self.dfa
            else:
                return self.root_first_left(d1)
        dqc = self.dq1(self.c)
        if self.dfa < 0 and dqc > 0 and d1 >= 0:
            return self.root_first_left(d1)

        d2 = self.delta_second()
        if self.qd <= 0:
            if self.bet == 0:
                return self.c - self.qc / (self.dfa + self.alp * (self.c - self.a))
            else:
                return self.root_second_left(d2)
        dqd = self.dq3(self.d)
        if dqc < 0 and dqd > 0 and d2 >= 0:
            return self.root_second_left(d2)

        d3 = self.delta_third()
        if self.fb <= 0:
            if self.alp == 0:
                return self.b - self.fb / self.dfb
            else:
                return self.root_third_left(d3)
        elif dqd < 0 and self.dfb > 0 and d3 >= 0:
            return self.root_third_left(d3)
        else:
            return None

    def get_left_end_old(self):
        """
            get the first root of under estimator
        """
        assert self.under
        if self.estimator(self.c) <= 0:
            d1 = self.delta_first()
            return self.root_first_left(d1)
        else:
            d1 = self.delta_first()
            if self.under_est_der_le_0(self.a, self.c) and d1 >= 0:
                return self.root_first_left(d1)

        if self.estimator(self.d) <= 0:
            d2 = self.delta_second()
            return self.root_second_left(d2)
        else:
            d2 = self.delta_second()
            if self.under_est_der_le_0(self.c, self.d) and d2 >= 0:
                return self.root_second_left(d2)

        if self.estimator(self.b) <= 0:
            d3 = self.delta_third()
            return self.root_third_left(d3)
        else:
            d3 = self.delta_third()
            if self.under_est_der_le_0(self.d, self.b) and d3 >= 0:
                return self.root_third_left(d3)
        return None

    def get_right_end_upper_bound(self):
        assert not self.under
        d1 = self.delta_first()
        if self.qc >= 0:
            if self.alp == 0:
                return self.a - self.fa / self.dfa
            else:
                return self.root_first_right(d1)
        dqc = self.dq1(self.c)
        if self.dfa > 0 and dqc < 0 and d1 >= 0:
            return self.root_first_right(d1)
        d2 = self.delta_second()
        if self.qd >= 0:
            if self.bet == 0:
                return self.c - self.qc / (self.dfa + self.alp * (self.c - self.a))
            else:
                return self.root_second_right(d2)
        dqd = self.dq3(self.d)
        if dqc > 0 and dqd < 0 and d2 >= 0:
            return self.root_second_right(d2)
        d3 = self.delta_third()
        if self.fb >= 0:
            if self.alp == 0:
                return self.b - self.fb / self.dfb
            else:
                return self.root_third_right(d3)
        elif dqd > 0 and self.dfb < 0 and d3 >= 0:
            return self.root_third_right(d3)
        return self.b

    def get_right_end_upper_bound_old(self):
        """
            get the last root of over estimator
        """
        assert not self.under
        if self.estimator(self.c) >= 0:
            d1 = self.delta_first()
            return self.root_first_right(d1)
        else:
            d1 = self.delta_first()
            if self.upper_est_der_le_0(self.a, self.c) and d1 >= 0:
                return self.root_first_right(d1)

        if self.estimator(self.d) >= 0:
            d2 = self.delta_second()
            return self.root_second_right(d2)
        else:
            d2 = self.delta_second()
            if self.upper_est_der_le_0(self.c, self.d) and d2 >= 0:
                return self.root_second_right(d2)

        if self.estimator(self.b) >= 0:
            d3 = self.delta_third()
            return self.root_third_right(d3)
        else:
            d3 = self.delta_third()
            if self.upper_est_der_le_0(self.d, self.b) and d3 >= 0:
                return self.root_third_right(d3)

        return self.b

    def get_right_end_under_bound(self):
        d3 = self.delta_third()
        if self.qd <= 0:
            if self.alp == 0:
                return self.b - self.fb / self.dfb
            else:
                return self.root_third_right(d3)
        dqd = self.dq3(self.d)
        if dqd < 0 and self.dfb > 0 and d3 >= 0:
            return self.root_third_right(d3)

        d2 = self.delta_second()
        if self.qc <= 0:
            if self.bet == 0:
                return self.c - self.qc / (self.dfa + self.alp * (self.c - self.a))
            else:
                return self.root_second_right(d2)
        dqc = self.dq1(self.c)
        if dqc < 0 and dqd > 0 and d2 >= 0:
            return self.root_second_right(d2)

        d1 = self.delta_first()
        if self.fa <= 0:
            if self.alp == 0:
                return self.a - self.fa / self.dfa
            else:
                return self.root_first_right(d1)
        if self.dfa < 0 and dqc > 0 and d1 >= 0:
            return self.root_first_right(d1)

        return None

    def get_right_end_under_bound_old(self):
        """
            get the last root of under estimator
        """
        assert self.under
        if self.estimator(self.d) <= 0:
            d1 = self.delta_third()
            return self.root_third_right(d1)
        else:
            d1 = self.delta_third()
            if self.under_est_der_le_0(self.d, self.b) and d1 >= 0:
                return self.root_third_right(d1)
        if self.estimator(self.c) <= 0:
            d2 = self.delta_second()
            return self.root_second_right(d2)
        else:
            d2 = self.delta_second()
            if self.under_est_der_le_0(self.c, self.d) and d2 >= 0:
                return self.root_second_right(d2)
        if self.estimator(self.a) <= 0:
            d3 = self.delta_first()
            return self.root_first_right(d3)
        else:
            d3 = self.delta_third()
            if self.under_est_der_le_0(self.a, self.c) and d3 >= 0:
                return self.root_first_right(d3)
        return None
