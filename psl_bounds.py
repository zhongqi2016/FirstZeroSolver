"""
The piece-wise linear underestimator from Casado, L. G., MartÍnez, J. A., GarcÍa, I., & Sergeyev, Y. D. (2003). New interval analysis support functions using gradient information in a global minimization algorithm. Journal of Global Optimization, 25(4), 345-362.
"""
import math


class PSL_Bounds:
    """
    Piecewise linear underestimator
    """

    def __init__(self, a: float, b: float, alp: float, bet: float, fa: float, fb: float, under: bool):
        """
        The piecewise linear estimator constiructor
        Args:
            a: left interval end
            b: right interval end
            alp: lower end of the Lipschitzian interval
            bet: upper end of the Lipschitzian interval
            f: objective
            under: True - compute the under estimator, False - compute the over estimator
        """
        self.a = a
        self.b = b
        self.under = under
        if under:
            self.gam = alp
            self.lam = bet
            self.fa = fa
            self.fb = fb
        else:
            self.gam = -bet
            self.lam = -alp
            self.fa = -fa
            self.fb = -fb
        if self.gam != self.lam:
            self.c = (self.fa - self.fb + self.lam * self.b - self.gam * self.a) / (self.lam - self.gam)
            self.u = (self.lam * self.fa - self.gam * self.fb + self.lam * self.gam * (self.b - self.a)) / (
                    self.lam - self.gam)
        else:
            self.c = None
            self.u = None
        if not self.under:
            self.u = -self.u

    def __repr__(self):
        return "Piecewise linear estimator " + "a = " + str(self.a) + ", b = " + str(self.b) + ", c = " + str(
            self.c) + ", alp = " + str(self.gam) + ", bet = " + str(self.lam) \
            + ", fa = " + str(self.fa) + ", fb = " + str(self.fb)

    def estimator(self, x):
        """
        The piecewise linear underestimator
        Args:
            x: argument within [a,b]

        Returns: underestimator's value
        """
        if self.gam == self.lam:
            return self.fa
        if x <= self.c:
            res = self.fa + self.gam * (x - self.a)
        else:
            res = self.fb + self.lam * (x - self.b)
        return res

    def nestimator(self, x):
        return -self.estimator(x)

    def lower_bound_and_point(self):
        """
        Returns: Tuple (point where it is achieved, lower bound on interval [a,b])
        """
        if self.gam >= 0:
            record_x = self.a
            record_v = self.fa
        elif self.lam <= 0:
            record_x = self.b
            record_v = self.fb
        else:
            record_x = self.c
            record_v = self.estimator(self.c)
        if not self.under:
            record_v = -record_v
        return record_x, record_v

    def record_and_point(self):
        """
        Returns: Tuple (point c where the best value of objective is achieved (a or b), f(c))
        """
        return (self.a, self.fa) if self.fa <= self.fb else (self.b, self.fb)

    def get_left_end(self):
        """
            get the first root of under estimator
        """
        assert self.under
        if self.gam==self.lam:
            if self.fa==0:
                return self.a
            else:
                return None

        if self.u <= 0:
            return self.a - self.fa / self.gam
        elif self.u > 0 and self.fb <= 0:
            return self.b - self.fb / self.lam
        else:
            return None

    def get_right_end_upper_bound(self):
        """
            get the first root of over estimator
        """
        assert not self.under
        if self.gam==self.lam:
            if self.fa==0:
                return self.a
            else:
                return None
        if self.u > 0:
            return self.b - self.fb / self.lam
        else:
            return self.a - self.fa / self.gam

    def get_right_end_under_bound(self):
        """
            get the last root of under estimator
        """
        if self.gam==self.lam:
            if self.fa==0:
                return self.a
            else:
                return None
        assert self.u <= 0
        return self.b - self.fb / self.lam

    def get_left_end_old(self):
        """
            get the first root of under estimator
        """
        assert self.under
        if self.gam >= 0:
            return None
        root_of_left_part = self.a - self.fa / self.gam
        if math.isnan(self.c):
            if root_of_left_part <= self.b:
                return root_of_left_part
            else:
                return None
        if root_of_left_part <= self.c:
            return root_of_left_part
        if self.lam < 0:
            root_of_right_part = self.b - self.fb / self.lam
            if root_of_right_part <= self.b:
                return root_of_right_part
        else:
            return None

    def get_right_end_upper_bound_old(self):
        """
            get the last root of over estimator
        """
        assert not self.under
        if self.gam > 0:
            root_of_left_part = self.a - self.fa / self.gam
            if root_of_left_part <= self.c:
                return root_of_left_part
        root_of_right_part = self.b - self.fb / self.lam
        if self.c <= root_of_right_part <= self.b:
            return root_of_right_part
        else:
            return self.b

    def get_right_end_under_bound_old(self):
        """
            get the last root of under estimator
        """
        assert self.under
        if self.lam < 0:
            return None
        root_of_right_part = self.b - self.fb / self.lam
        if root_of_right_part >= self.c:
            return root_of_right_part
        root_of_left_part = self.a - self.fa / self.gam
        if root_of_left_part <= self.c:
            return root_of_left_part
        return None
