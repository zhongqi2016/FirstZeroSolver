import sympy as sym
import interval as ival


class UniVarProblem:
    """ Univariate problem

    A class for storing information about a univariate optimization problem

    Attributes:
        name: name of a test problem
        sym_objective: objective in sympy formal
        objective: objective as python function
        a: left interval end
        b: right interval end
        min_f: global minumum function's value
        min_x: global minimum point
        logger: logging function
    """

    def __init__(self, name, objective, a, b, min_f, min_x, logger=lambda x: x, preproc=False):
        """ Constructor
        Args:
            name: name of a test example
            objective: objective in sympy format
            a: left end of a feasible interval
            b: right end of a feasible interval
            min_f: global minimum (best known) value
            min_x: global minimum (best known) point
        """
        self.name = name
        self.sym_objective = sym.sympify(objective)
        x = sym.symbols('x')
        if preproc and self.sym_objective.subs(x, a) < 0:
            self.sym_objective = -self.sym_objective

        self.sym_df = self.sym_objective.diff()
        self.sym_ddf = self.sym_df.diff()

        module_sin = {"sin": ival.sin}
        module_cos = {"cos": ival.cos}
        module_exp = {"exp": ival.exp}
        # module_abs = {"abs": abs}
        module_log = {"log": ival.log}
        module_sqrt = {"sqrt": ival.sqrt}

        obj_f = sym.lambdify(x, self.sym_objective,
                             modules=[module_sin, module_cos, module_exp, module_log, module_sqrt])

        def obj_log(x):
            logger(x)
            return obj_f(x)

        #         self.objective = sym.lambdify(x, self.sym_objective)
        #         self.objective = obj_f
        self.objective = obj_log
        self.df = sym.lambdify(x, self.sym_df,
                               modules=[module_sin, module_cos, module_exp, module_log, module_sqrt])
        self.ddf = sym.lambdify(x, self.sym_ddf,
                                modules=[module_sin, module_cos, module_exp, module_log, module_sqrt])
        self.a = a
        self.b = b
        self.min_f = min_f
        self.min_x = min_x

    def __repr__(self):
        return self.name + ": " + str(self.sym_objective) + " -> min, " + str(self.a) + " <= x <= " + str(
            self.b) + ", f* = " + str(self.min_f) + ", x* = " + str(self.min_x)
