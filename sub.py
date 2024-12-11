class Sub:
    """ Generic subproblem

    Class for storing information regarding the subproblem.

    Attributes:
        level(int): level in the BnB-tree
        bound(double): the bound (can be upper or lower depending on the problem)
        data(any): the problem-specific data associated with the subproblem
    """

    def __init__(self, level, bound, data):
        """
        Initializes the subproblem.

        Args:
            level: level in the BnB-tree
            bound: the bound (can be upper or lower depending on the problem)
            data: the problem-specific data associated with the subproblem
        """
        self.level = level
        self.bound = bound
        self.data = data

    def __repr__(self):
        return "[ level = " + str(self.level) + ", bound = " + str(self.bound) + ", data =  " + str(self.data) + "]"

