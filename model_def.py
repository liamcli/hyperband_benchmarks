import abc
class ModelInf:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def generate_arms(self,n, dir):
        pass
    @abc.abstractmethod
    def run_solver(self,niters,arm):
        pass
