import numpy as np
import sympy
import pysym
import symengine

# Real-life example (ion speciation problem in water chemistry)

_ref = np.array([37.252574322668998, 22.321937961124899, 10.9011158998744,
                 20.190422234652999, 27.8679190043357, 33.933606208922598,
                 33.552055153126204, 31.440168027241697, 37.999293413509498,
                 41.071619997204103, -20.619381941508539, 111.68831884983794,
                 29.210791083803763, 18.901100113049495, 17.18281828459045])


def get_syms_exprs(backend):
    x = backend.symarray('x', 14)
    p = backend.symarray('p', 14)
    syms = np.concatenate((x, p))
    exp = backend.exp
    exprs = [
        x[0] + x[1] - x[4] + 36.252574322669,
        x[0] - x[2] + x[3] + 21.3219379611249,
        x[3] + x[5] - x[6] + 9.9011158998744,
        2*x[3] + x[5] - x[7] + 18.190422234653,
        3*x[3] + x[5] - x[8] + 24.8679190043357,
        4*x[3] + x[5] - x[9] + 29.9336062089226,
        -x[10] + 5*x[3] + x[5] + 28.5520551531262,
        2*x[0] + x[11] - 2*x[4] - 2*x[5] + 32.4401680272417,
        3*x[1] - x[12] + x[5] + 34.9992934135095,
        4*x[1] - x[13] + x[5] + 37.0716199972041,
        (
            p[0] - p[1] + 2*p[10] + 2*p[11] - p[12] - 2*p[13] +
            p[2] + 2*p[5] + 2*p[6] + 2*p[7] + 2*p[8] + 2*p[9] -
            exp(x[0]) + exp(x[1]) - 2*exp(x[10]) - 2*exp(x[11]) +
            exp(x[12]) + 2*exp(x[13]) - exp(x[2]) - 2*exp(x[5]) -
            2*exp(x[6]) - 2*exp(x[7]) - 2*exp(x[8]) - 2*exp(x[9])
        ), (
            -p[0] - p[1] - 15*p[10] - 2*p[11] - 3*p[12] - 4*p[13] -
            4*p[2] - 3*p[3] - 2*p[4] - 3*p[6] - 6*p[7] - 9*p[8] -
            12*p[9] + exp(x[0]) + exp(x[1]) + 15*exp(x[10]) +
            2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + 4*exp(x[2]) +
            3*exp(x[3]) + 2*exp(x[4]) + 3*exp(x[6]) + 6*exp(x[7]) +
            9*exp(x[8]) + 12*exp(x[9])
        ), (
            -5*p[10] - p[2] - p[3] - p[6] - 2*p[7] - 3*p[8] - 4*p[9] +
            5*exp(x[10]) + exp(x[2]) + exp(x[3]) + exp(x[6]) +
            2*exp(x[7]) + 3*exp(x[8]) + 4*exp(x[9])
        ), (
            -p[1] - 2*p[11] - 3*p[12] - 4*p[13] - p[4] + exp(x[1]) +
            2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + exp(x[4])
        ), (
            -p[10] - 2*p[11] - p[12] - p[13] - p[5] - p[6] - p[7] -
            p[8] - p[9] + exp(x[10]) + 2*exp(x[11]) + exp(x[12]) +
            exp(x[13]) + exp(x[5]) + exp(x[6]) + exp(x[7]) +
            exp(x[8]) + exp(x[9])
        )
    ]
    return syms, exprs


class TimeLambdifyInit:

    params = ['sympy', 'symengine', 'pysym']

    def time_init(self, name):
        self.syms, self.exprs = get_syms_exprs(sym.Backend(name))


backend_names = list(sym.Backend.backends.keys())
n_backends = len(backend_names)


def _lzip(n):
    return list(zip([n]*n_backends, backend_names))


class _TimeLambdifyEval:

    params = (_lzip(1), _lzip(100))
    param_names = ('n', 'backend')

    def setup(self, n, name):
        self.inp = np.ones(28)
        self.backend = sym.Backend(name)
        self.syms, self.exprs = get_syms_exprs(self.backend)
        self.lmb = self.backend(self.syms, self.exprs)
        self.values = {}

    def time_evaluate(self, n, name):
        for i in range(n):
            res = self.lmb(self.inp)
        if not np.allclose(res, _ref):
            raise ValueError('Incorrect result')
