from .. import Backend
import pytest


backends = []
for bk in Backend.backends.keys():
    try:
        _be = Backend(bk)
    except ImportError:
        continue

    _x = _be.Symbol('x')
    try:
        _be.cse([_x])
    except:
        continue

    backends.append(bk)


def _inverse_cse(subs_cses, cse_exprs):
    subs = dict(subs_cses)
    return [expr.subs(subs) for expr in cse_exprs]


@pytest.mark.parametrize('key', backends)
def test_basic_cse(key):
    be = Backend(key)
    x, y = map(be.Symbol, "xy")
    exprs = [x**2 + y**2 + 3, be.exp(x**2 + y**2)]
    subs_cses, cse_exprs = be.cse(exprs)
    subs, cses = zip(*subs_cses)
    assert cses[0] == x**2 + y**2
    for cse_expr in cse_exprs:
        assert x not in cse_expr.atoms()
        assert y not in cse_expr.atoms()
    assert _inverse_cse(subs_cses, cse_exprs) == exprs


@pytest.mark.parametrize('key', backends)
def test_moot_cse(key):
    be = Backend(key)
    x, y = map(be.Symbol, "xy")
    exprs = [x**2 + y**2, y]
    subs_cses, cse_exprs = be.cse(exprs)
    assert not subs_cses
    assert _inverse_cse(subs_cses, cse_exprs) == exprs


@pytest.mark.parametrize('key', backends)
def test_cse_with_symbols(key):
    be = Backend(key)
    x = be.Symbol('x')
    exprs = [x**2, 1/(1 + x**2), be.log(x + 2), be.exp(x + 2)]
    subs_cses, cse_exprs = be.cse(exprs, symbols=be.numbered_symbols('y'))
    subs, cses = zip(*subs_cses)
    assert subs[0] == be.Symbol('y0')
    assert subs[1] == be.Symbol('y1')
    assert _inverse_cse(subs_cses, cse_exprs) == exprs


@pytest.mark.parametrize('key', backends)
def test_cse_with_symbols_overlap(key):
    be = Backend(key)
    x0, x1, y = map(be.Symbol, "x0 x1 y".split())
    exprs = [x0**2, x0**2 + be.exp(y)**2 + 3, x1 * be.exp(y), be.sin(x1 * be.exp(y) + 1)]
    subs_cses, cse_exprs = be.cse(exprs)
    assert _inverse_cse(subs_cses, cse_exprs) == exprs
