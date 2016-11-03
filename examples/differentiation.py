#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sym import Backend


def main():
    for key in 'sympy pysym symengine'.split():
        print(key)
        print('    Differentiation:')
        be = Backend(key)
        x, y = map(be.Symbol, 'x y'.split())
        expr = (x - be.acos(y))*be.exp(x + y)
        print(expr)
        Dexpr = expr.diff(y)
        print(Dexpr)
        print("")

if __name__ == '__main__':
    main()
