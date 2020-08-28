<p align="center">
  <a href="https://github.com/nschloe/krylov"><img alt="krylov" src="https://nschloe.github.io/krylov/logo.svg" width="60%"></a>
  <p align="center">Krylov subspace methods.</p>
</p>

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/krylov/ci?style=flat-square)](https://github.com/nschloe/krylov/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/krylov.svg?style=flat-square)](https://codecov.io/gh/nschloe/krylov)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/krylov.svg?style=flat-square)](https://pypi.org/pypi/krylov/)
[![PyPi Version](https://img.shields.io/pypi/v/krylov.svg?style=flat-square)](https://pypi.org/project/krylov)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/krylov.svg?logo=github&label=Stars&logoColor=white&style=flat-square)](https://github.com/nschloe/krylov)
[![PyPi downloads](https://img.shields.io/pypi/dm/krylov.svg?style=flat-square)](https://pypistats.org/packages/krylov)

Forked from André Gaul's [KryPy](https://github.com/andrenarchy/krypy), _krylov_
provides Krylov subspace methods for Python. It is more advanced than [SciPy's linalg
methods](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) in that it
provides for arbitrary inner products and is fully blocked/vectorized. _krylov_ is
written completely in Python.

Install with
```
pip install krylov
```
and use as
```python
import numpy
import krylov

A = numpy.diag([1.0e-3] + list(range(2, 101)))
b = numpy.ones(100)

# sol, info = krylov.cg(A, b)
# sol, info = krylov.minres(A, b)
sol, info = krylov.gmres(A, b)

# sol is None if no solution has been found
# onfo.resnorms contains the relative residual norms and some more data

# plot residuals
import matplotlib.pyplot as plt

plt.semilogy(info.resnorms)
plt.show()
```

### Testing
To run the krylov unit tests, check out this repository and type
```
pytest
```

### License
krylov is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
