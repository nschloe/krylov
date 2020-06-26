from collections import namedtuple

import numpy
import scipy.linalg

from . import utils
from ._helpers import Identity, Product
from .arnoldi import Arnoldi
from .errors import ArgumentError, ConvergenceError
from .givens import givens


def gmres(
    A,
    b,
    M=Identity(),
    Ml=Identity(),
    Mr=Identity(),
    inner=lambda x, y: numpy.dot(x.T.conj(), y),
    exact_solution=None,
    ortho="mgs",
    x0=None,
    U=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    r"""Preconditioned GMRES method.

    The *preconditioned generalized minimal residual method* can be used to
    solve a system of linear algebraic equations. Let the following linear
    algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y`.
    The preconditioned GMRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Arnoldi alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Arnoldi is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If :math:`M` is used the memory consumption is 2*(maxiter+1).

    If the operator :math:`M_l A M_r` is self-adjoint then consider using
    the MINRES method :py:class:`Minres`.
    """

    def _get_xk(y):
        if y is None:
            return x0
        k = arnoldi.iter
        if k > 0:
            yy = scipy.linalg.solve_triangular(R[:k, :k], y)
            yk = sum(c * v for c, v in zip(yy, V[:-1]))
            Mr_yk = yk if Mr is None else Mr @ yk
            return x0 + Mr_yk
        return x0

    def get_residual(z):
        r"""Compute residual.

        For a given :math:`z\in\mathbb{C}^N`, the residual

        .. math::

          r = M M_l ( b - A z )


        :param z: approximate solution.
        """
        r = b - A @ z
        Mlr = r if Ml is None else Ml @ r
        MMlr = Mlr if M is None else M @ Mlr
        return MMlr, Mlr

    def get_residual_norm(z):
        """
        The absolute residual norm

        .. math::

          \\| M M_l (b-Az)\\|_{M^{-1}}

        is computed.
        """
        return get_residual_and_norm(z)[2]

    def get_residual_and_norm(z):
        MMlr, Mlr = get_residual(z)
        return MMlr, Mlr, numpy.sqrt(inner(Mlr, MMlr))

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    N = A.shape[0]

    # linear_system = LinearSystem(
    #     A=A, b=b, M=M, Ml=Ml, inner=inner, exact_solution=exact_solution,
    # )

    # sanitize arguments
    maxiter = N if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = numpy.zeros_like(b)

    # get initial residual
    MMlr0, Mlr0, MMlr0_norm = get_residual_and_norm(x0)

    xk = None

    # find common dtype
    dtype = numpy.find_common_type([x0.dtype], [])

    MlAMr = Product(Ml, A, Mr)

    # TODO: reortho
    iter = 0

    resnorms = []

    Mlb = b if Ml is None else Ml @ b
    MMlb = Mlb if M is None else M @ Mlb
    MMlb_norm = numpy.sqrt(inner(Mlb, MMlb))

    # if rhs is exactly(!) zero, return zero solution.
    if MMlb_norm == 0:
        xk = x0 = numpy.zeros_like(b)
        resnorms.append(0.0)
    else:
        # initial relative residual norm
        resnorms.append(MMlr0_norm / MMlb_norm)

    # compute error?
    if exact_solution is not None:
        errnorms = []
        err = exact_solution - x0
        errnorms.append(numpy.sqrt(inner(err, err)))

    # initialize Arnoldi
    arnoldi = Arnoldi(
        MlAMr,
        Mlr0,
        maxiter=maxiter,
        ortho=ortho,
        M=M,
        Mv=MMlr0,
        Mv_norm=MMlr0_norm,
        inner=inner,
    )

    # Givens rotations:
    G = []
    # QR decomposition of Hessenberg matrix via Givens and R
    R = numpy.zeros([maxiter + 1, maxiter], dtype=dtype)
    y = numpy.zeros(maxiter + 1, dtype=dtype)
    # Right hand side of projected system:
    y[0] = MMlr0_norm

    # iterate Arnoldi
    while (
        resnorms[-1] > tol and arnoldi.iter < arnoldi.maxiter and not arnoldi.invariant
    ):
        k = iter = arnoldi.iter
        arnoldi.advance()

        # Copy new column from Arnoldi
        V = arnoldi.V
        R[: k + 2, k] = arnoldi.H[: k + 2, k]

        # Apply previous Givens rotations.
        for i in range(k):
            R[i : i + 2, k] = G[i] @ R[i : i + 2, k]

        # Compute and apply new Givens rotation.
        G.append(givens(R[k : k + 2, k]))
        R[k : k + 2, k] = G[k] @ R[k : k + 2, k]
        y[k : k + 2] = G[k] @ y[k : k + 2]

        yk = y[: k + 1]
        resnorm = abs(y[k + 1])
        xk = None
        # compute error norm if asked for
        if exact_solution is not None:
            xk = _get_xk(yk) if xk is None else xk
            err = exact_solution - xk
            errnorms.append(numpy.sqrt(inner(err, err)))

        rkn = None
        if use_explicit_residual:
            xk = _get_xk(yk) if xk is None else xk
            rkn = get_residual_norm(xk)
            resnorm = rkn

        resnorms.append(resnorm / MMlb_norm)

        # compute explicit residual if asked for or if the updated residual is below the
        # tolerance or if this is the last iteration
        if resnorm / MMlb_norm <= tol:
            # oh really?
            if not use_explicit_residual:
                xk = _get_xk(yk) if xk is None else xk
                rkn = get_residual_norm(xk)
                resnorms[-1] = rkn / MMlb_norm

            # # no convergence?
            # if resnorms[-1] > tol:
            #     # updated residual was below but explicit is not: warn
            #     if (
            #         not explicit_residual
            #         and resnorm / MMlb_norm <= tol
            #     ):
            #         warnings.warn(
            #             "updated residual is below tolerance, explicit residual is NOT!"
            #             f" (upd={resnorm} <= tol={tol} < exp={resnorms[-1]})"
            #         )

        elif iter + 1 == maxiter:
            # no convergence in last iteration -> raise exception
            # (approximate solution can be obtained from exception)
            # store arnoldi?
            if store_arnoldi:
                if M is not None:
                    V, H, P = arnoldi.get()
                else:
                    V, H = arnoldi.get()
            raise ConvergenceError(
                "No convergence in last iteration "
                f"(maxiter: {maxiter}, residual: {resnorms[-1]})."
            )

    # compute solution if not yet done
    if xk is None:
        xk = _get_xk(y[: arnoldi.iter])

    # store arnoldi?
    if store_arnoldi:
        if M is not None:
            V, H, P = arnoldi.get()
        else:
            V, H = arnoldi.get()

    operations = {
        "A": 1 + iter,
        "M": 2 + iter,
        "Ml": 2 + iter,
        "Mr": 1 + iter,
        "inner": 2 + iter + iter * (iter + 1) / 2,
        "axpy": 4 + 2 * iter + iter * (iter + 1) / 2,
    }

    Info = namedtuple("KrylovInfo", ["resnorms", "operations"])

    return xk if resnorms[-1] < tol else None, Info(resnorms, operations)


class _RestartedSolver:
    """Base class for restarted solvers."""

    def __init__(self, Solver, linear_system, max_restarts=0, **kwargs):
        """
        :param max_restarts: the maximum number of restarts. The maximum
          number of iterations is ``(max_restarts+1)*maxiter``.
        """
        # initial approximation will be set by first run of Solver
        self.xk = None

        # work on own copy of args in order to include proper initial guesses
        kwargs = dict(kwargs)

        # append dummy values for first run
        self.resnorms = [numpy.Inf]
        if linear_system.exact_solution is not None:
            self.errnorms = [numpy.Inf]

        # dummy value, gets reset in the first iteration
        tol = None

        restart = 0
        while restart == 0 or (self.resnorms[-1] > tol and restart <= max_restarts):
            try:
                if self.xk is not None:
                    # use last approximate solution as initial guess
                    kwargs.update({"x0": self.xk})

                # try to solve
                sol = Solver(linear_system, **kwargs)
            except utils.ConvergenceError as e:
                # use solver of exception
                sol = e.solver

            # set last approximate solution
            self.xk = sol.xk
            tol = sol.tol

            # concat resnorms / errnorms
            del self.resnorms[-1]
            self.resnorms += sol.resnorms
            if linear_system.exact_solution is not None:
                del self.errnorms[-1]
                self.errnorms += sol.errnorms

            restart += 1

        if self.resnorms[-1] > tol:
            raise utils.ConvergenceError(
                f"No convergence after {max_restarts} restarts.", self
            )


def bound_perturbed_gmres(pseudo, p, epsilon, deltas):
    """Compute GMRES perturbation bound based on pseudospectrum

    Computes the GMRES bound from [SifEM13]_.
    """
    if not numpy.all(numpy.array(deltas) > epsilon):
        raise ArgumentError("all deltas have to be greater than epsilon")

    bound = []
    for delta in deltas:
        # get boundary paths
        paths = pseudo.contour_paths(delta)

        # get vertices on boundary
        vertices = paths.vertices()

        # evaluate polynomial
        supremum = numpy.max(numpy.abs(p(vertices)))

        # compute bound
        bound.append(
            epsilon
            / (delta - epsilon)
            * paths.length()
            / (2 * numpy.pi * delta)
            * supremum
        )
    return bound
