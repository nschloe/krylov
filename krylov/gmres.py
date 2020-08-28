from collections import namedtuple

import numpy
import scipy.linalg

from ._helpers import Identity, Product
from .arnoldi import Arnoldi
from .errors import ConvergenceError
from .givens import givens


def multi_matmul(A, b):
    """A @ b for many A, b (i.e., A.shape == (m,n,...), y.shape == (n,...))"""
    return numpy.einsum("ij...,j...->i...", A, b)


def multi_solve_triangular(A, B):
    """This function calls scipy.linalg.solve_triangular for every single A. A
    vectorized version would be much better here.
    """
    A_shape = A.shape
    a = A.reshape(A.shape[0], A.shape[1], -1)
    b = B.reshape(B.shape[0], -1)
    y = []
    for k in range(a.shape[2]):
        if numpy.all(b[:, k] == 0.0):
            y.append(numpy.zeros(b[:, k].shape))
        else:
            y.append(scipy.linalg.solve_triangular(a[:, :, k], b[:, k]))
    y = numpy.array(y).T.reshape([A_shape[0]] + list(A_shape[2:]))
    return y


def gmres(
    A,
    b,
    M=Identity(),
    Ml=Identity(),
    Mr=Identity(),
    inner=lambda x, y: numpy.einsum("i...,i...->...", x.conj(), y),
    exact_solution=None,
    ortho="mgs",
    x0=None,
    U=None,
    tol=1e-5,
    atol=1.0e-15,
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
            yy = multi_solve_triangular(R[:k, :k], y)
            yk = sum(c * v for c, v in zip(yy, V[:-1]))
            return x0 + Mr @ yk
        return x0

    def get_residual_norm(z):
        # \\| M M_l (b-Az)\\|_{M^{-1}}
        return get_residual_and_norm(z)[2]

    def get_residual_and_norm(z):
        # r = M M_l ( b - A z )
        Ml_r = Ml @ (b - A @ z)
        M_Ml_r = M @ Ml_r
        return M_Ml_r, Ml_r, numpy.sqrt(inner(Ml_r, M_Ml_r))

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    N = A.shape[0]

    # sanitize arguments
    maxiter = N if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = numpy.zeros_like(b)

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm = get_residual_and_norm(x0)

    dtype = M_Ml_r0.dtype

    xk = None

    Ml_A_Mr = Product(Ml, A, Mr)

    # TODO: reortho
    k = 0

    resnorms = [M_Ml_r0_norm]

    Ml_b = Ml @ b
    M_Ml_b = M @ Ml_b
    M_Ml_b_norm = numpy.sqrt(inner(Ml_b, M_Ml_b))

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [numpy.sqrt(inner(err, err))]

    # initialize Arnoldi
    arnoldi = Arnoldi(
        Ml_A_Mr,
        Ml_r0,
        maxiter=maxiter,
        ortho=ortho,
        M=M,
        Mv=M_Ml_r0,
        Mv_norm=M_Ml_r0_norm,
        inner=inner,
    )

    # Givens rotations:
    G = []
    # QR decomposition of Hessenberg matrix via Givens and R
    R = numpy.zeros([maxiter + 1, maxiter] + list(b.shape[1:]), dtype=dtype)
    y = numpy.zeros([maxiter + 1] + list(b.shape[1:]), dtype=dtype)
    # Right hand side of projected system:
    y[0] = M_Ml_r0_norm

    # iterate Arnoldi
    k = 0
    criterion = numpy.maximum(tol * M_Ml_b_norm, atol)
    while numpy.any(resnorms[-1] > criterion) and k < maxiter and not arnoldi.invariant:
        V, H = next(arnoldi)

        # Copy new column from Arnoldi
        R[: k + 2, k] = H[: k + 2, k]

        # Apply previous Givens rotations.
        for i in range(k):
            R[i : i + 2, k] = multi_matmul(G[i], R[i : i + 2, k])

        # Compute and apply new Givens rotation.
        G.append(givens(R[k : k + 2, k]))
        R[k : k + 2, k] = multi_matmul(G[k], R[k : k + 2, k])
        y[k : k + 2] = multi_matmul(G[k], y[k : k + 2])

        yk = y[: k + 1]
        resnorm = numpy.abs(y[k + 1])
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

        resnorms.append(resnorm)

        # compute explicit residual if asked for or if the updated residual is below the
        # tolerance or if this is the last iteration
        if numpy.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                xk = _get_xk(yk) if xk is None else xk
                rkn = get_residual_norm(xk)
                resnorms[-1] = rkn

            if numpy.all(resnorms[-1] <= criterion):
                break

            # # no convergence?
            # if resnorms[-1] > tol:
            #     # updated residual was below but explicit is not: warn
            #     if (
            #         not explicit_residual
            #         and resnorm / M_Ml_b_norm <= tol
            #     ):
            #         warnings.warn(
            #             "updated residual is below tolerance, explicit residual is NOT!"
            #             f" (upd={resnorm} <= tol={tol} < exp={resnorms[-1]})"
            #         )

        if k + 1 == maxiter:
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
                f"(maxiter: {maxiter}, residual: {resnorms[-1]:.3e})."
            )

        k += 1

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
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + k + k * (k + 1) / 2,
        "axpy": 4 + 2 * k + k * (k + 1) / 2,
    }

    Info = namedtuple("KrylovInfo", ["resnorms", "operations", "errnorms"])

    return xk if numpy.all(resnorms[-1] < criterion) else None, Info(
        resnorms, operations, errnorms
    )
