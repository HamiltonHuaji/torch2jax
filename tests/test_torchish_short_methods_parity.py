import numpy as np
import pytest
import torch
import jax.numpy as jnp

from torch2jax import Torchish


RTOL = 1e-5
ATOL = 1e-6


def _to_numpy(x):
    if isinstance(x, Torchish):
        return np.asarray(x.value)
    if isinstance(x, torch.Tensor):
        # torch may return tensors with conjugate bit set (e.g. `conj`),
        # which require explicit resolution before NumPy conversion.
        return x.detach().resolve_conj().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    return x


def _assert_close(actual, expected, rtol=RTOL, atol=ATOL):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_close(a, e, rtol=rtol, atol=atol)
        return

    a = _to_numpy(actual)
    e = _to_numpy(expected)

    if isinstance(a, (bool, np.bool_)) and isinstance(e, (bool, np.bool_)):
        assert bool(a) == bool(e)
        return

    if np.isscalar(a) and np.isscalar(e):
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)
        return

    if getattr(a, "dtype", None) == np.bool_ or getattr(e, "dtype", None) == np.bool_:
        np.testing.assert_array_equal(a, e)
    else:
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)


def _torch_arg(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x)
    return x


def _torchish_arg(x):
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    return x


def _run_method(method_name, x, *args, **kwargs):
    tx = torch.tensor(x)
    jx = Torchish(jnp.array(x))

    t_args = [_torch_arg(a) for a in args]
    j_args = [_torchish_arg(a) for a in args]

    t_out = getattr(tx, method_name)(*t_args, **kwargs)
    j_out = getattr(jx, method_name)(*j_args, **kwargs)
    return t_out, j_out


@pytest.mark.parametrize(
    "method_name,input_x",
    [
        ("ceil", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("floor", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("frac", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("cosh", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("sinh", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("tan", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("log", np.array([0.2, 0.4, 1.0, 2.0], dtype=np.float32)),
        ("log10", np.array([0.2, 0.4, 1.0, 2.0], dtype=np.float32)),
        ("log1p", np.array([0.2, 0.4, 1.0, 2.0], dtype=np.float32)),
        ("log2", np.array([0.2, 0.4, 1.0, 2.0], dtype=np.float32)),
        ("neg", np.array([-1.2, -0.1, 0.2, 1.9], dtype=np.float32)),
        ("sign", np.array([-1.2, -0.1, 0.0, 1.9], dtype=np.float32)),
        ("sgn", np.array([-1.2, -0.1, 0.0, 1.9], dtype=np.float32)),
        ("conj", np.array([1 + 2j, -3 + 4j], dtype=np.complex64)),
        ("ravel", np.arange(12, dtype=np.float32).reshape(3, 4)),
    ],
)
def test_torchish_short_unary_method_parity(method_name, input_x):
    t_out, j_out = _run_method(method_name, input_x)
    _assert_close(j_out, t_out)


@pytest.mark.parametrize(
    "method_name,input_x,other",
    [
        ("fmod", np.array([5.5, -4.0, 2.5], dtype=np.float32), np.array([2.0, 3.0, 2.0], dtype=np.float32)),
        ("fmax", np.array([1.0, 7.0, -2.0], dtype=np.float32), np.array([2.0, 6.0, -3.0], dtype=np.float32)),
        ("fmin", np.array([1.0, 7.0, -2.0], dtype=np.float32), np.array([2.0, 6.0, -3.0], dtype=np.float32)),
        ("eq", np.array([1, 2, 3], dtype=np.int32), np.array([1, 0, 3], dtype=np.int32)),
        ("ne", np.array([1, 2, 3], dtype=np.int32), np.array([1, 0, 3], dtype=np.int32)),
        ("ge", np.array([1, 2, 3], dtype=np.int32), np.array([1, 3, 3], dtype=np.int32)),
        ("gt", np.array([1, 2, 3], dtype=np.int32), np.array([1, 3, 3], dtype=np.int32)),
        ("le", np.array([1, 2, 3], dtype=np.int32), np.array([1, 3, 3], dtype=np.int32)),
        ("lt", np.array([1, 2, 3], dtype=np.int32), np.array([1, 3, 3], dtype=np.int32)),
        ("hypot", np.array([3.0, 5.0, 8.0], dtype=np.float32), np.array([4.0, 12.0, 15.0], dtype=np.float32)),
        ("gcd", np.array([12, 18, 20], dtype=np.int32), np.array([8, 24, 6], dtype=np.int32)),
        ("lcm", np.array([3, 4, 5], dtype=np.int32), np.array([2, 6, 10], dtype=np.int32)),
    ],
)
def test_torchish_short_binary_method_parity(method_name, input_x, other):
    t_out, j_out = _run_method(method_name, input_x, other)
    _assert_close(j_out, t_out)


def test_torchish_short_linear_algebra_method_parity():
    rng = np.random.default_rng(0)

    x_dot = rng.normal(size=(8,)).astype(np.float32)
    y_dot = rng.normal(size=(8,)).astype(np.float32)
    for method in ["dot", "inner", "vdot"]:
        t_out, j_out = _run_method(method, x_dot, y_dot)
        _assert_close(j_out, t_out)

    x_outer = rng.normal(size=(4,)).astype(np.float32)
    y_outer = rng.normal(size=(3,)).astype(np.float32)
    t_out, j_out = _run_method("outer", x_outer, y_outer)
    _assert_close(j_out, t_out)

    x_mm = rng.normal(size=(3, 4)).astype(np.float32)
    y_mm = rng.normal(size=(4, 5)).astype(np.float32)
    t_out, j_out = _run_method("mm", x_mm, y_mm)
    _assert_close(j_out, t_out)

    x_mv = rng.normal(size=(3, 4)).astype(np.float32)
    y_mv = rng.normal(size=(4,)).astype(np.float32)
    t_out, j_out = _run_method("mv", x_mv, y_mv)
    _assert_close(j_out, t_out)

    x_bmm = rng.normal(size=(2, 3, 4)).astype(np.float32)
    y_bmm = rng.normal(size=(2, 4, 5)).astype(np.float32)
    t_out, j_out = _run_method("bmm", x_bmm, y_bmm)
    _assert_close(j_out, t_out)


@pytest.mark.parametrize(
    "method_name,input_x,method_args,method_kwargs",
    [
        ("roll", np.arange(8, dtype=np.float32), (2,), {}),
        ("rot90", np.arange(9, dtype=np.float32).reshape(3, 3), (), {"k": 1, "dims": (0, 1)}),
        ("tile", np.arange(6, dtype=np.float32).reshape(2, 3), ((2, 1),), {}),
        ("trace", np.arange(9, dtype=np.float32).reshape(3, 3), (), {}),
        ("tril", np.arange(16, dtype=np.float32).reshape(4, 4), (), {"diagonal": -1}),
        ("triu", np.arange(16, dtype=np.float32).reshape(4, 4), (), {"diagonal": 1}),
        ("diag", np.arange(5, dtype=np.float32), (), {"diagonal": 0}),
    ],
)
def test_torchish_short_shape_method_parity(method_name, input_x, method_args, method_kwargs):
    t_out, j_out = _run_method(method_name, input_x, *method_args, **method_kwargs)
    _assert_close(j_out, t_out)


@pytest.mark.parametrize(
    "method_name,input_x",
    [
        ("bool", np.array([0.0, -1.0, 2.5], dtype=np.float32)),
        ("byte", np.array([0.0, 1.0, 2.0], dtype=np.float32)),
        ("short", np.array([0.0, -1.0, 2.0], dtype=np.float32)),
        ("int", np.array([0.0, -1.0, 2.0], dtype=np.float32)),
        ("long", np.array([0.0, -1.0, 2.0], dtype=np.float32)),
        ("half", np.array([0.1, -1.5, 2.25], dtype=np.float32)),
        ("char", np.array([0.0, -1.0, 2.0], dtype=np.float32)),
    ],
)
def test_torchish_short_cast_method_parity(method_name, input_x):
    t_out, j_out = _run_method(method_name, input_x)
    _assert_close(j_out, t_out)


def test_torchish_short_inplace_method_parity():
    x = np.array([-1.7, -0.2, 0.0, 2.4], dtype=np.float32)

    t = torch.tensor(x)
    j = Torchish(jnp.array(x))

    t_ret = t.ceil_()
    j_ret = j.ceil_()
    assert t_ret is t
    assert j_ret is j
    _assert_close(j, t)

    t2 = torch.tensor(x)
    j2 = Torchish(jnp.array(x))
    t2.sign_()
    j2.sign_()
    _assert_close(j2, t2)

    t3 = torch.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    j3 = Torchish(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))
    t3.triu_(diagonal=1)
    j3.triu_(diagonal=1)
    _assert_close(j3, t3)

    t4 = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    j4 = Torchish(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    t4.zero_()
    j4.zero_()
    _assert_close(j4, t4)
