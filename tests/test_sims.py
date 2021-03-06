"""Simulations tests."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from simulations.datagen import simple_triangle, collinear_confounders


@st.composite
def _n_x_support_strat(draw):
    n_x = draw(st.integers(min_value=0, max_value=10))
    support = draw(st.integers(min_value=0, max_value=n_x))
    return (n_x, support)


@given(
    n_x=st.shared(_n_x_support_strat(), key="nxs").map(lambda tup: tup[0]),
    support_size=st.shared(_n_x_support_strat(), key="nxs").map(lambda tup: tup[1]),
    random_state=st.integers(min_value=1, max_value=2**32 - 1),
)
def test_simple_data_generation_deterministic(n_x, support_size, random_state):
    """Test that the simple simulation runs deterministically.

    Samples from two DGPS constructed with same parameters.
    """
    result = simple_triangle(
        alpha=0.3,
        n_x=n_x,
        support_size=support_size,
        random_state=random_state
    )
    repeat = simple_triangle(
        alpha=0.3,
        n_x=n_x,
        support_size=support_size,
        random_state=random_state
    )
    # sample from the dgps and compare samples for equality
    X_res = result.sample(100)
    X_rep = repeat.sample(100)
    same_results = _dicts_of_arrays_equal(X_res, X_rep)
    # if not same_results:
    #     breakpoint()
    assert same_results


@given(
    confounder_dim=st.integers(min_value=1, max_value=100),
    latent_dim=st.integers(min_value=1, max_value=100),
    random_state=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_collinear_data_generation_deterministic(
    confounder_dim, latent_dim, random_state
):
    """Test that the collinear simulation runs deterministically.

    Samples from two DGPS constructed with same parameters.
    """
    result = collinear_confounders(
        true_ate=0.3,
        confounder_dim=confounder_dim,
        latent_dim=latent_dim,
        random_state=random_state,
    )
    repeat = collinear_confounders(
        true_ate=0.3,
        confounder_dim=confounder_dim,
        latent_dim=latent_dim,
        random_state=random_state,
    )
    # sample from the dgps and compare samples for equality
    X_res = result.sample(100)
    X_rep = repeat.sample(100)
    same_results = _dicts_of_arrays_equal(X_res, X_rep)
    assert same_results


def _dicts_of_arrays_equal(d1, d2):
    key_els_equal = [
        np.array_equal(d1[k], d2[k]) for k in list(d1.keys()) + list(d2.keys())
    ]
    return all(key_els_equal)
