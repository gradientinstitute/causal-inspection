"""Test simulations/simple_sim.py"""
import hypothesis as hyp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from simulations import simple_sim


@st.composite
def n_x_support_strat(draw):
    n_x = draw(st.integers(min_value=0, max_value=10))
    support = draw(st.integers(min_value=0, max_value=n_x))
    return (n_x, support)


@given(
    n_x=st.shared(n_x_support_strat(), key="nxs").map(lambda tup: tup[0]),
    support_size=st.shared(n_x_support_strat(), key="nxs").map(lambda tup: tup[1]),
    random_state=st.one_of(st.integers(min_value=0, max_value=2**32 - 1)),
)
def test_data_generation_deterministic(n_x, support_size, random_state):
    result = simple_sim.data_generation(n_x, support_size, random_state)
    repeat = simple_sim.data_generation(n_x, support_size, random_state)
    X_res = result.sample(100)
    X_rep = repeat.sample(100)
    same_results = _dicts_of_arrays_equal(X_res, X_rep)
    assert same_results


def _dicts_of_arrays_equal(d1, d2):
    key_els_equal = [np.array_equal(d1[k], d2[k]) for k in list(d1.keys()) + list(d2.keys())]
    return all(key_els_equal)
