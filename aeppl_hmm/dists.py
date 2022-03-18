from copy import copy
from typing import Sequence

import aesara
import aesara.tensor as at
import numpy as np
from aeppl.abstract import MeasurableVariable
from aeppl.dists import dirac_delta
from aeppl.logprob import _logprob
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Constant
from aesara.tensor.basic import make_vector
from aesara.tensor.random.basic import categorical
from aesara.tensor.random.utils import broadcast_params, normalize_size_param
from aesara.tensor.var import TensorVariable


def non_constant(x):
    x = at.as_tensor_variable(x)
    if isinstance(x, Constant):
        # XXX: This isn't good for `size` parameters, because it could result
        # in `at.get_vector_length` exceptions.
        res = x.type()
        res.tag = copy(res.tag)
        if aesara.config.compute_test_value != "off":
            res.tag.test_value = x.data
        res.name = x.name
        return res
    else:
        return x


def switching_process(
    comp_rvs: Sequence[TensorVariable],
    states: TensorVariable,
):
    """Construct a switching process over arbitrary univariate mixtures and a state sequence.

    This simply constructs a graph of the following form:

        at.stack(comp_rvs)[states, *idx]

    where ``idx`` makes sure that `states` selects mixture components along all
    the other axes.

    Parameters
    ----------
    comp_rvs
        A list containing `RandomVariable` objects for each mixture component.
    states
        The hidden state sequence.  It should have a number of states
        equal to the size of `comp_dists`.

    """  # noqa: E501

    states = at.as_tensor(states, dtype=np.int64)
    comp_rvs_bcast = at.broadcast_arrays(*[at.as_tensor(rv) for rv in comp_rvs])
    M_rv = at.stack(comp_rvs_bcast)
    indices = (states,)
    indices += tuple(at.arange(d) for d in tuple(M_rv.shape)[1:])
    rv_var = M_rv[indices]
    return rv_var


def poisson_zero_process(mu=None, states=None, srng=None, **kwargs):
    """A Poisson-Dirac-delta (at zero) mixture process.

    The first mixture component (at index 0) is the Dirac-delta at zero, and
    the second mixture component is the Poisson random variable.

    Parameters
    ----------
    mu: tensor
        The Poisson rate(s)
    states: tensor
        A vector of integer 0-1 states that indicate which component of
        the mixture is active at each point/time.
    """
    mu = at.as_tensor_variable(mu)
    states = at.as_tensor_variable(states)

    # NOTE: This creates distributions that are *not* part of a `Model`
    return switching_process(
        [dirac_delta(at.as_tensor(0, dtype=np.int64)), srng.poisson(mu)],
        states,
        **kwargs
    )


class DiscreteMarkovChainFactory(OpFromGraph):
    # Add `RandomVariable`-like "metadata"
    ndim_supp = 1
    ndims_params = (3, 1)


MeasurableVariable.register(DiscreteMarkovChainFactory)


def create_discrete_mc_op(rng, size, Gammas, gamma_0):
    """Construct a `DiscreteMarkovChainFactory` `Op`.

    This returns a `Scan` that performs the follow:

        states[0] = categorical(gamma_0)
        for t in range(1, N):
            states[t] = categorical(Gammas[t, state[t-1]])

    The Aesara graph representing the above is wrapped in an `OpFromGraph` so
    that we can easily assign it a specific log-probability.

    TODO: Eventually, AePPL should be capable of parsing more sophisticated
    `Scan`s and producing nearly the same log-likelihoods, and the use of
    `OpFromGraph` will no longer be necessary.

    """

    # Again, we need to preserve the length of this symbolic vector, so we do
    # this.
    size_param = make_vector(
        *[non_constant(size[i]) for i in range(at.get_vector_length(size))]
    )
    size_param.name = "size"

    # We make shallow copies so that unwanted ancestors don't appear in the
    # graph.
    Gammas_param = non_constant(Gammas).type()
    Gammas_param.name = "Gammas_param"

    gamma_0_param = non_constant(gamma_0).type()
    gamma_0_param.name = "gamma_0_param"

    bcast_Gammas_param, bcast_gamma_0_param = broadcast_params(
        (Gammas_param, gamma_0_param), (3, 1)
    )

    # Sample state 0 in each state sequence
    state_0 = categorical(
        bcast_gamma_0_param,
        size=tuple(size_param) + tuple(bcast_gamma_0_param.shape[:-1]),
        # size=at.join(0, size_param, bcast_gamma_0_param.shape[:-1]),
        rng=rng,
    )

    N = bcast_Gammas_param.shape[-3]
    states_shape = tuple(state_0.shape) + (N,)

    bcast_Gammas_param = at.broadcast_to(
        bcast_Gammas_param, states_shape + tuple(bcast_Gammas_param.shape[-2:])
    )

    def loop_fn(n, state_nm1, Gammas_inner, rng):
        gamma_t = Gammas_inner[..., n, :, :]
        idx = tuple(at.ogrid[[slice(None, d) for d in tuple(state_0.shape)]]) + (
            state_nm1.T,
        )
        gamma_t = gamma_t[idx]
        state_n = categorical(gamma_t, rng=rng)
        return state_n.T

    res, _ = aesara.scan(
        loop_fn,
        outputs_info=[{"initial": state_0.T, "taps": [-1]}],
        sequences=[at.arange(N)],
        non_sequences=[bcast_Gammas_param, rng],
        # strict=True,
    )

    return DiscreteMarkovChainFactory(
        [size_param, Gammas_param, gamma_0_param],
        [res.T],
        inline=True,
        on_unused_input="ignore",
    )


def discrete_markov_chain(
    Gammas: TensorVariable, gamma_0: TensorVariable, size=None, rng=None, **kwargs
):
    """Construct a first-order discrete Markov chain distribution.

    This characterizes vector random variables consisting of state indicator
    values (i.e. ``0`` to ``M - 1``) that are driven by a discrete Markov chain.


    Parameters
    ----------
    Gammas
        An array of transition probability matrices.  `Gammas` takes the
        shape ``... x N x M x M`` for a state sequence of length ``N`` having
        ``M``-many distinct states.  Each row, ``r``, in a transition probability
        matrix gives the probability of transitioning from state ``r`` to each
        other state.
    gamma_0
        The initial state probabilities.  The last dimension should be length ``M``,
        i.e. the number of distinct states.
    """
    gamma_0 = at.as_tensor_variable(gamma_0)

    assert Gammas.ndim >= 3

    Gammas = at.as_tensor_variable(Gammas)

    size = normalize_size_param(size)

    if rng is None:
        rng = aesara.shared(np.random.RandomState(), borrow=True)

    DiscreteMarkovChainOp = create_discrete_mc_op(rng, size, Gammas, gamma_0)
    rv_var = DiscreteMarkovChainOp(size, Gammas, gamma_0)

    testval = kwargs.pop("testval", None)

    if testval is not None:
        rv_var.tag.test_value = testval

    return rv_var


@_logprob.register(DiscreteMarkovChainFactory)
def discrete_mc_logp(op, states, *dist_params, **kwargs):
    r"""Create a Aesara graph that computes the log-likelihood for a discrete Markov chain.

    This is the log-likelihood for the joint distribution of states, :math:`S_t`, conditional
    on state samples, :math:`s_t`, given by the following:

    .. math::

        \int_{S_0} P(S_1 = s_1 \mid S_0) dP(S_0) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    The first term (i.e. the integral) simply computes the marginal :math:`P(S_1 = s_1)`, so
    another way to express this result is as follows:

    .. math::

        P(S_1 = s_1) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    """  # noqa: E501

    (states,) = states
    _, Gammas, gamma_0 = dist_params[: len(dist_params) - len(op.shared_inputs)]

    Gammas = at.shape_padleft(Gammas, states.ndim - (Gammas.ndim - 2))

    # Multiply the initial state probabilities by the first transition
    # matrix by to get the marginal probability for state `S_1`.
    # The integral that produces the marginal is essentially
    # `gamma_0.dot(Gammas[0])`
    Gamma_1 = Gammas[..., 0:1, :, :]
    gamma_0 = at.expand_dims(gamma_0, (-3, -1))
    P_S_1 = at.sum(gamma_0 * Gamma_1, axis=-2)

    # The `tt.switch`s allow us to broadcast the indexing operation when
    # the replication dimensions of `states` and `Gammas` don't match
    # (e.g. `states.shape[0] > Gammas.shape[0]`)
    S_1_slices = tuple(
        slice(
            at.switch(at.eq(P_S_1.shape[i], 1), 0, 0),
            at.switch(at.eq(P_S_1.shape[i], 1), 1, d),
        )
        for i, d in enumerate(states.shape)
    )
    S_1_slices = (tuple(at.ogrid[S_1_slices]) if S_1_slices else tuple()) + (
        states[..., 0:1],
    )
    logp_S_1 = at.log(P_S_1[S_1_slices]).sum(axis=-1)

    # These are slices for the extra dimensions--including the state
    # sequence dimension (e.g. "time")--along which which we need to index
    # the transition matrix rows using the "observed" `states`.
    trans_slices = tuple(
        slice(
            at.switch(at.eq(Gammas.shape[i], 1), 0, 1 if i == states.ndim - 1 else 0),
            at.switch(at.eq(Gammas.shape[i], 1), 1, d),
        )
        for i, d in enumerate(states.shape)
    )
    trans_slices = (tuple(at.ogrid[trans_slices]) if trans_slices else tuple()) + (
        states[..., :-1],
    )

    # Select the transition matrix row of each observed state; this yields
    # `P(S_t | S_{t-1} = s_{t-1})`
    P_S_2T = Gammas[trans_slices]

    obs_slices = tuple(slice(None, d) for d in P_S_2T.shape[:-1])
    obs_slices = (tuple(at.ogrid[obs_slices]) if obs_slices else tuple()) + (
        states[..., 1:],
    )
    logp_S_1T = at.log(P_S_2T[obs_slices])

    res = logp_S_1 + at.sum(logp_S_1T, axis=-1)
    res.name = "DiscreteMarkovChain_logp"

    if kwargs.get("sum", False):
        res = res.sum()

    return res
