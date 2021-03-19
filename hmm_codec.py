from functools import partial
from dataclasses import dataclass

import numpy as np
from numpy import random

from scipy.special import logsumexp

import craystack as cs


def exp2(x):
    return 1 << x  # Assumes x is an integer

ln2 = np.log(2)

@dataclass
class HyperParams:
    latent_K: int = 64
    obs_K: int = 64
    # Can't set quant_prec higher than 21 because at some point we multiply
    # three uints and we need the product to fit in a uint64
    quant_prec: int = 21
    T: int = 1 << 14

def quantized_cdf(h: HyperParams, masses):
    return np.concatenate([
        np.zeros(masses.shape[:-1] + (1,)),
        np.round(exp2(h.quant_prec)
                 * np.cumsum(masses, -1))], -1).astype('uint64')

def hmm_params_sample(h: HyperParams, rng: random.Generator):
    alpha_latent, alpha_obs = np.ones(h.latent_K), np.ones(h.obs_K)
    a0 = quantized_cdf(h, rng.dirichlet(alpha_latent))
    a  = quantized_cdf(h, rng.dirichlet(alpha_latent, h.latent_K))
    b  = quantized_cdf(h, rng.dirichlet(alpha_obs,    h.latent_K))
    return a0, a, b

def quantized_cdf_to_mass(h: HyperParams, cdf):
    return np.diff(cdf) / exp2(h.quant_prec)

def hmm_sample(h: HyperParams, params, rng: random.Generator):
    a0, a, b = map(partial(quantized_cdf_to_mass, h), params)
    xs = []
    for t in range(h.T):
        z = rng.choice(h.latent_K, p=a[z] if t else a0)
        xs.append(rng.choice(h.obs_K, p=b[z]))
    return np.array(xs)

def hmm_logpmf(h: HyperParams, params, xs):
    # Based on Matt Johnson's hmm_em.py Autograd example
    err_settings = np.seterr(divide='ignore') # Suppress div by zero warning
    log_a0, log_a, log_b = map(np.log, map(
        partial(quantized_cdf_to_mass, h), params))
    np.seterr(**err_settings)
    log_alpha = log_a0
    for x in xs:
        log_alpha = logsumexp(log_alpha[:, None] + log_a, 0) + log_b[:, x]
    return logsumexp(log_alpha) / ln2

def SSM(priors, likelihoods, posterior):
    # priors = [p(z_1), p(z_2 | z_1), ..., p(z_T | z_{T-1})]
    # likelihoods = [p(x_1 | z_1), ..., p(x_T | z_T)]
    # [lambda z_{t+1}: Q(z_t | x_{1:t}, z_{t+1}) for t in range(T)]
    post_init_state, post_update = posterior
    def push(message, xs):
        post_codecs = []
        post_state = post_init_state
        for t, x in enumerate(xs):  # Forward inference pass
            post_state, post_codec = post_update(t, post_state, x)
            post_codecs.append(post_codec)
        message, z_next = post_codecs[-1].pop(message)
        for t in range(len(priors) - 1, 0, -1):  # Backward encoding pass
            message = likelihoods[t](z_next).push(message, xs[t])
            message, z = post_codecs[t - 1](z_next).pop(message)
            message = priors[t](z).push(message, z_next)
            z_next = z
        message = likelihoods[0](z_next).push(message, xs[0])
        message = priors[0].push(message, z_next)
        return message
    def pop(message):
        xs = []
        message, z_next = priors     [0]        .pop(message)
        message, x      = likelihoods[0](z_next).pop(message)
        post_state, post_codec = post_update(0, post_init_state, x)
        xs.append(x)
        for t in range(1, len(priors)):  # Forward decoding pass
            z = z_next
            message, z_next = priors     [t](z)     .pop (message)
            message         = post_codec    (z_next).push(message, z)
            message, x      = likelihoods[t](z_next).pop (message)
            post_state, post_codec = post_update(t, post_state, x)
            xs.append(x)
        message = post_codec.push(message, z_next)
        return message, xs
    return cs.Codec(push, pop)

def Categorical(h: HyperParams, cdf):
    def enc_statfun(x):
        return cdf[x], cdf[x + 1] - cdf[x]
    def dec_statfun(cf):
        assert cf.shape == (1,)
        return (np.searchsorted(cdf, cf, 'right') - 1)[0]
    return cs.NonUniform(enc_statfun, dec_statfun, h.quant_prec)

def HMM(h: HyperParams, params):
    a0, a, b = params
    a_mass, b_mass = np.diff(a), np.diff(b)
    priors = [(lambda z: Categorical(h, a[z])) if t else Categorical(h, a0)
              for t in range(h.T)]
    likelihoods = [(lambda z: Categorical(h, b[z])) for _ in range(h.T)]
    def post_update(t, alpha, x):
        # alpha = p(z_t | x_{1:t-1})
        mixing_coeffs = alpha * b_mass[:, x]
        if t < h.T - 1:
            alpha = np.diff(np.dot(mixing_coeffs, a) // np.sum(mixing_coeffs))
            def z_codec(z_next):
                masses = mixing_coeffs * a_mass[:, z_next]
                return Categorical(h, quantized_cdf(h, masses / np.sum(masses)))
        else:
            alpha = None
            z_codec = Categorical(
                h, quantized_cdf(h, mixing_coeffs / np.sum(mixing_coeffs)))
        return alpha, z_codec
    return SSM(priors, likelihoods, (np.diff(a0), post_update))


if __name__ == '__main__':
    h = HyperParams()
    print('Hyperparameter settings:')
    print(h)
    rng = random.default_rng(1)
    params = hmm_params_sample(h, rng)
    xs = hmm_sample(h, params, rng)
    print('h(x) = {:.2f} bits/symbol.'.format(- hmm_logpmf(h, params, xs)))
    codec = HMM(h, params)
    message = cs.base_message(1)
    message = codec.push(message, xs)
    print('Compression rate: {:.2f} bits/symbol.'.format(
          len(cs.flatten(message)) * 32))
    message, xs_decoded = codec.pop(message)
    assert np.all(xs == xs_decoded)
    print('Decoded OK!')
