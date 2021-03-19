# Based on examples/hmm_em.py, from the Autograd repository.
from __future__ import division, print_function
from dataclasses import replace
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import value_and_grad as vgrad
from functools import partial
from os.path import join, dirname
import string

import craystack as cs
import hmm_codec


def EM(init_params, data, callback=None):
    def EM_update(params):
        natural_params = list(map(np.log, params))
        # E step:
        loglike, E_stats = vgrad(log_partition_function)(natural_params, data)
        if callback: callback(loglike, params)
        return list(map(normalize, E_stats)) # M step

    def fixed_point(f, x0, max_iter=50):
        x1 = f(x0)
        # while different(x0, x1):
        #     x0, x1 = x1, f(x1)
        for _ in range(max_iter):
            x0, x1 = x1, f(x1)
        return x1

    # def different(params1, params2):
    #     allclose = partial(np.allclose, atol=5e-2, rtol=5e-2)
    #     return not all(map(allclose, params1, params2))

    return fixed_point(EM_update, init_params)


def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)
    return a / replace_zeros(a.sum(-1, keepdims=True))


def log_partition_function(natural_params, data):
    if isinstance(data, list):
        return sum(map(partial(log_partition_function, natural_params), data))

    log_pi, log_A, log_B = natural_params

    log_alpha = log_pi
    for y in data:
        log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) + log_B[:,y]

    return logsumexp(log_alpha)


def initialize_hmm_parameters(num_states, num_outputs):
    init_pi = normalize(npr.rand(num_states))
    init_A = normalize(npr.rand(num_states, num_states))
    init_B = normalize(npr.rand(num_states, num_outputs))
    return init_pi, init_A, init_B


def build_dataset(filename, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    encodings = dict(list(map(reversed, enumerate(string.printable))))
    digitize = lambda char: (encodings[char]
                             if char in encodings else len(encodings))
    encode_line = lambda line: np.array(list(map(digitize, line)))
    nonblank_line = lambda line: len(line) > 2

    with open(filename) as f:
        lines = f.readlines()

    encoded_lines = list(map(
        encode_line, list(filter(nonblank_line, lines))[:max_lines]))
    num_outputs = len(encodings) + 1

    return encoded_lines, num_outputs


if __name__ == '__main__':
    np.random.seed(0)
    np.seterr(divide='ignore')
    train_size, test_size, compress_size = 100, 20, 100

    # callback to print log likelihoods during training
    print_loglike = lambda loglike, params: print(
        loglike / train_size,
        log_partition_function(list(map(np.log, params)), test_inputs)
        / test_size)

    # load training data
    lstm_filename = join(dirname(__file__), 'war_and_peace.txt')
    inputs, num_outputs = build_dataset(
        lstm_filename, max_lines=train_size + test_size + compress_size)
    train_inputs, test_inputs, compress_inputs = (
        inputs[:train_size],
        inputs[train_size:train_size + test_size],
        inputs[train_size + test_size:]
    )

    # train with EM
    num_states = 32
    init_params = initialize_hmm_parameters(num_states, num_outputs)
    print('Training hmm_codec with EM...')
    a0, a, b = EM(init_params, train_inputs, print_loglike)

    from matplotlib import pyplot as plt
    from matplotlib import ticker
    plt.rcParams.update({"text.usetex": True})
    h = hmm_codec.HyperParams(num_states, num_outputs, T=1 << 9)
    print('Hyperparameter settings:')
    print(h)
    xs = np.concatenate(compress_inputs)[:h.T]
    params = list(map(partial(hmm_codec.quantized_cdf, h), (a0, a, b)))
    rng = np.random.default_rng(1)
    lengths = []
    hs = []
    message_lengths = []
    l = 4
    while l <= h.T:
        codec = hmm_codec.hmm_codec(replace(h, T=l), params)
        lengths.append(l)
        hs.append(-hmm_codec.hmm_logpmf(h, params, xs[:l]) / l)
        message = cs.base_message(1)
        message = codec.push(message, xs[:l])
        message_lengths.append(len(cs.flatten(message)) * 32 / l)
        l = 2 * l

    fig, ax = plt.subplots(figsize=[2.7,1.8])
    ax.plot(lengths, np.divide(message_lengths, hs), color='black', lw=.5)
    ax.set_yscale('log')
    ax.yaxis.set_minor_locator(ticker.FixedLocator([1., 2., 3., 6., 10.]))
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

    ax.hlines(y=1, xmin=0, xmax=h.T + 1, color='gray', lw=.5)
    ax.set_xlim(0, h.T)
    ax.set_xlabel('$T$')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=.5)
    plt.tight_layout()
    plt.savefig('em_model.pdf')
