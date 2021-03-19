from dataclasses import replace

import numpy as np
from numpy import random

import craystack as cs
import hmm_codec


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    plt.rcParams.update({"text.usetex": True})

    h = hmm_codec.HyperParams(T=1 << 9)
    print('Hyperparameter settings:')
    print(h)
    rng = random.default_rng(1)
    params = hmm_codec.hmm_params_sample(h, rng)
    xs = hmm_codec.hmm_sample(h, params, rng)
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
    ax.yaxis.set_minor_locator(ticker.FixedLocator([1., 2., 3., 4.]))
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

    ax.hlines(y=1, xmin=0, xmax=h.T + 1, color='gray', lw=.5)
    ax.set_xlim(0, h.T)
    ax.set_xlabel('$T$')
    ax.set_ylabel('$l(m)/h(x)$')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=.5)
    plt.tight_layout()
    plt.savefig('perfect_model.pdf')
