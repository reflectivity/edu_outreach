import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from scipy.optimize import curve_fit

import _fig_params

from refnx.reflect import Structure, SLD, reflectivity
import emcee

np.random.seed(2)

c = _fig_params.colors

print(c)

def kine():
    # kinematic approximation for reflectivity

    fig = plt.figure(constrained_layout=True, figsize=(10, 7.5))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[1, :])

    # SLD plot
    z = np.linspace(-20, 20, 201)
    rho = np.zeros_like(z)
    rho[np.where(z >= 0)] = 2.074e-6

    ax1.plot(z, rho, c=c[0])
    ax1.set_xlabel(r'$z$/Å')
    ax1.set_ylabel(r'$\rho(z)$/Å$^{-2}$')
    ax1.text(0.05, 0.95, '(a)', horizontalalignment='left',
             verticalalignment='top', transform=ax1.transAxes)

    # first differential of rho
    rho_dash = np.zeros_like(z)
    rho_dash[z==0] = 2.074e-6

    ax2.plot(z, rho_dash, c=c[0])
    ax2.set_xlabel(r"$z$/Å")
    ax2.set_ylabel(r"$\rho'(z)$/Å$^{-3}$")
    ax2.text(0.05, 0.95, '(b)', horizontalalignment='left',
             verticalalignment='top', transform=ax2.transAxes)

    # plot kinematic reflectivity
    q = np.linspace(0.002, 0.05, 500)
    kinematic_r = 16 * np.pi ** 2 * 2.074e-6 ** 2 / (q ** 4)

    ax3.plot(q, kinematic_r, c=c[0])
    ax3.axhline(1, c=c[1])
    ax3.set_yscale('log')
    ax3.set_xlabel(r"$q$/Å$^{-1}$")
    ax3.set_ylabel(r"$R(q)$")
    ax3.set_yticks([10, 1, 0.1, 0.01, 0.001, 0.0001])
    ax3.set_yticklabels(['$10^1$', '$1$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])
    ax3.text(0.975, 0.95, '(c)', horizontalalignment='right',
             verticalalignment='top', transform=ax3.transAxes)
    ax3.set_xlim(0, 0.05)

    plt.savefig('kine.pdf')
    plt.close()


def dyna():
    # a comparison of dynamical + kinematic approaches.
    fig = plt.figure(constrained_layout=True, figsize=(10, 7.5/2))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax3 = fig.add_subplot(spec[0, :])

    # kinematic reflectivity
    q = np.linspace(0.002, 0.05, 500)
    kinematic_r = 16 * np.pi ** 2 * 2.074e-6 ** 2 / (q ** 4)
    ax3.plot(q, kinematic_r, c=c[0])

    # dynamical reflectivity
    q2 = np.linspace(0, 0.05, 1000)
    layers = np.array([[0, 0, 0, 0,], [0, 2.074, 0, 0]])

    ax3.plot(q2, reflectivity(q2, layers, dq=0), c=c[2], zorder=10)
    ax3.axhline(1, c=c[1])
    ax3.set_yscale('log')
    ax3.set_xlabel(r"$q$/Å$^{-1}$")
    ax3.set_ylabel(r"$R(q)$")
    ax3.set_yticks([10, 1, 0.1, 0.01, 0.001, 0.0001])
    ax3.set_yticklabels(['$10^1$', '$1$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])
    ax3.set_xlim(0, 0.05)

    plt.savefig('dyna.pdf')
    plt.close()


def likelihood():
    def abeles_build(q, thick, sld1, sld2):
        layers = np.array(
            [[0, 0, 0, 0],
             [thick, sld1, 0, 0],
             [0, sld2, 0, 0]]
        )
        return reflectivity(q, layers, dq=0.0)

    q2 = np.logspace(-3, -0.5, 40)
    layers = np.array(
        [[0,  0,     0, 0],
         [40, 6.335, 0, 0],
         [0,  2.074, 0, 0]]
    )

    r = reflectivity(q2, layers, dq=0)
    r += np.abs((np.random.random((q2.size)) - 0.5) * q2 * r)
    dr = np.abs(1/r) * 1e-9

    popt, pcov = curve_fit(abeles_build, q2, r, sigma=dr, p0=[40, 6.335, 2.074])

    fig = plt.figure(constrained_layout=True, figsize=(10, 7.5/2))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax3 = fig.add_subplot(spec[0, :])
    q3 = np.logspace(-3, -0.5, 1000)

    rr = np.random.randn(len(popt)) * 10
    rr[-1] = 0

    rm = abeles_build(q2, *popt)
    rm2 = abeles_build(q2, *(np.array(popt) + rr))
    Lm = -0.5 * np.sum(np.square((r-rm)/dr) + np.log(2 * np.pi * dr))
    Lm2 = -0.5 * np.sum(np.square((r-rm2)/dr) + np.log(2 * np.pi * dr))

    rm = abeles_build(q3, *popt)
    rm2 = abeles_build(q3, *(np.array(popt) + rr))

    ax3.errorbar(q2, r, dr, c=c[0], zorder=10, marker='o', ls='')
    ax3.plot(q3, rm, c=c[1], zorder=10, label='{:.2e}'.format(Lm))
    ax3.plot(q3, rm2, c=c[2], zorder=10, label='{:.2e}'.format(Lm2))

    ax3.set_yscale('log')
    ax3.set_xlabel(r"$q$/Å$^{-1}$")
    ax3.set_ylabel(r"$R(q)$")
    ax3.legend(title=r"$\ln\;{\hat{L}}$")
    ax3.set_xlim(0, 0.3)

    plt.savefig('likelihood.pdf')
    plt.close()


def ackley():
    def mut(p, b, km):
        m = np.zeros_like(p)
        R = np.random.randint(p.shape[1], size=(2, p.shape[1]))
        for j in range(p.shape[1]):
            m[:, j] = b + km * (p[:, R[0, j]] - p[:, R[1, j]])
        return m

    def recomb(p, m, kr):
        o = np.array(p)
        rand = np.random.rand(p.shape[0], p.shape[1])
        o[rand < kr] = m[rand < kr]
        return o


    def sel(p, o, f):
        new_p = np.array(p)
        for j in range(p.shape[1]):
            p_fom = f(p[:, j])
            o_fom = f(o[:, j])
            if o_fom > p_fom:
                new_p[:, j] = o[:, j]
        return new_p

    def differential_evolution(population, f, km, kr, bounds, max_iter):
        history = np.array([population])
        best = population[:, np.argmin(f(population))]
        i = 0
        while i < max_iter:
            mutant = mut(population, best, km)
            offspring = recomb(population, mutant, kr)
            offspring[
                np.where(offspring >= bounds[1])
                or np.where(offspring < bounds[0])
            ] = np.random.uniform(bounds[0], bounds[1], 1)
            selected = sel(population, offspring, f)
            history = np.append(history, selected)
            history = np.reshape(
                history, (i + 2, population.shape[0], population.shape[1])
            )
            population = np.array(selected)
            best = population[:, np.argmin(f(population))]
            i += 1
        return history

    startx1 = np.random.uniform(-40, 40, 8)
    startx2 = np.random.uniform(-40, 40, 8)

    def ackley(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        first = -a * np.exp(
            -b * np.sqrt(1 / 2 * np.sum(np.square(x), axis=0))
        )
        second = np.exp(1 / 2 * np.sum(np.cos(c * x), axis=0))
        return -1 * (first - second + a + np.exp(1))


    route = differential_evolution(
        np.array([startx1, startx2]),
        ackley,
        0.5,
        0.5,
        [-40, 40],
        100,
    )

    xs = np.linspace(-40.0, 40.0, 50)
    ys = np.linspace(-40.0, 40.0, 50)
    es = np.zeros((xs.size, ys.size))
    for i in range(xs.size):
        for j in range(ys.size):
            es[i, j] = ackley(np.array([xs[i], ys[j]]))

    fig = plt.figure(constrained_layout=True, figsize=(10, 3.304*2.421))
    ax = fig.add_subplot(111)
    im = ax.contourf(ys, xs, es, 100, cmap="Blues")
    for i in range(route.shape[2]):
        ax.plot(
            route[:, 0, i],
            route[:, 1, i],
            marker="o",
            ms=4,
            c=c[i + 1],
        )
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_2$")
    plt.colorbar(im, label=r"$F(\mathbf{x})$")

    plt.tight_layout()
    plt.savefig("ackley.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def mcmc():
    def gaussianfit(x, a1, a2, b1, b2):
        return a1 * norm.pdf(x - b1) + a2 * norm.pdf(x - b2)


    def gaussian(x, a1):
        return a1[0] * norm.pdf(x - a1[2]) + a1[1] * norm.pdf(x - a1[3])

    x = np.linspace(-5, 8, 25)
    y = gaussian(x, [0.5, 1, 3, 0]) + np.random.randn(
        25
    ) * 0.05 * gaussian(x, [0.5, 1, 3, 0])
    dy = y * 0.2

    popt, pcov = curve_fit(gaussianfit, x, y, sigma=dy, p0=[0.5, 1, 3, 0])

    x2 = np.linspace(-5, 8, 2500)

    theta1 = np.zeros((4))
    guess1 = gaussian(x, popt)

    def log_likelihood(theta, x, y, dy):
        model = gaussian(x, theta)
        sigma2 = dy ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    pos = popt + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x, y, dy))
    sampler.run_mcmc(pos, 1000, progress=True)

    history = sampler.get_chain(discard=100, thin=1, flat=True)

    plt.figure(constrained_layout=True, figsize=(10, 5.1625*2.2))
    gs = gridspec.GridSpec(3, 2)

    ax1 = plt.subplot(gs[2, :])
    ax1.errorbar(
        x, y, marker="o", ls="", yerr=dy, c=c[0], zorder=10
    )

    choice = np.random.randint(low=0, high=history.shape[0], size=100)

    for i in choice:
        ax1.plot(
            x2,
            gaussian(
                x2,
                [
                    history[i, 0],
                    history[i, 1],
                    history[i, 2],
                    history[i, 3],
                ],
            ),
            alpha=0.07,
            c=c[2],
        )

    ax1.plot(x2, gaussian(x2, popt), c=c[1])

    ax2 = plt.subplot(gs[0, 0])
    ax2.hist(history[:, 0], bins=20, density=True, histtype="stepfilled")
    ax3 = plt.subplot(gs[0, 1])
    ax3.hist(history[:, 1], bins=20, density=True, histtype="stepfilled")
    ax4 = plt.subplot(gs[1, 0])
    ax4.hist(history[:, 2], bins=20, density=True, histtype="stepfilled")
    ax5 = plt.subplot(gs[1, 1])
    ax5.hist(history[:, 3], bins=20, density=True, histtype="stepfilled")
    ax2.set_ylabel(r"$p(\theta_1)$")
    ax3.set_ylabel(r"$p(\theta_2)$")
    ax4.set_ylabel(r"$p(\theta_3)$")
    ax5.set_ylabel(r"$p(\theta_4)$")
    ax2.set_xlabel(r"$\theta_1$")
    ax3.set_xlabel(r"$\theta_2$")
    ax4.set_xlabel(r"$\theta_3$")
    ax5.set_xlabel(r"$\theta_4$")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylim(0, 0.55)

    plt.tight_layout()
    plt.savefig("mcmc.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def roughness():
    sld1 = SLD(0, 0)
    sld2 = SLD(64.405, 0) #ni
    sld3 = SLD(35.799, 0) #ti

    layer1 = sld1(0, 0)
    layer3 = sld3(10, 0)

    fig = plt.figure(constrained_layout=True, figsize=(10, 7.5/2))
    gs = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs[0:2])
    ax2 = plt.subplot(gs[2])
    for i in range(3, 9, 2):
        layer2 = sld2(10, i)
        structure = Structure(layer1 | layer2 | layer3)

        ax1.plot(*structure.sld_profile())
        ax2.plot(*structure.sld_profile())
    ax2.set_ylim(30, 40)
    ax2.set_xlim(5, 15)
    ax1.axhline(sld3.real, color='k', alpha=0.3)
    ax2.axhline(sld3.real, color='k', alpha=0.3)
    ax1.set_xlabel(r'$z$/Å')
    ax1.set_ylabel(r'$\rho(z)$/Å$^{-2}$')
    ax1.text(0.025, 0.95, '(a)', horizontalalignment='left',
             verticalalignment='top', transform=ax1.transAxes)
    ax2.set_xlabel(r'$z$/Å')
    ax2.set_ylabel(r'$\rho(z)$/Å$^{-2}$')
    ax2.text(0.025, 0.95, '(b)', horizontalalignment='left',
             verticalalignment='top', transform=ax2.transAxes)
    plt.tight_layout()
    plt.savefig("roughness.pdf")
    plt.close()

# produce the graphs
kine()
dyna()
roughness()
likelihood()
ackley()
mcmc()
