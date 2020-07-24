import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import _fig_params
from refnx.reflect import Structure, SLD, ReflectModel


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result


def phase_problem_plot(flag):
    if flag != "SLD" and flag != "ACF":
        raise Exception("Invalid flag to plot")

    qvals = np.linspace(0,0.2,1025)[1:]
    sld1 = SLD(0, 0)
    sld3 = SLD(2.074, 0)

    layer1 = sld1(0, 0)
    layer3 = sld3(0, 0)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    for i, style in zip([0,1.0], ['k-', 'r:']):
        sld20 = SLD(6.335 - i, 0)
        sld21 = SLD(6.335 + i, 0)
        layer20 = sld20(100, 0)
        layer21 = sld21(100, 0)
        structure = Structure(layer1 | layer20 | layer21 | layer3)
        z, rho = structure.sld_profile()
        drho = (rho[1:] - rho[:-1]) / (z[1:] - z[:-1])
        z_ = 0.5 * (z[:-1] + z[1:])
        acf = autocorr(drho)
        z_acf = z_ - z_.min()
        z_acf = np.hstack((-np.flip(z_acf)[:-1], z_acf))

        if flag == 'SLD':
            ax1.plot(z, rho, style)
            ax2.semilogy(qvals, ReflectModel(structure, dq=0.0).model(qvals), style)

        else:
            ax1.stem(z_,   drho, style, markerfmt=' ', basefmt=style, use_line_collection=True)
            ax2.stem(z_acf, acf, style, markerfmt=' ', basefmt=style, use_line_collection=True)


    if flag == 'SLD':
        ax1.set_xlabel(r'$z$/Å')
        ax1.set_ylabel(r'$\rho(z)$/Å$^{-2}$')

        ax2.set_xlabel(r'$q$/Å')
        ax2.set_ylabel(r'$R(q)$')

    else:
        ax1.set_xlabel(r'$z$/Å')
        ax1.set_ylabel(r'$\rho\'(z)$/Å$^{-3}$')

        ax2.set_xlabel(r'$z$/Å')
        ax2.set_ylabel("$ \\rm{ACF}_{\\rho'}(z)/ \\AA^{-5}$")

    plt.tight_layout()
    figfilename = f"phase_problem_{flag}.pdf"
    plt.savefig(figfilename)
    print(f"Figure saved as {figfilename}")
    plt.close()

if __name__ == "__main__":
    phase_problem_plot("SLD")
    phase_problem_plot("ACF")