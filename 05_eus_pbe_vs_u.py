"""
Step 5: EuS with plain PBE versus PBE+U.

EuS is a ferromagnetic semiconductor with a measured gap of about 1.65 eV.
The occupied Eu 4f states sit at the top of the valence band and the
conduction band is mostly Eu 5d.

Plain PBE gets this badly wrong. It spreads the seven 4f electrons over states
that sit right at the Fermi level, so EuS comes out with essentially no gap.
The reason is self-interaction error: in semilocal DFT an electron partly
repels itself, which pushes localized orbitals like 4f up in energy and
smears them out.

PBE+U adds an on-site penalty U to the 4f orbitals (the Hubbard correction).
It pushes the filled 4f states down and the empty ones up, which opens a gap.

For each method this script computes:
  - the lattice constant and bulk modulus (equation of state fit)
  - the band gap and magnetic moment at that lattice constant
  - the projected density of states (which orbitals make up which bands)

Outputs:
  results/eus_pdos.png        projected DOS, PBE and PBE+U side by side
  results/eus_eos.png         energy vs lattice constant for both methods
  results/eus_pbe_vs_u.json   all numbers, with experiment for comparison
"""

import numpy as np
from ase.eos import EquationOfState
from ase.units import GPa

import eus
import qe_tools as qe
from plotting import BLUE, GREEN, INK, MUTED, ORANGE, plt

U_VALUE = 7.0          # eV on Eu 4f, a common choice for Eu2+ compounds
NSCF_KMESH = 12        # denser mesh for a smooth density of states

settings = qe.load_json("eus_convergence.json")
ecut, k = settings["ecutwfc_Ry"], settings["kmesh"]
workdir = qe.RUN_DIR / "05_eus_pbe_vs_u"
methods = {"PBE": 0.0, "PBE+U": U_VALUE}

summary = {}
eos_data = {}
pdos_data = {}

for method, u in methods.items():
    print(f"\n{method}")
    tag = "pbe" if u == 0 else "pbeu"

    # --- 1. Equation of state ------------------------------------------------
    lattice = eus.A_EXP * np.linspace(0.97, 1.03, 7)
    energies = []
    for a in lattice:
        r = eus.run(f"{tag}_a{a:.4f}", workdir, a=a, ecut=ecut, kmesh=k, u=u)
        energies.append(r["energy"])
    volumes = lattice ** 3 / 4                      # fcc primitive cell volume
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, b0 = eos.fit()
    a0 = (4 * v0) ** (1 / 3)
    eos_data[method] = (lattice, np.array(energies), eos, v0, e0)

    # --- 2. Ground state at the predicted lattice constant ------------------
    name = f"{tag}_eq"
    scf = eus.run(name, workdir, a=a0, ecut=ecut, kmesh=k, u=u)

    # --- 3. Denser k-mesh for the density of states, then project onto orbitals
    nscf = eus.run(name, workdir, a=a0, ecut=ecut, kmesh=NSCF_KMESH, u=u,
                   calculation="nscf", nbnd=30, occupations="tetrahedra_opt")
    fermi = nscf["fermi"]
    qe.run_projwfc(name, workdir, emin=fermi - 12, emax=fermi + 12)
    energy, total, proj = qe.read_pdos(workdir, name, eus.SPECIES)
    vbm, cbm, gap = qe.band_edges(nscf)
    pdos_data[method] = (energy - vbm, total, proj)   # zero at the valence band top

    summary[method] = {
        "U_eV": u,
        "lattice_constant_A": round(a0, 4),
        "lattice_error_percent": round(100 * (a0 / eus.A_EXP - 1), 2),
        "bulk_modulus_GPa": round(b0 / GPa, 1),
        "gap_eV": round(gap, 3),
        "moment_bohr_magneton": round(scf["magnetization"], 3),
    }
    print(f"  a = {a0:.4f} A, gap = {gap:.3f} eV, moment = {scf['magnetization']:.2f} muB")

summary["experiment"] = {
    "lattice_constant_A": eus.A_EXP,
    "gap_eV": eus.GAP_EXP,
    "moment_bohr_magneton": eus.MOMENT_EXP,
}
qe.save_json("eus_pbe_vs_u.json", summary)

print("\n            a (A)    gap (eV)   moment (muB)")
for key in ["PBE", "PBE+U", "experiment"]:
    s = summary[key]
    print(f"{key:<11} {s['lattice_constant_A']:<8} {s['gap_eV']:<10} {s['moment_bohr_magneton']}")


# --- Plot 1: projected density of states -------------------------------------
colors = {"Eu 4f": ORANGE, "Eu 5d": BLUE, "S 3p": GREEN}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, method in zip(axes, methods):
    e, total, proj = pdos_data[method]
    ax.fill_between(e, total[0], color=MUTED, alpha=0.15, lw=0, label="total")
    ax.fill_between(e, total[1], color=MUTED, alpha=0.15, lw=0)
    for label, color in colors.items():
        ax.plot(e, proj[label][0], color=color, lw=1.4, label=label)
        ax.plot(e, proj[label][1], color=color, lw=1.4)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    s = summary[method]
    ax.set_title(f"{method}   gap {s['gap_eV']:.2f} eV", loc="left")
    ax.set_xlim(-7, 12)
    ax.set_xticks(range(-6, 13, 2))
    ax.set_xlabel("E - VBM (eV)")
axes[0].set_ylim(-8, 8)
axes[0].set_ylabel("DOS (states/eV), spin down below 0")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle(f"EuS projected density of states (experimental gap {eus.GAP_EXP} eV)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(qe.RESULTS_DIR / "eus_pdos.png")

# --- Plot 2: equation of state -----------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for method, color in zip(methods, [MUTED, BLUE]):
    lattice, energies, eos, v0, e0 = eos_data[method]
    v_fine = np.linspace(min(lattice) ** 3 / 4, max(lattice) ** 3 / 4, 200)
    e_fine = eos.func(v_fine, *eos.eos_parameters)
    a_fine = (4 * v_fine) ** (1 / 3)
    ax.plot(a_fine, (e_fine - e0) * 1000, color=color, label=method)
    ax.plot(lattice, (energies - e0) * 1000, "o", color=color, mfc="white", mew=2)
ax.axvline(eus.A_EXP, color=ORANGE, lw=1, ls="--", label=f"experiment ({eus.A_EXP} Å)")
ax.set_xlabel("Lattice constant (Å)")
ax.set_ylabel("E - E_min (meV/cell)")
ax.set_title("EuS equation of state", loc="left")
ax.legend(loc="upper center")
fig.savefig(qe.RESULTS_DIR / "eus_eos.png")
print("\nSaved results/eus_pdos.png and results/eus_eos.png")
