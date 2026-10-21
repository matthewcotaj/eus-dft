"""
Step 1: convergence tests on silicon.

DFT has two main numerical knobs:
  - ecutwfc: the plane-wave cutoff. Higher means a more complete basis set.
  - the k-point mesh: how finely the Brillouin zone is sampled.

Both make the answer more accurate and the calculation slower. A converged
calculation is one where turning the knob further changes the total energy by
less than a chosen tolerance. Here that tolerance is 1 meV/atom, a common
standard. Silicon is used because it is cheap and well understood, so any
problem is a problem with the workflow, not the material.

Outputs:
  results/si_convergence.png   energy error vs cutoff and vs k-mesh
  results/si_convergence.json  the chosen cutoff and k-mesh
"""

import numpy as np
from ase.build import bulk

import qe_tools as qe
from plotting import BLUE, INK, MUTED, plt

TOLERANCE = 1.0    # meV/atom
A_EXP = 5.431      # experimental Si lattice constant (Angstrom)

atoms = bulk("Si", "diamond", a=A_EXP)
n_atoms = len(atoms)
species = {"Si": (28.085, "Si.pbe-n-rrkjus_psl.1.0.0.UPF")}
workdir = qe.RUN_DIR / "01_si_convergence"


def total_energy(ecut, kmesh):
    """Total energy (eV per atom) for one cutoff and one k-mesh."""
    name = f"ecut{ecut}_k{kmesh}"
    text = qe.make_pw_input(
        atoms,
        namelists={
            "control": {"calculation": "scf", "prefix": name},
            # ultrasoft pseudopotentials need a charge-density cutoff of about 8x ecutwfc
            "system": {"ecutwfc": ecut, "ecutrho": 8 * ecut},
            "electrons": {"conv_thr": 1e-10},
        },
        species=species,
        kpoints=(kmesh, kmesh, kmesh),
    )
    return qe.run_pw(name, text, workdir)["energy"] / n_atoms


def first_converged(values, errors):
    """Smallest setting whose error, and every error after it, is below tolerance."""
    for i in range(len(values)):
        if np.all(errors[i:] < TOLERANCE):
            return values[i]
    return values[-1]


# --- 1. Cutoff test at a fixed, generous k-mesh -----------------------------
print("Cutoff test")
cutoffs = [20, 25, 30, 35, 40, 45, 50, 60, 70]
e_cut = np.array([total_energy(ec, 8) for ec in cutoffs])
err_cut = np.abs(e_cut - e_cut[-1]) * 1000          # meV/atom vs the largest cutoff
best_cut = first_converged(cutoffs, err_cut)

# --- 2. k-point test at the chosen cutoff ----------------------------------
print("k-point test")
meshes = [2, 4, 6, 8, 10, 12]
e_k = np.array([total_energy(best_cut, k) for k in meshes])
err_k = np.abs(e_k - e_k[-1]) * 1000
best_k = first_converged(meshes, err_k)

print(f"\nConverged to {TOLERANCE} meV/atom: ecutwfc = {best_cut} Ry, k-mesh = {best_k}x{best_k}x{best_k}")

qe.save_json("si_convergence.json", {
    "tolerance_meV_per_atom": TOLERANCE,
    "ecutwfc_Ry": best_cut,
    "kmesh": best_k,
    "cutoff_test": {"ecutwfc_Ry": cutoffs, "error_meV_per_atom": err_cut.round(4).tolist()},
    "kpoint_test": {"kmesh": meshes, "error_meV_per_atom": err_k.round(4).tolist()},
})

# --- 3. Plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
panels = [
    (axes[0], cutoffs[:-1], err_cut[:-1], best_cut, "Plane-wave cutoff ecutwfc (Ry)", "Cutoff (k-mesh 8x8x8)"),
    (axes[1], meshes[:-1], err_k[:-1], best_k, "k-mesh (n x n x n)", f"k-points (ecutwfc {best_cut} Ry)"),
]
for ax, x, y, best, xlabel, title in panels:
    y = np.maximum(y, 1e-4)   # log axis cannot show exact zeros
    ax.semilogy(x, y, "o-", color=BLUE)
    ax.axhline(TOLERANCE, color=MUTED, ls="--", lw=1)
    ax.text(x[0], TOLERANCE * 1.3, f"{TOLERANCE:g} meV/atom", color=MUTED, fontsize=8)
    ax.axvline(best, color=INK, lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
axes[0].set_ylabel("|E - E_converged| (meV/atom)")
fig.suptitle("Silicon convergence tests (PBE)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(qe.RESULTS_DIR / "si_convergence.png")
print("Saved results/si_convergence.png")
