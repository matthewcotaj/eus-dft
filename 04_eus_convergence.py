"""
Step 4: convergence tests for EuS.

Same idea as step 1, but EuS is much harder than silicon. The Eu
pseudopotential keeps 27 electrons in the valence, including the semicore 4d
and the tightly bound 4f shell. Tightly bound orbitals vary quickly in space,
and describing them needs many more plane waves.

The density cutoff ecutrho is kept at 4x ecutwfc, and checked against 8x
at the end of the cutoff test.

The convergence criteria also change. The total energy itself converges very
slowly with cutoff, but the results here depend only on energy differences
and on the pressure (the slope of energy vs volume, which sets the lattice
constant). So the cutoff is chosen by when the pressure and the band gap stop
changing. The k-mesh is chosen by the total energy, as for silicon.

Tests are run with PBE+U (U = 7 eV on Eu 4f), the setup used for the main results.

Outputs:
  results/eus_convergence.png   pressure error vs cutoff, energy error vs k-mesh
  results/eus_convergence.json  chosen cutoff and k-mesh
"""

import numpy as np

import eus
import qe_tools as qe
from plotting import BLUE, INK, MUTED, plt

U_TEST = 7.0
PRESSURE_TOL = 2.0     # kbar; about 0.1% in lattice constant (bulk modulus ~50 GPa)
GAP_TOL = 0.01         # eV
ENERGY_TOL = 1.0       # meV/atom
workdir = qe.RUN_DIR / "04_eus_convergence"


def first_converged(values, *errors_and_tols):
    """Smallest setting where every error, at it and above it, is under its tolerance."""
    for i in range(len(values)):
        if all(np.all(err[i:] < tol) for err, tol in errors_and_tols):
            return values[i]
    return values[-1]


# --- 1. Cutoff test at a fixed k-mesh ----------------------------------------
print("Cutoff test")
cutoffs = [60, 80, 100, 120, 140, 160]
runs = [eus.run(f"ecut{ec}_k6", workdir, ecut=ec, kmesh=6, u=U_TEST, tstress=True)
        for ec in cutoffs]
pressure = np.array([r["pressure"] for r in runs])
gap = np.array([qe.band_edges(r)[2] for r in runs])
err_p = np.abs(pressure - pressure[-1])
err_gap = np.abs(gap - gap[-1])
best_cut = first_converged(cutoffs, (err_p, PRESSURE_TOL), (err_gap, GAP_TOL))

for ec, p, g in zip(cutoffs, pressure, gap):
    print(f"  {ec:>4} Ry   pressure {p:9.1f} kbar   gap {g:.3f} eV")

# The density cutoff (ecutrho) is 4x ecutwfc throughout. Check that doubling it
# to 8x changes nothing, so the cheaper setting is safe.
dual8 = eus.run(f"ecut{best_cut}_k6_dual8", workdir, ecut=best_cut, kmesh=6, u=U_TEST,
                tstress=True, dual=8)
i = cutoffs.index(best_cut)
dual_dp = abs(dual8["pressure"] - pressure[i])
dual_dgap = abs(qe.band_edges(dual8)[2] - gap[i])
print(f"  ecutrho 8x instead of 4x: pressure changes {dual_dp:.2f} kbar, gap {dual_dgap:.4f} eV")

# --- 2. k-point test at the chosen cutoff ------------------------------------
print("k-point test")
meshes = [4, 6, 8, 10]
runs = [eus.run(f"ecut{best_cut}_k{kk}", workdir, ecut=best_cut, kmesh=kk, u=U_TEST, tstress=True)
        for kk in meshes]
energy = np.array([r["energy"] / 2 for r in runs])       # eV per atom
gap_k = np.array([qe.band_edges(r)[2] for r in runs])
err_e = np.abs(energy - energy[-1]) * 1000
err_gap_k = np.abs(gap_k - gap_k[-1])
best_k = first_converged(meshes, (err_e, ENERGY_TOL), (err_gap_k, GAP_TOL))

print(f"\nConverged: ecutwfc = {best_cut} Ry, k-mesh = {best_k}x{best_k}x{best_k}")

qe.save_json("eus_convergence.json", {
    "ecutwfc_Ry": best_cut,
    "kmesh": best_k,
    "U_eV": U_TEST,
    "tolerances": {"pressure_kbar": PRESSURE_TOL, "gap_eV": GAP_TOL,
                   "energy_meV_per_atom": ENERGY_TOL},
    "cutoff_test": {"ecutwfc_Ry": cutoffs,
                    "pressure_kbar": pressure.round(2).tolist(),
                    "gap_eV": gap.round(4).tolist()},
    "density_cutoff_check": {"pressure_change_kbar": round(dual_dp, 3),
                             "gap_change_eV": round(dual_dgap, 4)},
    "kpoint_test": {"kmesh": meshes,
                    "energy_error_meV_per_atom": err_e.round(4).tolist(),
                    "gap_eV": gap_k.round(4).tolist()},
})

# --- 3. Plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
panels = [
    (axes[0], cutoffs[:-1], err_p[:-1], PRESSURE_TOL, best_cut, "kbar",
     "Plane-wave cutoff ecutwfc (Ry)", "|P - P_converged| (kbar)", "Cutoff (k-mesh 6x6x6)"),
    (axes[1], meshes[:-1], err_e[:-1], ENERGY_TOL, best_k, "meV/atom",
     "k-mesh (n x n x n)", "|E - E_converged| (meV/atom)", f"k-points (ecutwfc {best_cut} Ry)"),
]
for ax, x, y, tol, best, unit, xlabel, ylabel, title in panels:
    ax.semilogy(x, np.maximum(y, 1e-3), "o-", color=BLUE)
    ax.axhline(tol, color=MUTED, ls="--", lw=1)
    ax.text(x[0], tol * 1.3, f"{tol:g} {unit}", color=MUTED, fontsize=8)
    ax.axvline(best, color=INK, lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
fig.suptitle("EuS convergence tests (PBE+U)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(qe.RESULTS_DIR / "eus_convergence.png")
print("Saved results/eus_convergence.png")
