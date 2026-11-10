"""
Step 6: how the EuS results depend on the Hubbard U.

U is an input to PBE+U, not something it calculates. Step 5 used U = 7 eV,
a common literature value for Eu2+, but any single choice hides how much the
answer depends on it. This script repeats the calculation for U from 0 to
10 eV at the experimental lattice constant and tracks:
  - the band gap
  - where the occupied Eu 4f band sits relative to the valence band top
  - where the S 3p band sits, for comparison

If some U reproduced the experimental gap of 1.65 eV with a sensible 4f
position, PBE+U would be a complete fix. If not, the remaining error lies
somewhere U does not reach.

Outputs:
  results/eus_u_scan.png   gap and 4f position vs U
  results/eus_u_scan.json  the numbers
"""

import numpy as np

import eus
import qe_tools as qe
from plotting import BLUE, GREEN, INK, MUTED, ORANGE, plt

U_VALUES = [0, 2, 4, 6, 7, 8, 10]

settings = qe.load_json("eus_convergence.json")
ecut, k = settings["ecutwfc_Ry"], settings["kmesh"]
workdir = qe.RUN_DIR / "06_eus_u_scan"


def centroid(energy, dos, below=None):
    """Average energy of a DOS curve, optionally only counting states below an energy."""
    mask = np.ones_like(energy, bool) if below is None else energy <= below
    return np.trapezoid(energy[mask] * dos[mask], energy[mask]) / np.trapezoid(dos[mask], energy[mask])


gaps, f_center, p_center = [], [], []
for u in U_VALUES:
    name = f"u{u}"
    r = eus.run(name, workdir, ecut=ecut, kmesh=k, u=u)
    vbm, cbm, gap = qe.band_edges(r)
    fermi = r["fermi"]

    qe.run_projwfc(name, workdir, emin=fermi - 12, emax=fermi + 12)
    energy, _, proj = qe.read_pdos(workdir, name, eus.SPECIES)
    # the occupied 4f states are all spin up; S 3p is nearly the same in both channels
    f_up = proj["Eu 4f"][0]
    p_up = proj["S 3p"][0]
    f_center.append(centroid(energy, f_up, below=fermi) - vbm)
    p_center.append(centroid(energy, p_up, below=fermi) - vbm)
    gaps.append(gap)
    print(f"  U = {u:>4} eV   gap {gap:.3f} eV   4f center {f_center[-1]:6.2f} eV   "
          f"S 3p center {p_center[-1]:6.2f} eV")

qe.save_json("eus_u_scan.json", {
    "lattice_constant_A": eus.A_EXP,
    "U_eV": U_VALUES,
    "gap_eV": np.round(gaps, 3).tolist(),
    "occupied_4f_center_minus_vbm_eV": np.round(f_center, 3).tolist(),
    "S_3p_center_minus_vbm_eV": np.round(p_center, 3).tolist(),
    "gap_exp_eV": eus.GAP_EXP,
})

# --- Plot -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

ax = axes[0]
ax.plot(U_VALUES, gaps, "o-", color=BLUE)
ax.axhline(eus.GAP_EXP, color=ORANGE, ls="--", lw=1)
ax.text(U_VALUES[0], eus.GAP_EXP + 0.05, f"experiment {eus.GAP_EXP} eV", color=MUTED, fontsize=8)
ax.set_xlabel("U on Eu 4f (eV)")
ax.set_ylabel("Band gap (eV)")
ax.set_ylim(bottom=0)
ax.set_title("Band gap", loc="left")

ax = axes[1]
ax.plot(U_VALUES, f_center, "o-", color=ORANGE, label="occupied Eu 4f")
ax.plot(U_VALUES, p_center, "o-", color=GREEN, label="S 3p")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("U on Eu 4f (eV)")
ax.set_ylabel("Band center - VBM (eV)")
ax.set_title("Where the 4f states sit", loc="left")
ax.legend(loc="center left")

for ax in axes:
    ax.set_xticks(U_VALUES)
fig.suptitle(f"EuS: dependence on U (a = {eus.A_EXP} Å)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(qe.RESULTS_DIR / "eus_u_scan.png")
print("Saved results/eus_u_scan.png")
