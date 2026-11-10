"""
Step 3: the band structure of silicon.

Two calculations:
  1. scf   : find the ground-state electron density on a uniform k-mesh
  2. bands : using that fixed density, compute energies along a path of
             high-symmetry points in the Brillouin zone (L - Gamma - X - U | K - Gamma)

Silicon has an indirect gap, with the valence band maximum at Gamma and the
conduction band minimum about 85% of the way from Gamma to X. PBE should give
a gap of roughly 0.6 eV, about half the experimental 1.12 eV. This band gap
problem comes back for EuS in steps 5 and 6.

Outputs:
  results/si_bands.png   band structure with the gap marked
  results/si_bands.json  gap, VBM and CBM locations
"""

import numpy as np
from ase.build import bulk

import qe_tools as qe
from plotting import BLUE, INK, MUTED, ORANGE, plt

GAP_EXP = 1.12         # eV, experimental indirect gap
N_OCCUPIED = 4         # 2 Si atoms x 4 valence electrons = 8 electrons = 4 filled bands

settings = qe.load_json("si_convergence.json")
ecut, k = settings["ecutwfc_Ry"], settings["kmesh"]
a0 = qe.load_json("si_lattice.json")["a_pbe_A"]   # use the PBE lattice constant from step 2

atoms = bulk("Si", "diamond", a=a0)
species = {"Si": (28.085, "Si.pbe-n-rrkjus_psl.1.0.0.UPF")}
workdir = qe.RUN_DIR / "03_si_bands"
common_system = {"ecutwfc": ecut, "ecutrho": 8 * ecut}

# --- 1. Self-consistent ground state ----------------------------------------
scf = qe.make_pw_input(
    atoms,
    namelists={
        "control": {"calculation": "scf", "prefix": "si"},
        "system": common_system,
        "electrons": {"conv_thr": 1e-10},
    },
    species=species,
    kpoints=(k, k, k),
)
qe.run_pw("scf", scf, workdir)

# --- 2. Bands along the high-symmetry path ----------------------------------
path = atoms.cell.bandpath("LGXU,KG", npoints=150)
bands = qe.make_pw_input(
    atoms,
    namelists={
        "control": {"calculation": "bands", "prefix": "si"},
        "system": {**common_system, "nbnd": 8},   # 4 valence + 4 conduction bands
    },
    species=species,
    kpoints=path.kpts,
)
result = qe.run_pw("bands", bands, workdir)

energies = result["eigenvalues"][0]                  # shape (k-points, bands)
vbm, cbm, gap = qe.band_edges_by_count(result["eigenvalues"], N_OCCUPIED)
energies = energies - vbm                            # put the valence band top at 0 eV

x, special_x, special_labels = path.get_linear_kpoint_axis()
special_labels = ["Γ" if s == "G" else s for s in special_labels]
# a path break (U,K) gives two labels at the same position: merge them into "U|K"
merged_x, merged_labels = [], []
for sx, lab in zip(special_x, special_labels):
    if merged_x and np.isclose(sx, merged_x[-1]):
        merged_labels[-1] += "|" + lab
    else:
        merged_x.append(sx)
        merged_labels.append(lab)
special_x, special_labels = merged_x, merged_labels
# first k-point where the top valence band reaches its maximum (Gamma)
i_vbm = np.flatnonzero(energies[:, N_OCCUPIED - 1] > -1e-6)[0]
i_cbm = np.argmin(energies[:, N_OCCUPIED])

# where along Gamma-X the conduction band minimum sits (fraction of the distance)
g_x = [special_x[1], special_x[2]]
cbm_fraction = (x[i_cbm] - g_x[0]) / (g_x[1] - g_x[0])

print(f"\nPBE indirect gap: {gap:.3f} eV   (experiment {GAP_EXP} eV)")
print(f"CBM is {100 * cbm_fraction:.0f}% of the way from Gamma to X")

qe.save_json("si_bands.json", {
    "lattice_constant_A": a0,
    "gap_pbe_eV": round(gap, 3),
    "gap_exp_eV": GAP_EXP,
    "cbm_fraction_gamma_to_x": round(cbm_fraction, 2),
})

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.axhspan(0, gap, color=ORANGE, alpha=0.12, lw=0)
for n in range(energies.shape[1]):
    ax.plot(x, energies[:, n], color=BLUE, lw=1.6)
ax.plot(x[i_vbm], 0, "o", color=INK, ms=5, zorder=5)
ax.plot(x[i_cbm], energies[i_cbm, N_OCCUPIED], "o", color=INK, ms=5, zorder=5)
ax.text(0, 1.02, f"Indirect gap (shaded): PBE {gap:.2f} eV vs experiment {GAP_EXP} eV",
        transform=ax.transAxes, fontsize=9, color=MUTED)
for sx in special_x:
    ax.axvline(sx, color=MUTED, lw=0.6)
ax.set_xticks(special_x, special_labels)
ax.grid(axis="x", visible=False)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-12.5, 6)
ax.set_ylabel("E - E_VBM (eV)")
ax.set_title("Silicon band structure (PBE)", loc="left", pad=20)
fig.savefig(qe.RESULTS_DIR / "si_bands.png")
print("Saved results/si_bands.png")
