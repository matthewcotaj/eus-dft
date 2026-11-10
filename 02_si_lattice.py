"""
Step 2: the PBE lattice constant of silicon.

Compute the total energy at a range of lattice constants, then fit an
equation of state E(V). The minimum gives the predicted lattice constant and
the curvature gives the bulk modulus.

PBE usually overestimates lattice constants by about 1%. That is a
systematic error of the functional, not a bug. Step 5 makes the same
comparison for EuS.

Outputs:
  results/si_lattice.png   E(V) points and the fitted curve
  results/si_lattice.json  lattice constant and bulk modulus vs experiment
"""

import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from ase.units import GPa

import qe_tools as qe
from plotting import BLUE, INK, MUTED, ORANGE, plt

A_EXP = 5.431    # Angstrom
B_EXP = 98.0     # GPa, experimental bulk modulus

settings = qe.load_json("si_convergence.json")
ecut, k = settings["ecutwfc_Ry"], settings["kmesh"]
species = {"Si": (28.085, "Si.pbe-n-rrkjus_psl.1.0.0.UPF")}
workdir = qe.RUN_DIR / "02_si_lattice"

scales = np.linspace(0.97, 1.03, 9)    # +/- 3% around experiment
volumes, energies = [], []
for s in scales:
    atoms = bulk("Si", "diamond", a=A_EXP * s)
    name = f"a{A_EXP * s:.4f}"
    text = qe.make_pw_input(
        atoms,
        namelists={
            "control": {"calculation": "scf", "prefix": name},
            "system": {"ecutwfc": ecut, "ecutrho": 8 * ecut},
            "electrons": {"conv_thr": 1e-10},
        },
        species=species,
        kpoints=(k, k, k),
    )
    result = qe.run_pw(name, text, workdir)
    volumes.append(atoms.get_volume())
    energies.append(result["energy"])

eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
v0, e0, b0 = eos.fit()
# diamond Si: 2-atom primitive cell, conventional cubic cell holds 8 atoms (4x the volume)
a0 = (4 * v0) ** (1 / 3)
b0_gpa = b0 / GPa

print(f"\nPBE lattice constant: {a0:.4f} A   (experiment {A_EXP} A, error {100 * (a0 / A_EXP - 1):+.2f}%)")
print(f"PBE bulk modulus:     {b0_gpa:.1f} GPa (experiment {B_EXP} GPa)")

qe.save_json("si_lattice.json", {
    "a_pbe_A": round(a0, 4),
    "a_exp_A": A_EXP,
    "a_error_percent": round(100 * (a0 / A_EXP - 1), 2),
    "bulk_modulus_pbe_GPa": round(b0_gpa, 1),
    "bulk_modulus_exp_GPa": B_EXP,
})

# --- Plot: energy vs lattice constant (easier to read than vs volume) ------
a_points = (4 * np.array(volumes)) ** (1 / 3)
v_fine = np.linspace(min(volumes), max(volumes), 200)
a_fine = (4 * v_fine) ** (1 / 3)
e_fine = eos.func(v_fine, *eos.eos_parameters)   # the fitted curve on a fine grid

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(a_fine, (e_fine - e0) * 1000 / 2, color=BLUE, label="Birch-Murnaghan fit")
ax.plot(a_points, (np.array(energies) - e0) * 1000 / 2, "o", color=BLUE, mfc="white", mew=2, label="DFT (PBE)")
ax.axvline(a0, color=BLUE, lw=1, ls="--")
ax.axvline(A_EXP, color=ORANGE, lw=1, ls="--")
top = ax.get_ylim()[1]
ax.text(a0, top * 0.55, f" PBE\n {a0:.3f} Å", color=INK, fontsize=8)
ax.text(A_EXP, top * 0.55, f"experiment \n{A_EXP} Å ", color=MUTED, fontsize=8, ha="right")
ax.set_xlabel("Lattice constant (Å)")
ax.set_ylabel("E - E_min (meV/atom)")
ax.set_title("Silicon equation of state", loc="left")
ax.legend(loc="upper right")
fig.savefig(qe.RESULTS_DIR / "si_lattice.png")
print("Saved results/si_lattice.png")
