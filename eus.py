"""
Shared setup for the EuS calculations: structure, pseudopotentials and the
pw.x settings every EuS script uses.

EuS has the rocksalt structure (like NaCl): Eu²⁺ and S²⁻ on two interleaved
fcc lattices. The primitive cell holds one Eu and one S. Eu²⁺ has a half-filled
4f shell (4f⁷), so with all seven spins aligned the cell carries 7 Bohr magnetons.
"""

import numpy as np
from ase.build import bulk, make_supercell

import qe_tools as qe

# Experimental reference values
A_EXP = 5.968        # lattice constant (Angstrom), room temperature
GAP_EXP = 1.65       # optical gap (eV): the occupied 4f -> empty 5d edge. It
                     # red-shifts by about 0.1 eV when EuS orders below T_C
MOMENT_EXP = 7.0     # Bohr magnetons per Eu (4f7, spin only)
TC_EXP = 16.5        # Curie temperature (K)

SPECIES = {
    "Eu": (151.964, "Eu.pbe-spdfn-kjpaw_psl.1.0.0.UPF"),
    "S": (32.06, "S.pbe-n-kjpaw_psl.1.0.0.UPF"),
}


def structure(a=A_EXP):
    """Primitive rocksalt cell of EuS with lattice constant a (Angstrom)."""
    return bulk("EuS", "rocksalt", a=a)


def hubbard_card(u):
    """
    HUBBARD card for pw.x. U is applied to the Eu 4f orbitals only.
    'ortho-atomic' projectors are the current recommended choice in QE.
    """
    if u == 0:
        return ""
    return f"HUBBARD ortho-atomic\n  U Eu-4f {u}\n"


def run(name, workdir, a=A_EXP, ecut=140, kmesh=6, u=0.0, calculation="scf",
        nbnd=None, tstress=False, occupations="smearing", dual=4):
    """
    Run one ferromagnetic EuS calculation and return the parsed results.

    a           lattice constant (Angstrom)
    ecut        plane-wave cutoff (Ry)
    dual        density cutoff as a multiple of ecut; step 4 shows 4 is enough
                for these PAW pseudopotentials
    kmesh       n for an n x n x n Monkhorst-Pack mesh
    u           Hubbard U on Eu 4f (eV); 0 means plain PBE
    calculation 'scf', or 'nscf' to reuse a finished scf with the same name
    occupations 'smearing' for the scf, 'tetrahedra_opt' for a smooth DOS (nscf)
    """
    system = {
        "ecutwfc": ecut,
        "ecutrho": dual * ecut,
        "nspin": 2,
        "starting_magnetization(1)": 0.8,   # start with the Eu spins aligned
        "occupations": occupations,
    }
    if occupations == "smearing":
        system.update({"smearing": "mv", "degauss": 0.01})
    if nbnd is not None:
        system["nbnd"] = nbnd

    control = {"calculation": calculation, "prefix": name}
    if tstress:
        control["tstress"] = True

    text = qe.make_pw_input(
        structure(a),
        namelists={
            "control": control,
            "system": system,
            "electrons": {"conv_thr": 1e-9, "mixing_beta": 0.3},
        },
        species=SPECIES,
        kpoints=(kmesh, kmesh, kmesh),
        extra_cards=hubbard_card(u),
    )
    return qe.run_pw(f"{name}_{calculation}", text, workdir)


# ---------------------------------------------------------------------------
# Magnetic configurations for exchange constants (step 7)
# ---------------------------------------------------------------------------
#
# The Eu ions form an fcc lattice. Each Eu has 12 nearest neighbors (J1, at
# a/sqrt2) and 6 next-nearest neighbors (J2, at a). With the classical
# Heisenberg model H = -sum over pairs J_ij S_i.S_j (each pair counted once)
# the energy per Eu of three simple orderings is
#
#   FM      all spins parallel                -S^2 (6 J1 + 3 J2)
#   AFM-I   alternating (001) planes          -S^2 (-2 J1 + 3 J2)
#   AFM-II  alternating (111) planes          +S^2 (3 J2)
#
# so E(AFM-I) - E(FM) = 8 S^2 J1 and E(AFM-II) - E(FM) = 6 S^2 (J1 + J2).
# Each AFM state is compared with FM computed in the same two-Eu cell, so
# k-point and basis-set errors cancel in the difference.

S_EU = 3.5           # Eu2+ spin (4f7)

# Two-formula-unit cells, as rows of primitive lattice vectors. Each holds one
# Eu on each of the two alternating planes of its AFM ordering. Both matrices
# have determinant +2, so the cells stay right-handed, which pw.x assumes.
MAGNETIC_CELLS = {
    # tetragonal, a/sqrt2 x a/sqrt2 x a, AFM planes along [001]
    "I": [[0, 0, 1], [1, -1, 0], [1, 1, -1]],
    # rhombohedral, AFM planes along [111]
    "II": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
}


def magnetic_structure(cell, a=A_EXP):
    """
    Two-Eu cell of type 'I' or 'II' and the plane index (0 or 1) of each atom.

    The plane index is the parity of the (001) or (111) plane an atom sits on.
    In the AFM state the Eu on plane 0 points up and the one on plane 1 down.
    """
    atoms = make_supercell(structure(a), MAGNETIC_CELLS[cell])
    pos = atoms.get_positions()
    if cell == "I":
        layer = np.rint(2 * pos[:, 2] / a)          # (001) planes are a/2 apart
    else:
        layer = np.rint(pos.sum(axis=1) / a)        # (111) planes: x + y + z = n a
    plane = layer.astype(int) % 2
    return atoms, plane


def kmesh_for(atoms, kmesh, a=A_EXP):
    """
    An automatic k-mesh for a supercell with at least the k-point density of
    an n x n x n mesh on the primitive cell (rounded up, never coarser).
    """
    prim = np.linalg.norm(structure(a).cell.reciprocal(), axis=1)[0]
    lengths = np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return tuple(max(1, int(np.ceil(kmesh * b / prim - 1e-6))) for b in lengths)


def run_magnetic(name, workdir, cell, afm, a=A_EXP, ecut=140, kmesh=6, u=0.0, dual=4):
    """
    One scf of the two-Eu cell 'I' or 'II', ferromagnetic (afm=False) or with
    the corresponding antiferromagnetic order (afm=True).

    The two Eu atoms are separate species (Eu1, Eu2) so they can start with
    opposite moments. Tighter convergence than the other steps, because the
    energy differences here are only a few meV.
    """
    atoms, plane = magnetic_structure(cell, a)
    symbols = atoms.get_chemical_symbols()
    labels = [f"Eu{p + 1}" if s == "Eu" else s for s, p in zip(symbols, plane)]
    species = {"Eu1": SPECIES["Eu"], "Eu2": SPECIES["Eu"], "S": SPECIES["S"]}

    system = {
        "ecutwfc": ecut,
        "ecutrho": dual * ecut,
        "nspin": 2,
        "starting_magnetization(1)": 0.8,
        "starting_magnetization(2)": -0.8 if afm else 0.8,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,
    }
    hubbard = ""
    if u != 0:
        hubbard = f"HUBBARD ortho-atomic\n  U Eu1-4f {u}\n  U Eu2-4f {u}\n"

    text = qe.make_pw_input(
        atoms,
        namelists={
            "control": {"calculation": "scf", "prefix": name},
            "system": system,
            "electrons": {"conv_thr": 1e-10, "mixing_beta": 0.3},
        },
        species=species,
        kpoints=kmesh_for(atoms, kmesh, a),
        labels=labels,
        extra_cards=hubbard,
    )
    return qe.run_pw(f"{name}_scf", text, workdir)


def exchange_constants(e_fm_1, e_afm_1, e_fm_2, e_afm_2, n_eu=2, spin=S_EU):
    """
    J1 and J2 (same units as the energies) from total energies of the FM and
    AFM states in cells I and II, each holding n_eu Eu atoms. Convention:
    H = -sum over pairs J_ij S_i.S_j, each pair once; J > 0 is ferromagnetic.
    """
    d1 = (e_afm_1 - e_fm_1) / n_eu          # = 8 S^2 J1
    d2 = (e_afm_2 - e_fm_2) / n_eu          # = 6 S^2 (J1 + J2)
    j1 = d1 / (8 * spin ** 2)
    j2 = d2 / (6 * spin ** 2) - j1
    return j1, j2
