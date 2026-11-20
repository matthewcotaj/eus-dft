"""
Shared setup for the EuS calculations: structure, pseudopotentials and the
pw.x settings every EuS script uses.

EuS has the rocksalt structure (like NaCl): Eu²⁺ and S²⁻ on two interleaved
fcc lattices. The primitive cell holds one Eu and one S. Eu²⁺ has a half-filled
4f shell (4f⁷), so with all seven spins aligned the cell carries 7 Bohr magnetons.
"""

from ase.build import bulk

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
