"""
Tests for the input writer and output readers in qe_tools.py.

These use small hand-made files in the same format QE writes, so they run in
a few seconds and do not need Quantum ESPRESSO installed.

Run with:  pytest
"""

import numpy as np
import pytest
from ase.build import bulk

import eus
import qe_tools as qe

HA = qe.HARTREE_TO_EV


# ---------------------------------------------------------------------------
# Helpers that build minimal QE-style output files
# ---------------------------------------------------------------------------

def write_xml(path, eigenvalues, occupations, etot=-15.8, lsda=False, nbnd_up=None,
              fermi=None, magnetization=None, stress=None):
    """Write a stripped-down data-file-schema.xml with only the tags qe_tools reads."""
    ks = ""
    for e, f in zip(eigenvalues, occupations):
        ks += (
            "<ks_energies>"
            "<k_point weight='1.0'>0.0 0.0 0.0</k_point>"
            f"<eigenvalues size='{len(e)}'>{' '.join(map(str, e))}</eigenvalues>"
            f"<occupations size='{len(f)}'>{' '.join(map(str, f))}</occupations>"
            "</ks_energies>"
        )
    bands = f"<lsda>{'true' if lsda else 'false'}</lsda>"
    if lsda:
        bands += f"<nbnd_up>{nbnd_up}</nbnd_up><nbnd_dw>{nbnd_up}</nbnd_dw>"
    else:
        bands += f"<nbnd>{len(eigenvalues[0])}</nbnd>"
    if fermi is not None:
        bands += f"<fermi_energy>{fermi}</fermi_energy>"
    extra = ""
    if magnetization is not None:
        extra += f"<magnetization><total>{magnetization}</total></magnetization>"
    if stress is not None:
        extra += f"<stress rank='2' dims='3 3'>{' '.join(map(str, np.ravel(stress)))}</stress>"
    path.write_text(
        "<?xml version='1.0'?><qes:espresso xmlns:qes='x'><output>"
        f"<band_structure>{bands}{ks}</band_structure>"
        f"<total_energy><etot>{etot}</etot></total_energy>{extra}"
        "</output></qes:espresso>"
    )
    return path


# ---------------------------------------------------------------------------
# Input writer
# ---------------------------------------------------------------------------

def test_input_has_counts_and_mesh():
    text = qe.make_pw_input(
        bulk("Si", "diamond", a=5.43),
        namelists={"control": {"calculation": "scf", "tstress": True},
                   "system": {"ecutwfc": 40}},
        species={"Si": (28.085, "Si.upf")},
        kpoints=(4, 4, 4),
    )
    assert "nat = 2" in text
    assert "ntyp = 1" in text
    assert "tstress = .true." in text          # Python bool -> Fortran logical
    assert "calculation = 'scf'" in text       # strings are quoted
    assert "K_POINTS automatic\n  4 4 4 0 0 0" in text
    positions = text.split("ATOMIC_POSITIONS crystal\n")[1].split("K_POINTS")[0]
    assert positions.count("  Si ") == 2


def test_numpy_values_written_as_plain_numbers():
    text = qe.make_pw_input(
        bulk("Si", "diamond", a=5.43),
        namelists={"system": {"ecutwfc": np.int64(40), "degauss": np.float64(0.01)},
                   "control": {"tstress": np.bool_(True)}},
        species={"Si": (28.085, "Si.upf")},
        kpoints=(4, 4, 4),
    )
    assert "ecutwfc = 40\n" in text
    assert "degauss = 0.01\n" in text
    assert "tstress = .true." in text
    assert "np." not in text


def test_input_explicit_kpoints_and_extra_card():
    kpts = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    text = qe.make_pw_input(
        eus.structure(),
        namelists={"control": {"calculation": "bands"}},
        species=eus.SPECIES,
        kpoints=kpts,
        extra_cards=eus.hubbard_card(7.0),
    )
    assert "K_POINTS crystal\n  3\n" in text
    assert text.rstrip().endswith("U Eu-4f 7.0")


def test_hubbard_card_off_for_zero_u():
    assert eus.hubbard_card(0) == ""
    assert "ortho-atomic" in eus.hubbard_card(5.0)


# ---------------------------------------------------------------------------
# XML reader
# ---------------------------------------------------------------------------

def test_read_nonmagnetic_insulator(tmp_path):
    # two k-points, four bands, two of them filled; energies in Hartree
    eig = [[-0.2, 0.1, 0.25, 0.4], [-0.1, 0.12, 0.3, 0.45]]
    occ = [[1, 1, 0, 0], [1, 1, 0, 0]]
    r = qe.read_xml(write_xml(tmp_path / "si.xml", eig, occ, etot=-15.8))

    assert r["eigenvalues"].shape == (1, 2, 4)
    assert r["energy"] == pytest.approx(-15.8 * HA)
    assert r["magnetization"] == 0.0
    assert r["pressure"] is None

    vbm, cbm, gap = qe.band_edges(r)
    assert vbm == pytest.approx(0.12 * HA)
    assert cbm == pytest.approx(0.25 * HA)
    assert gap == pytest.approx(0.13 * HA)
    assert qe.band_edges_by_count(r["eigenvalues"], 2)[2] == pytest.approx(gap)


def test_read_spin_polarized_splits_up_and_down(tmp_path):
    # QE lists the 3 spin-up bands then the 3 spin-down bands for each k-point
    eig = [[-0.3, -0.1, 0.2, -0.25, 0.15, 0.3]]
    occ = [[1, 1, 0, 1, 0, 0]]
    r = qe.read_xml(write_xml(tmp_path / "eus.xml", eig, occ, lsda=True, nbnd_up=3,
                              fermi=0.0, magnetization=7.0))

    assert r["eigenvalues"].shape == (2, 1, 3)
    np.testing.assert_allclose(r["eigenvalues"][0, 0], np.array([-0.3, -0.1, 0.2]) * HA)
    np.testing.assert_allclose(r["eigenvalues"][1, 0], np.array([-0.25, 0.15, 0.3]) * HA)
    assert r["magnetization"] == 7.0
    assert r["fermi"] == 0.0


def test_pressure_sign_and_units(tmp_path):
    # QE's stress is minus the pressure: a compressed crystal has negative diagonal stress
    sigma = -1e-5 * np.eye(3)                       # Hartree / bohr^3
    r = qe.read_xml(write_xml(tmp_path / "p.xml", [[0.0]], [[1.0]], stress=sigma))
    assert r["pressure"] == pytest.approx(1e-5 * qe.HARTREE_PER_BOHR3_TO_KBAR)
    assert r["pressure"] == pytest.approx(2.942, abs=1e-3)   # kbar


# ---------------------------------------------------------------------------
# Projected DOS reader
# ---------------------------------------------------------------------------

def test_read_pdos_labels_and_spin(tmp_path):
    e = np.linspace(-5, 5, 11)
    up, down = np.full_like(e, 2.0), np.full_like(e, 0.5)
    np.savetxt(tmp_path / "eus.pdos_tot", np.column_stack([e, up, down, up, down]))
    # local DOS columns are: energy, ldos up, ldos down, then one column per m and spin
    for name in ["eus.pdos_atm#1(Eu)_wfc#6(f)", "eus.pdos_atm#1(Eu)_wfc#5(d)",
                 "eus.pdos_atm#2(S)_wfc#2(p)"]:
        np.savetxt(tmp_path / name, np.column_stack([e, up, down, up, down]))

    energies, total, proj = qe.read_pdos(tmp_path, "eus", eus.SPECIES)

    np.testing.assert_allclose(energies, e)
    assert set(proj) == {"Eu 4f", "Eu 5d", "S 3p"}   # names come from the pseudopotentials
    np.testing.assert_allclose(total[0], up)
    np.testing.assert_allclose(total[1], -down)      # spin down plotted below the axis
    np.testing.assert_allclose(proj["Eu 4f"][1], -down)
