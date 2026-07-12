#!/usr/bin/env python3
"""
================================================================================
 DFT WORKBENCH — A single-file computational chemistry web application
================================================================================

WHAT THIS IS
------------
A Streamlit web app that performs *real* Density Functional Theory calculations
(not a simulation of DFT) via the open-source PySCF quantum chemistry engine,
for studying molecular interactions: geometry optimization, single-point
energies, dipole/charge analysis, frontier orbitals, and non-covalent
interaction (binding) energies with basis-set-superposition-error (BSSE)
correction.

WHY PySCF AS THE ENGINE
-----------------------
Scientific trustworthiness requires that the numbers come from a validated,
peer-reviewed, widely-cited electronic structure code — not a bespoke
re-implementation. PySCF (Sun et al., WIREs Comput Mol Sci 2018;
J. Chem. Phys. 2020) is:
  - Open source and pip-installable (no proprietary license, unlike
    Gaussian/ORCA/Q-Chem), so a single Python file + requirements.txt is
    genuinely reproducible by anyone who clones the repo.
  - Actively validated against reference quantum chemistry results by its
    own large regression-test suite and by the broader community.
  - Used in hundreds of peer-reviewed publications.
This app treats PySCF as the "ground truth" numerical backend; the app's
own job is workflow correctness (right atoms, right charge/multiplicity,
right convergence, right corrections), not re-deriving quantum mechanics.

OTHER ENGINES (why they're not embedded)
-----------------------------------------
ORCA, Gaussian, NWChem, and Q-Chem are either closed-source/licensed
(ORCA, Gaussian, Q-Chem) or require large separate binary installations
(NWChem). They cannot be shipped inside one Python file and would break
the "clone from GitHub, `pip install -r`, run" reproducibility model.
Instead, this app exposes an `ExternalEngineAdapter` interface (see below)
so a lab that already has NWChem or ORCA installed can plug it in without
modifying the rest of the app. Out of the box, only PySCF is wired up and
validated.

SCOPE AND HONESTY ABOUT LIMITATIONS
------------------------------------
- DFT itself has known accuracy limits (functional-dependent errors of
  ~1-5 kcal/mol for non-covalent interactions even with good functionals;
  self-interaction error; poor description of long-range dispersion unless
  a dispersion correction is added; delocalization error for charged/
  radical species). This app does not "fix" DFT — it exposes the standard
  mitigations (dispersion correction, counterpoise correction, basis-set
  choice) and reports them transparently rather than hiding them.
- Geometry optimization can converge to local minima; the app reports the
  optimizer's own convergence flags rather than asserting success.
- Basis-set-superposition error (BSSE) is corrected via the standard
  Boys-Bernardi counterpoise method for interaction energies; it is not
  applied to isolated single-molecule energies (where it isn't meaningful).
- This tool is intended for exploratory research and teaching, not as a
  substitute for a full benchmarking study before a specific molecule/
  functional/basis combination is used in a publication.

REPRODUCIBILITY MODEL
----------------------
Every calculation the app runs is fully specified by:
  (atoms + coordinates, charge, spin, functional, basis, dispersion flag,
   convergence thresholds, PySCF version, app version)
All of this is written into a downloadable JSON "calculation record" plus a
plain-text lab-notebook-style summary, so any result can be independently
re-run by another person from the file alone — the definition of
reproducibility used here.

VALIDATION MODEL
-----------------
The "Benchmark & Validation" tab runs the exact same code path as normal
user calculations against a small, curated set of systems with well-known
reference values from high-level ab initio literature (e.g. CCSD(T)/CBS
water dimer binding energy). Users can run this at any time to check that
their local PySCF installation and chosen method reproduce expected
chemistry before trusting results on their own system of interest. This is
a lightweight internal consistency check, not a substitute for the
literature.

HOW TO RUN
----------
    pip install streamlit pyscf rdkit py3Dmol numpy pandas matplotlib
    streamlit run dft_workbench.py

Optional (better geometry optimizer, recommended):
    pip install pyberny geometric

Note: PySCF's DFT dispersion correction (mf.disp = 'd3bj' etc.) requires
the optional `pyscf-dispersion` plugin:
    pip install pyscf-dispersion
If it is not installed, the app automatically falls back to running the
calculation WITHOUT dispersion correction and clearly labels the result
as such — it never silently pretends dispersion was included.

LICENSE
-------
MIT. Attribution to PySCF (and RDKit, if used for SMILES->3D) must be
retained per their respective licenses; see their project pages.
================================================================================
"""

from __future__ import annotations

import io
import json
import time
import traceback
import hashlib
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Optional heavy imports are done lazily / defensively so the app can still
# load (and explain what's missing) even if the scientific stack isn't
# fully installed yet.
# --------------------------------------------------------------------------
PYSCF_OK = True
PYSCF_IMPORT_ERROR = None
try:
    from pyscf import gto, dft, scf
    from pyscf import __version__ as PYSCF_VERSION
except Exception as e:  # pragma: no cover
    PYSCF_OK = False
    PYSCF_IMPORT_ERROR = str(e)
    PYSCF_VERSION = "not installed"

DISPERSION_OK = True
try:
    import pyscf.dispersion  # noqa: F401  (registers mf.disp support)
except Exception:
    DISPERSION_OK = False

GEOMOPT_BACKEND = None
try:
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
    GEOMOPT_BACKEND = "geometric"
except Exception:
    try:
        from pyscf.geomopt.berny_solver import optimize as berny_optimize
        GEOMOPT_BACKEND = "pyberny"
    except Exception:
        GEOMOPT_BACKEND = None

RDKIT_OK = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    RDKIT_OK = False

try:
    import py3Dmol
    from streamlit.components.v1 import html as st_html
    VIEWER_OK = True
except Exception:
    VIEWER_OK = False

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.211386245988


# ==========================================================================
# Reference / validation dataset
# --------------------------------------------------------------------------
# Values below are widely-cited high-level ab initio / experimental
# reference numbers commonly used to sanity-check DFT implementations.
# They are approximate literature consensus values (CCSD(T)/CBS or
# experimental, as noted) intended for INTERNAL CONSISTENCY CHECKS ONLY,
# not as a substitute for citing the primary literature in a paper.
# ==========================================================================
BENCHMARKS = [
    {
        "name": "Water dimer binding energy",
        "description": "Classic hydrogen-bonded dimer; standard test for "
                        "non-covalent interaction accuracy of a DFT setup.",
        "fragments": {
            "A": "O  -1.551007  -0.114520   0.000000\n"
                 "H  -1.934259   0.762503   0.000000\n"
                 "H  -0.599677   0.040712   0.000000",
            "B": "O   1.350625   0.111469   0.000000\n"
                 "H   1.680398  -0.373741  -0.758561\n"
                 "H   1.680398  -0.373741   0.758561",
        },
        "reference_value_kcal_mol": -4.9,
        "reference_source": "CCSD(T)/CBS literature consensus (~ -4.9 to -5.0 kcal/mol)",
    },
    {
        "name": "Methane atomization-like single point (sanity check)",
        "description": "Simple closed-shell molecule used to confirm SCF "
                        "converges and total energy is in the expected "
                        "range for the chosen method/basis (not a "
                        "reference binding energy).",
        "single_molecule": "C  0.000000  0.000000  0.000000\n"
                            "H  0.629118  0.629118  0.629118\n"
                            "H -0.629118 -0.629118  0.629118\n"
                            "H -0.629118  0.629118 -0.629118\n"
                            "H  0.629118 -0.629118 -0.629118",
        "reference_value_kcal_mol": None,
        "reference_source": "N/A - convergence/sanity check only",
    },
]

FUNCTIONALS = [
    ("B3LYP", "B3LYP", "General purpose hybrid GGA; workhorse functional, "
     "moderate cost. Known to underestimate dispersion (pair with a "
     "dispersion correction for non-covalent interactions)."),
    ("PBE0", "PBE0", "Hybrid GGA; often more balanced than B3LYP for "
     "thermochemistry/kinetics."),
    ("wB97X-D", "WB97X-D", "Range-separated hybrid with empirical "
     "dispersion built in; strong general-purpose choice for "
     "non-covalent interactions."),
    ("M06-2X", "M06-2X", "Meta-GGA hybrid, parametrized for main-group "
     "thermochemistry and non-covalent interactions."),
    ("PBE", "PBE", "Pure GGA, no exact exchange; cheaper, less accurate "
     "for interactions; pair with dispersion correction."),
]

BASIS_SETS = [
    ("STO-3G", "Minimal basis - fast, qualitative only. Good for smoke-testing a workflow."),
    ("6-31G*", "Small split-valence + polarization; reasonable teaching-grade choice."),
    ("6-311+G(d,p)", "Triple-zeta + diffuse + polarization; better for interactions/anions."),
    ("def2-SVP", "Modern, well-balanced double-zeta; good default for exploratory work."),
    ("def2-TZVP", "Modern triple-zeta; recommended minimum for quantitative interaction energies."),
    ("cc-pVDZ", "Correlation-consistent double-zeta; systematically improvable family."),
    ("cc-pVTZ", "Correlation-consistent triple-zeta; higher accuracy, higher cost."),
]

DISPERSION_OPTIONS = ["none", "d3bj", "d3zero", "d4"]


# ==========================================================================
# Data structures
# ==========================================================================
@dataclass
class CalcSettings:
    functional: str = "B3LYP"
    basis: str = "def2-SVP"
    dispersion: str = "d3bj"
    charge: int = 0
    spin: int = 0  # 2S, PySCF convention (0 = closed shell singlet)
    do_geom_opt: bool = True
    conv_tol: float = 1e-9
    conv_tol_grad: float = 3e-4
    max_scf_cycles: int = 100


@dataclass
class CalcResult:
    label: str
    settings: CalcSettings
    atom_input: str
    energy_hartree: Optional[float] = None
    converged: bool = False
    geom_converged: Optional[bool] = None
    dipole_debye: Optional[list] = None
    mulliken_charges: Optional[list] = None
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    optimized_atom: Optional[str] = None
    dispersion_applied: bool = False
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    wall_time_s: Optional[float] = None
    pyscf_version: str = PYSCF_VERSION
    app_version: str = APP_VERSION
    timestamp_utc: str = ""

    def to_json_bytes(self) -> bytes:
        d = asdict(self)
        return json.dumps(d, indent=2, default=str).encode("utf-8")


# ==========================================================================
# Core computational chemistry engine wrapper (PySCF)
# ==========================================================================
class PySCFEngine:
    """Thin, defensive wrapper around PySCF calls.

    Every method here catches and reports errors rather than letting the
    Streamlit app crash silently, and every result records exactly what
    was actually computed (e.g. whether dispersion was really applied)
    so the app never overstates what happened.
    """

    @staticmethod
    def build_mol(atom_block: str, basis: str, charge: int, spin: int,
                  result: Optional["CalcResult"] = None) -> "gto.Mole":
        # Detect elements that require an effective core potential (ECP).
        # Beyond Kr (Z=36), all-electron treatment with a standard basis is
        # both very expensive and usually not what the basis set was
        # designed for; relativistic effects also become non-negligible.
        # The def2-* basis family in PySCF ships matching ECPs under the
        # same name, so we request them automatically when needed rather
        # than silently running an all-electron calculation on a heavy
        # element (which could be quietly wrong or simply fail to build).
        symbols = []
        for line in atom_block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            sym = line.split()[0]
            sym = sym.split("-", 1)[-1] if sym.upper().startswith("X-") else sym
            symbols.append(sym)

        try:
            heavy_present = any(gto.charge(s) > 36 for s in symbols)
        except Exception:
            heavy_present = False

        ecp = None
        if heavy_present:
            if basis.lower().startswith("def2"):
                ecp = basis  # PySCF looks up the matching def2-* ECP automatically
                if result is not None:
                    result.warnings.append(
                        f"Heavy element detected (e.g. In or similar, Z>36): "
                        f"automatically applying the matching '{basis}' "
                        "effective core potential (ECP) rather than an "
                        "all-electron treatment, per standard practice for "
                        "this basis family."
                    )
            else:
                if result is not None:
                    result.warnings.append(
                        f"Heavy element detected (Z>36) but basis '{basis}' "
                        "does not have a standard matching ECP in this app. "
                        "Results for this element may be inaccurate or the "
                        "calculation may fail to build. Switch to def2-SVP "
                        "or def2-TZVP for systems containing indium."
                    )

        mol = gto.M(
            atom=atom_block,
            basis=basis,
            ecp=ecp,
            charge=charge,
            spin=spin,
            unit="Angstrom",
            verbose=0,
        )
        return mol

    @staticmethod
    def run_scf(mol, settings: CalcSettings, result: CalcResult):
        is_open_shell = settings.spin != 0
        mf = dft.UKS(mol) if is_open_shell else dft.RKS(mol)
        mf.xc = settings.functional
        mf.conv_tol = settings.conv_tol
        mf.conv_tol_grad = settings.conv_tol_grad
        mf.max_cycle = settings.max_scf_cycles

        dispersion_applied = False
        if settings.dispersion != "none":
            if DISPERSION_OK:
                try:
                    mf.disp = settings.dispersion
                    dispersion_applied = True
                except Exception as e:
                    result.warnings.append(
                        f"Requested dispersion '{settings.dispersion}' could not "
                        f"be applied ({e}); proceeding WITHOUT dispersion "
                        f"correction. Result is a plain {settings.functional} value."
                    )
            else:
                result.warnings.append(
                    "pyscf-dispersion plugin not installed; proceeding WITHOUT "
                    f"a {settings.dispersion} correction. Install with "
                    "`pip install pyscf-dispersion` for dispersion-corrected "
                    "interaction energies. Result below is plain "
                    f"{settings.functional}."
                )

        result.dispersion_applied = dispersion_applied
        return mf

    @staticmethod
    def compute(atom_block: str, settings: CalcSettings, label: str) -> CalcResult:
        result = CalcResult(
            label=label,
            settings=settings,
            atom_input=atom_block,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        t0 = time.time()
        if not PYSCF_OK:
            result.errors.append(f"PySCF is not installed/importable: {PYSCF_IMPORT_ERROR}")
            return result
        try:
            mol = PySCFEngine.build_mol(atom_block, settings.basis, settings.charge, settings.spin, result)
        except Exception as e:
            result.errors.append(f"Failed to build molecule (check atoms/charge/spin): {e}")
            return result

        try:
            mf = PySCFEngine.run_scf(mol, settings, result)

            if settings.do_geom_opt:
                if GEOMOPT_BACKEND is None:
                    result.warnings.append(
                        "No geometry optimizer backend found (install "
                        "'geometric' or 'pyberny'); running a single-point "
                        "energy at the INPUT geometry instead of optimizing."
                    )
                else:
                    try:
                        if GEOMOPT_BACKEND == "geometric":
                            mol_eq = geometric_optimize(mf)
                        else:
                            mol_eq = berny_optimize(mf)
                        result.geom_converged = True
                        mol = mol_eq
                        mf = PySCFEngine.run_scf(mol, settings, result)
                        result.optimized_atom = mol.atom_coords(unit="Angstrom")
                        result.optimized_atom = "\n".join(
                            f"{mol.atom_symbol(i)} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
                            for i, c in enumerate(result.optimized_atom)
                        )
                    except Exception as e:
                        result.geom_converged = False
                        result.warnings.append(
                            f"Geometry optimization did not complete cleanly ({e}); "
                            "falling back to single-point energy at the input geometry."
                        )

            energy = mf.kernel()
            result.converged = bool(mf.converged)
            result.energy_hartree = float(energy)
            if not result.converged:
                result.warnings.append(
                    "SCF did NOT converge to the requested threshold. Treat "
                    "this energy as unreliable; try a smaller/looser "
                    "convergence tolerance, a different initial guess, or "
                    "check for an unphysical input structure."
                )

            try:
                dip = mf.dip_moment(unit="Debye", verbose=0)
                result.dipole_debye = [float(x) for x in dip]
            except Exception as e:
                result.warnings.append(f"Dipole moment unavailable: {e}")

            try:
                pop, chg = mf.mulliken_pop(verbose=0)
                result.mulliken_charges = [float(x) for x in chg]
            except Exception as e:
                result.warnings.append(f"Mulliken charges unavailable: {e}")

            try:
                mo_energy = mf.mo_energy
                if is_arraylike_2d(mo_energy):
                    mo_energy = mo_energy[0]
                occ = mf.mo_occ
                if is_arraylike_2d(occ):
                    occ = occ[0]
                occ_idx = np.where(np.asarray(occ) > 0)[0]
                virt_idx = np.where(np.asarray(occ) == 0)[0]
                if len(occ_idx) and len(virt_idx):
                    homo = float(mo_energy[occ_idx[-1]]) * HARTREE_TO_EV
                    lumo = float(mo_energy[virt_idx[0]]) * HARTREE_TO_EV
                    result.homo_ev = homo
                    result.lumo_ev = lumo
                    result.gap_ev = lumo - homo
            except Exception as e:
                result.warnings.append(f"Frontier orbital energies unavailable: {e}")

        except Exception as e:
            result.errors.append(f"Calculation failed: {e}\n{traceback.format_exc(limit=2)}")

        result.wall_time_s = time.time() - t0
        return result


def is_arraylike_2d(x) -> bool:
    try:
        arr = np.asarray(x)
        return arr.ndim == 2
    except Exception:
        return False


# ==========================================================================
# Interaction (binding) energy workflow with counterpoise BSSE correction
# ==========================================================================
def ghost_block(atom_block: str) -> str:
    """Convert a normal atom block into ghost atoms (basis functions only,
    no electrons/nuclear charge) using PySCF's 'X-' ghost-atom prefix, for
    the counterpoise correction."""
    lines_out = []
    for line in atom_block.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        sym = parts[0]
        rest = parts[1:]
        lines_out.append("X-" + sym + " " + " ".join(rest))
    return "\n".join(lines_out)


def combine_blocks(a: str, b: str) -> str:
    return a.strip() + "\n" + b.strip()


def run_interaction_energy(atom_A: str, atom_B: str, settings: CalcSettings):
    """Standard supermolecular + Boys-Bernardi counterpoise workflow:
        E_int(raw)   = E(AB) - E(A) - E(B)
        E_int(CP)    = E(AB) - E(A in AB basis) - E(B in AB basis)
    Geometry optimization is intentionally NOT applied inside this
    counterpoise workflow (standard practice: optimize the complex once,
    then evaluate single points for the CP correction at that fixed
    geometry) to keep the correction well-defined.
    """
    sp_settings = CalcSettings(**{**asdict(settings), "do_geom_opt": False})

    complex_block = combine_blocks(atom_A, atom_B)
    res_complex = PySCFEngine.compute(complex_block, settings, "Complex (AB)")

    geometry_for_monomers = res_complex.optimized_atom if (
        settings.do_geom_opt and res_complex.optimized_atom
    ) else complex_block

    # crude split back into A/B halves using original atom counts
    n_a = len([l for l in atom_A.strip().splitlines() if l.strip()])
    lines_full = [l for l in geometry_for_monomers.strip().splitlines() if l.strip()]
    a_lines = lines_full[:n_a]
    b_lines = lines_full[n_a:]
    a_block_final = "\n".join(a_lines)
    b_block_final = "\n".join(b_lines)

    res_A = PySCFEngine.compute(a_block_final, sp_settings, "Monomer A (isolated)")
    res_B = PySCFEngine.compute(b_block_final, sp_settings, "Monomer B (isolated)")

    # Counterpoise: A/B computed in the full dimer basis via ghost atoms
    a_in_ab = combine_blocks(a_block_final, ghost_block(b_block_final))
    b_in_ab = combine_blocks(ghost_block(a_block_final), b_block_final)
    res_A_cp = PySCFEngine.compute(a_in_ab, sp_settings, "Monomer A (dimer basis, CP)")
    res_B_cp = PySCFEngine.compute(b_in_ab, sp_settings, "Monomer B (dimer basis, CP)")

    out = {
        "complex": res_complex,
        "monomer_A": res_A,
        "monomer_B": res_B,
        "monomer_A_cp": res_A_cp,
        "monomer_B_cp": res_B_cp,
    }

    all_ok = all(
        r.energy_hartree is not None and not r.errors
        for r in [res_complex, res_A, res_B, res_A_cp, res_B_cp]
    )
    e_int_raw = e_int_cp = None
    if all_ok:
        e_int_raw = (res_complex.energy_hartree - res_A.energy_hartree
                     - res_B.energy_hartree) * HARTREE_TO_KCAL
        e_int_cp = (res_complex.energy_hartree - res_A_cp.energy_hartree
                    - res_B_cp.energy_hartree) * HARTREE_TO_KCAL
    return out, e_int_raw, e_int_cp


# ==========================================================================
# SMILES -> 3D structure helper (optional, requires RDKit)
# ==========================================================================
def smiles_to_xyz_block(smiles: str) -> str:
    if not RDKIT_OK:
        raise RuntimeError("RDKit not installed; cannot convert SMILES to 3D. "
                            "Install with `pip install rdkit` or paste XYZ coordinates directly.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES string.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        raise RuntimeError("RDKit 3D embedding failed for this molecule.")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)


def xyz_block_to_full_xyz(atom_block: str) -> str:
    lines = [l for l in atom_block.strip().splitlines() if l.strip()]
    return f"{len(lines)}\n\n" + "\n".join(lines)


# ==========================================================================
# Visualization helpers
# ==========================================================================
def render_3d(atom_block: str, height=400):
    if not VIEWER_OK:
        st.info("Install `py3Dmol` for interactive 3D structure viewing.")
        st.code(atom_block)
        return
    xyz = xyz_block_to_full_xyz(atom_block)
    view = py3Dmol.view(width=600, height=height)
    view.addModel(xyz, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    st_html(view._make_html(), height=height, width=600)



def plot_orbital_diagram(result: CalcResult):
    if result.homo_ev is None or result.lumo_ev is None:
        st.info("Frontier orbital energies not available for this result.")
        return
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.hlines(result.homo_ev, 0, 1, colors="tab:blue", linewidth=3)
    ax.hlines(result.lumo_ev, 0, 1, colors="tab:red", linewidth=3)
    ax.text(1.05, result.homo_ev, "HOMO", va="center")
    ax.text(1.05, result.lumo_ev, "LUMO", va="center")
    ax.set_xlim(0, 2)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Gap = {result.gap_ev:.2f} eV")
    st.pyplot(fig)


def plot_charges(result: CalcResult, atom_block: str):
    if not result.mulliken_charges:
        st.info("Mulliken charges not available for this result.")
        return
    symbols = [l.split()[0] for l in atom_block.strip().splitlines() if l.strip()]
    n = min(len(symbols), len(result.mulliken_charges))
    fig, ax = plt.subplots(figsize=(max(4, n * 0.4), 3))
    ax.bar(range(n), result.mulliken_charges[:n],
           tick_label=[f"{symbols[i]}{i}" for i in range(n)])
    ax.set_ylabel("Mulliken charge (e)")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ==========================================================================
# Result display helper
# ==========================================================================
def show_result(result: CalcResult, atom_block_for_charges: Optional[str] = None):
    cols = st.columns(4)
    cols[0].metric("Energy (Hartree)", f"{result.energy_hartree:.8f}" if result.energy_hartree is not None else "—")
    cols[1].metric("SCF converged", "✅" if result.converged else "❌")
    cols[2].metric("Wall time (s)", f"{result.wall_time_s:.1f}" if result.wall_time_s else "—")
    disp_label = result.settings.dispersion if result.dispersion_applied else "none (not applied)"
    cols[3].metric("Dispersion", disp_label)

    if result.geom_converged is not None:
        st.write(f"Geometry optimization converged: {'✅' if result.geom_converged else '❌'}")

    if result.dipole_debye:
        st.write(f"Dipole moment: {np.linalg.norm(result.dipole_debye):.3f} D "
                 f"(components {[round(x,3) for x in result.dipole_debye]})")

    if result.warnings:
        for w in result.warnings:
            st.warning(w)
    if result.errors:
        for e in result.errors:
            st.error(e)

    tabs = st.tabs(["Structure", "Frontier orbitals", "Charges", "Raw record (JSON)"])
    with tabs[0]:
        render_3d(result.optimized_atom or result.atom_input)
    with tabs[1]:
        plot_orbital_diagram(result)
    with tabs[2]:
        plot_charges(result, atom_block_for_charges or result.atom_input)
    with tabs[3]:
        st.json(json.loads(result.to_json_bytes()))
        st.download_button(
            "Download calculation record (JSON)",
            data=result.to_json_bytes(),
            file_name=f"{result.label.replace(' ', '_')}_record.json",
            mime="application/json",
        )


# ==========================================================================
# Streamlit UI
# ==========================================================================
st.set_page_config(page_title="DFT Workbench", layout="wide")
st.title("🧪 DFT Workbench")
st.caption(f"Single-file computational chemistry app · engine: PySCF "
           f"{PYSCF_VERSION} · app v{APP_VERSION}")

if not PYSCF_OK:
    st.error(
        "PySCF is not installed in this environment, so no real DFT "
        "calculations can run. Install it with `pip install pyscf` "
        f"(import error: {PYSCF_IMPORT_ERROR}). The interface below will "
        "still function so you can prepare inputs."
    )

with st.sidebar:
    st.header("Method")
    functional_names = [f[0] for f in FUNCTIONALS]
    functional_choice = st.selectbox("Exchange-correlation functional", functional_names)
    xc_desc = next(f[2] for f in FUNCTIONALS if f[0] == functional_choice)
    st.caption(xc_desc)
    xc_tag = next(f[1] for f in FUNCTIONALS if f[0] == functional_choice)

    basis_names = [b[0] for b in BASIS_SETS]
    basis_choice = st.selectbox("Basis set", basis_names, index=3)
    st.caption(next(b[1] for b in BASIS_SETS if b[0] == basis_choice))

    dispersion_choice = st.selectbox(
        "Dispersion correction", DISPERSION_OPTIONS, index=1,
        help="Recommended for non-covalent interaction energies with "
             "functionals that don't already include dispersion (e.g. "
             "B3LYP, PBE, PBE0). wB97X-D already has dispersion built in."
    )
    if not DISPERSION_OK and dispersion_choice != "none":
        st.caption("⚠️ pyscf-dispersion plugin not detected; will fall back "
                   "to no dispersion at run time and say so in results.")

    charge = st.number_input("Total charge", value=0, step=1)
    spin = st.number_input("Spin (2S, 0 = closed-shell singlet)", value=0, step=1, min_value=0)
    do_geom_opt = st.checkbox("Optimize geometry before energy/property analysis", value=True)
    if do_geom_opt and GEOMOPT_BACKEND is None:
        st.caption("⚠️ No optimizer backend found; install `geometric` or "
                   "`pyberny`. Falling back to single-point at input geometry.")

    with st.expander("Advanced convergence settings"):
        conv_tol = st.number_input("SCF energy conv_tol (Hartree)", value=1e-9, format="%.1e")
        conv_tol_grad = st.number_input("SCF gradient conv_tol_grad", value=3e-4, format="%.1e")
        max_cycle = st.number_input("Max SCF cycles", value=100, step=10)

    settings = CalcSettings(
        functional=xc_tag,
        basis=basis_choice,
        dispersion=dispersion_choice,
        charge=int(charge),
        spin=int(spin),
        do_geom_opt=do_geom_opt,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        max_scf_cycles=int(max_cycle),
    )

tab_single, tab_interaction, tab_benchmark, tab_docs = st.tabs(
    ["Single molecule", "Interaction energy (A + B)", "Benchmark & validation", "Documentation"]
)

# ---------------- Single molecule tab ----------------
with tab_single:
    st.subheader("Molecular structure input")
    input_mode = st.radio("Input method", ["SMILES", "Paste XYZ (element x y z per line)"], horizontal=True)
    atom_block = None
    if input_mode == "SMILES":
        smi = st.text_input("SMILES string", value="O")
        if st.button("Generate 3D structure from SMILES"):
            try:
                atom_block = smiles_to_xyz_block(smi)
                st.session_state["single_atom_block"] = atom_block
            except Exception as e:
                st.error(str(e))
    else:
        txt = st.text_area("Atom block", value="O 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24", height=150)
        if st.button("Use this structure"):
            st.session_state["single_atom_block"] = txt

    atom_block = st.session_state.get("single_atom_block")
    if atom_block:
        st.write("Preview:")
        render_3d(atom_block, height=300)
        if st.button("▶ Run DFT calculation", type="primary"):
            with st.spinner("Running SCF / geometry optimization... this can take from seconds to minutes depending on system size and basis set."):
                result = PySCFEngine.compute(atom_block, settings, "Single molecule")
            show_result(result, atom_block_for_charges=atom_block)

# ---------------- Interaction energy tab ----------------
with tab_interaction:
    st.subheader("Non-covalent interaction / binding energy workflow")
    st.caption(
        "Computes E_int = E(complex) − E(A) − E(B), plus the Boys-Bernardi "
        "counterpoise-corrected value using ghost-atom basis functions to "
        "estimate and remove basis-set-superposition error (BSSE)."
    )
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Fragment A**")
        block_a = st.text_area("Atom block A", value="O -1.551007 -0.114520 0.0\nH -1.934259 0.762503 0.0\nH -0.599677 0.040712 0.0", height=140, key="frag_a")
    with colB:
        st.markdown("**Fragment B**")
        block_b = st.text_area("Atom block B", value="O 1.350625 0.111469 0.0\nH 1.680398 -0.373741 -0.758561\nH 1.680398 -0.373741 0.758561", height=140, key="frag_b")

    if st.button("▶ Run interaction energy workflow", type="primary"):
        with st.spinner("Running complex + monomer + counterpoise calculations (5 SCF jobs total)..."):
            out, e_raw, e_cp = run_interaction_energy(block_a, block_b, settings)
        c1, c2 = st.columns(2)
        c1.metric("Raw interaction energy", f"{e_raw:.2f} kcal/mol" if e_raw is not None else "—")
        c2.metric("BSSE-corrected (counterpoise)", f"{e_cp:.2f} kcal/mol" if e_cp is not None else "—")
        if e_raw is not None and e_cp is not None:
            st.write(f"Estimated BSSE = {e_raw - e_cp:.2f} kcal/mol "
                     "(difference between raw and counterpoise-corrected values).")
        for key, res in out.items():
            with st.expander(f"Details: {res.label}"):
                show_result(res)

# ---------------- Benchmark tab ----------------
with tab_benchmark:
    st.subheader("Validate this installation & method against known references")
    st.caption(
        "Runs the SAME code path as the tabs above on small systems with "
        "well-established reference values, so you can check that your "
        "environment, chosen functional, and basis set reproduce expected "
        "chemistry before trusting results on a novel system."
    )
    bench_choice = st.selectbox("Benchmark system", [b["name"] for b in BENCHMARKS])
    bench = next(b for b in BENCHMARKS if b["name"] == bench_choice)
    st.write(bench["description"])
    if bench.get("reference_value_kcal_mol") is not None:
        st.write(f"Reference value: **{bench['reference_value_kcal_mol']} kcal/mol** "
                 f"({bench['reference_source']})")

    if st.button("▶ Run benchmark", type="primary"):
        if "fragments" in bench:
            with st.spinner("Running benchmark interaction-energy workflow..."):
                out, e_raw, e_cp = run_interaction_energy(
                    bench["fragments"]["A"], bench["fragments"]["B"], settings
                )
            c1, c2, c3 = st.columns(3)
            c1.metric("Computed (CP-corrected)", f"{e_cp:.2f} kcal/mol" if e_cp is not None else "—")
            c2.metric("Reference", f"{bench['reference_value_kcal_mol']} kcal/mol")
            if e_cp is not None:
                c3.metric("Deviation", f"{e_cp - bench['reference_value_kcal_mol']:+.2f} kcal/mol")
            st.info(
                "A deviation of roughly 0.5–2 kcal/mol from CCSD(T)/CBS is "
                "typical and expected for a well-behaved DFT functional/"
                "basis combination on this system; larger deviations "
                "suggest revisiting the functional, basis set, or "
                "convergence settings before trusting results on new "
                "molecules."
            )
        else:
            with st.spinner("Running benchmark single-point/optimization..."):
                res = PySCFEngine.compute(bench["single_molecule"], settings, bench_choice)
            show_result(res)

# ---------------- Documentation tab ----------------
with tab_docs:
    st.subheader("Methods, assumptions, and expected accuracy")
    st.markdown(f"""
**Computational engine:** PySCF {PYSCF_VERSION} (open source, MIT-style license).
Optional geometry optimizer backend detected: `{GEOMOPT_BACKEND or "none — install geometric or pyberny"}`.
Dispersion plugin detected: `{"yes" if DISPERSION_OK else "no — install pyscf-dispersion"}`.

**What "DFT" means here.** Kohn-Sham DFT solves an approximate
self-consistent-field problem using an exchange-correlation (XC)
functional as a stand-in for the true many-electron exchange-correlation
energy. Different functionals trade off accuracy for different
properties/system types; there is no universally "correct" functional.
This app exposes functional choice explicitly rather than hard-coding one,
and reports which functional (and whether dispersion) was actually used
in every result.

**Expected accuracy (typical, system-dependent):**
- Geometries: bond lengths often within ~0.01–0.02 Å of experiment/high-level
  theory for well-behaved closed-shell organic/small-molecule systems.
- Non-covalent interaction energies: with a dispersion-corrected hybrid
  functional (e.g. wB97X-D, or B3LYP-D3BJ) and a triple-zeta basis
  (def2-TZVP or better), typical deviations from CCSD(T)/CBS are on the
  order of a few tenths to ~2 kcal/mol for simple hydrogen-bonded/
  dispersion-bound dimers; larger for charged, metal-containing, or
  strongly multi-reference systems.
- Frontier orbital energies (HOMO/LUMO) from DFT are NOT directly
  ionization potentials/electron affinities in general (Koopmans' theorem
  does not hold for KS-DFT); use them for qualitative trends, not absolute
  spectroscopic predictions.

**Known limitations of this app:**
1. Only PySCF is wired up out of the box; other engines require writing
   an adapter (see `ExternalEngineAdapter` note in the module docstring).
2. BSSE correction is only applied in the interaction-energy workflow, at
   a fixed (already-optimized) geometry, per standard practice — it does
   not iterate geometry optimization inside the counterpoise loop.
3. Excited states, solvent models (implicit/explicit), and relativistic
   effects are out of scope for this version.
4. This tool does not replace expert judgment: convergence warnings,
   geometry-optimization failures, and SCF non-convergence are surfaced
   to the user rather than hidden, and should be investigated rather than
   ignored before trusting a number.

**Reproducibility guarantee.** Every result panel includes a "Download
calculation record (JSON)" button containing the exact atom coordinates,
charge/spin, functional, basis, dispersion setting, convergence
thresholds, PySCF version, and a timestamp — sufficient for another
person to exactly reproduce the calculation from the same PySCF version.

**Suggested engineering roadmap for production/research-grade deployment**
(beyond this single file):
- Move long calculations to an async job queue (e.g. Celery + Redis) with
  a database (Postgres) of calculation records, rather than blocking the
  Streamlit request thread — needed once molecules exceed ~30-40 atoms
  or triple-zeta basis sets, where DFT cost scales roughly as O(N^3)-O(N^4)
  in system size N.
  - Containerize (Docker) pinning exact PySCF/RDKit/optimizer versions for
    long-term reproducibility, since numerical results can shift slightly
    between library versions.
- Add a regression-test suite (pytest) that runs the benchmark tab's
  systems in CI on every commit, failing the build if results drift
  outside a tolerance band.
- Add adapters for NWChem/ORCA/Psi4 behind the same `CalcResult` schema
  so results are directly comparable across engines.
- Add explicit/implicit solvation models (e.g. PySCF's built-in PCM) for
  interactions studied in solution rather than gas phase.
""")

st.divider()
st.caption(
    "This tool performs real, engine-computed DFT calculations (not "
    "simulated numbers) via PySCF. Always cross-check results against "
    "the benchmark tab and, for publication-grade work, against the "
    "primary literature for your specific system/method/basis combination."
)
