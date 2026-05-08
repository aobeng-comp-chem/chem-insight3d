"""
fchk_read.py
============
Reads a Gaussian formatted checkpoint (.fchk) file and exposes the same
interface that nbo_read.py provides for .47/.31 files, so that
chemview.py can call it uniformly.

Public API (mirrors nbo_read equivalents)
------------------------------------------
load_basis_from_fchk(fchk_path)
    → (final_norm_basis, coordinates_ang, atom_info)
    Parses basis set, normalises basis functions via the same pipeline
    used by nbo_read (convert_to_molden → normalize_by_self_overlap →
    iterative overlap normalisation).

get_orbital_count_fchk(fchk_path)
    → (orbital_type_str, nbas, is_open_shell)

get_orbital_energies_and_occupations_fchk(fchk_path)
    → (ene_alpha, occ_alpha, ene_beta, occ_beta)

load_cmos_from_fchk(fchk_path, orbital_indices, spin='alpha')
    → list of 1-D numpy arrays (one per requested orbital)

compute_cube_data_fchk(fchk_path, orbital_indices, spin,
                        grid_quality, ext_dist, bohr_const)
    → same list-of-dicts format as nbo_read.compute_cube_data()
"""

import re
import os
import math
import copy
import numpy as np
from scipy.constants import physical_constants

# ---------------------------------------------------------------------------
# Low-level fchk parser
# ---------------------------------------------------------------------------

def _parse_scalar(lines, name, dtype=float):
    """Parse a scalar (non-array) fchk entry: 'Name   R   value'."""
    for line in lines:
        if line.startswith(name):
            parts = line.split()
            try:
                return dtype(parts[-1])
            except (ValueError, IndexError):
                pass
    return None


def _parse_array(lines, name, dtype=float):
    """
    Parse an array fchk section:
        Name   [IRC]   N=   <count>
        <data lines...>
    Returns a numpy array, or None if section not found.
    """
    header_re = re.compile(
        r'^' + re.escape(name) + r'\s+[IRCL]\s+N=\s*(\d+)$'
    )
    data = []
    reading = False
    n_expected = None

    for line in lines:
        if not reading:
            m = header_re.match(line.rstrip())
            if m:
                n_expected = int(m.group(1))
                reading = True
        else:
            # Stop at the next section header
            if re.match(r'^[A-Za-z]', line.strip()) and line.strip():
                break
            tokens = line.strip().split()
            if dtype == float:
                data.extend(float(t) for t in tokens)
            elif dtype == int:
                data.extend(int(t) for t in tokens)
            else:
                data.extend(tokens)

    if not reading:
        return None

    arr = np.array(data, dtype=dtype)
    if n_expected is not None and arr.size != n_expected:
        raise ValueError(
            f"fchk section '{name}': expected {n_expected} values, got {arr.size}"
        )
    return arr


# ---------------------------------------------------------------------------
# Basis-set extraction from fchk
# ---------------------------------------------------------------------------

# Angular-momentum label maps matching nbo_read conventions
_COMP_MAP = {
    "s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11, "i": 13, "j": 15
}
_LABEL_MAP = {
    "s":  ["s"],
    "p":  ["px", "py", "pz"],
    "d":  ["d0", "dc1", "ds1", "dc2", "ds2"],
    "f":  ["f0", "fc1", "fs1", "fc2", "fs2", "fc3", "fs3"],
    "g":  ["g0", "gc1", "gs1", "gc2", "gs2", "gc3", "gs3", "gc4", "gs4"],
    "h":  ["h0", "hc1", "hs1", "hc2", "hs2", "hc3", "hs3",
           "hc4", "hs4", "hc5", "hs5"],
    "i":  ["i0", "ic1", "is1", "ic2", "is2", "ic3", "is3",
           "ic4", "is4", "ic5", "is5", "ic6", "is6"],
    "j":  ["j0", "jc1", "js1", "jc2", "js2", "jc3", "js3",
           "jc4", "js4", "jc5", "js5", "jc6", "js6", "jc7", "js7"],
}

# Gaussian shell-type integers → angular-momentum string
_TYPE_MAP = {
    0: "s",  1: "p",  2: "d",  3: "f",  4: "g",  5: "h",  6: "i",  7: "j",
    -1: "sp",
    -2: "d", -3: "f", -4: "g", -5: "h", -6: "i", -7: "j",
}


def _extract_basis_set(lines):
    """
    Parse fchk basis-set sections and return a list of basis-function dicts
    using the same field names as nbo_read:
        N, CENTER, shell_num, type, orb_val, exps, coeffs,
        xcenter, ycenter, zcenter
    Coordinates are in Angstrom (converted from bohr as stored in fchk).
    """
    bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10

    shell_types    = _parse_array(lines, "Shell types",                        int)
    num_primitives = _parse_array(lines, "Number of primitives per shell",     int)
    shell_to_atom  = _parse_array(lines, "Shell to atom map",                  int)
    prim_exponents = _parse_array(lines, "Primitive exponents",                float)
    contr_coeffs   = _parse_array(lines, "Contraction coefficients",           float)
    shell_coords   = _parse_array(lines, "Coordinates of each shell",          float)
    
    LABEL_MAPPING = {
    1: 's', 51: 's',
    101: 'px', 102: 'py', 103: 'pz',
    151: 'px', 152: 'py', 153: 'pz',
    251: 'ds2', 252: 'ds1', 253: 'dc1', 254: 'dc2', 255: 'd0',
    351: 'f0', 352: 'fc1', 353: 'fs1', 354: 'fc2', 355: 'fs2',
    356: 'fc3', 357: 'fs3',
    451: 'g0', 452: 'gc1', 453: 'gs1', 454: 'gc2', 455: 'gs2',
    456: 'gc3', 457: 'gs3', 458: 'gc4', 459: 'gs4',
    551: 'h0', 552: 'hc1', 553: 'hs1', 554: 'hc2', 555: 'hs2',
    556: 'hc3', 557: 'hs3', 558: 'hc4', 559: 'hs4', 560: 'hc5', 561: 'hs5',
    651: 'i0', 652: 'ic1', 653: 'is1', 654: 'ic2', 655: 'is2',
    656: 'ic3', 657: 'is3', 658: 'ic4', 659: 'is4',
    660: 'ic5', 661: 'is5', 662: 'ic6', 663: 'is6',
    751: 'j0', 752: 'jc1', 753: 'js1', 754: 'jc2', 755: 'js2',
    756: 'jc3', 757: 'js3', 758: 'jc4', 759: 'js4',
    760: 'jc5', 761: 'js5', 762: 'jc6', 763: 'js6', 764: 'jc7', 765: 'js7',
}
    
    ORBVAL_TO_LABELCODE = {}
    for code, lbl in LABEL_MAPPING.items():
        
        ORBVAL_TO_LABELCODE.setdefault(lbl, code)


    if any(a is None for a in [shell_types, num_primitives, shell_to_atom,
                                prim_exponents, contr_coeffs, shell_coords]):
        raise ValueError("fchk file is missing one or more required basis sections.")

    shell_coords = shell_coords.reshape(-1, 3) * bohr_to_ang   

    # P(S=P) coefficients (only present when SP shells exist)
    psp_coeffs = _parse_array(lines, "P(S=P) Contraction coefficients", float)

    basis = []
    prim_idx    = 0
    fn_counter  = 0

    for shell_idx, stype in enumerate(shell_types):
        shell_type = _TYPE_MAP.get(int(stype), f"type{stype}")
        n_prim     = int(num_primitives[shell_idx])
        atom_idx   = int(shell_to_atom[shell_idx])
        coords     = shell_coords[shell_idx]                    # Å
        exps       = prim_exponents[prim_idx:prim_idx + n_prim].tolist()
        s_coeffs   = contr_coeffs[prim_idx:prim_idx + n_prim].tolist()

        if shell_type == "sp":
            # S component
            basis.append({
                "N":         fn_counter + 1,
                "CENTER":    atom_idx,
                "shell_num": shell_idx + 1,
                "type":  "s",
                "orb_val":   "s",
                "exps":      exps,
                "coeffs":    s_coeffs,
                "xcenter":   float(coords[0]),
                "ycenter":   float(coords[1]),
                "zcenter":   float(coords[2]),
            })
            fn_counter += 1

            # P components — use P(S=P) coefficients
            if psp_coeffs is None:
                raise RuntimeError(
                    "SP shell found but 'P(S=P) Contraction coefficients' "
                    "section is missing from fchk file."
                )
            p_coeffs = psp_coeffs[prim_idx:prim_idx + n_prim].tolist()
            for label in ("px", "py", "pz"):
                basis.append({
                    "N":         fn_counter + 1,
                    "CENTER":    atom_idx,
                    "shell_num": shell_idx + 1,
                    "type":  label,
                    "orb_val":   label,
                    "exps":      exps,
                    "coeffs":    p_coeffs,
                    "xcenter":   float(coords[0]),
                    "ycenter":   float(coords[1]),
                    "zcenter":   float(coords[2]),
                })
                fn_counter += 1

        else:
            n_comp      = _COMP_MAP.get(shell_type, 1)
            comp_labels = _LABEL_MAP.get(shell_type, [shell_type] * n_comp)
            for j in range(n_comp):
                basis.append({
                    "N":         fn_counter + 1,
                    "CENTER":    atom_idx,
                    "shell_num": shell_idx + 1,
                    "type":  comp_labels[j],
                    "orb_val":   comp_labels[j],
                    "exps":      exps,
                    "coeffs":    s_coeffs,
                    "xcenter":   float(coords[0]),
                    "ycenter":   float(coords[1]),
                    "zcenter":   float(coords[2]),
                })
                fn_counter += 1

        prim_idx += n_prim
        
    for bf in basis:
        ov = bf["orb_val"]
        bf["LABEL"] = ORBVAL_TO_LABELCODE.get(ov)

    return basis


def _extract_atoms(lines):
    """
    Return atom coordinates and atomic numbers from the fchk file.

    fchk stores:
        'Atomic numbers'           I  N= <natoms>
        'Current cartesian coordinates'  R  N= 3*natoms   (in bohr)

    Returns:
        coordinates_ang : list of (x, y, z) in Angstrom
        atom_info       : list of (Z, x, y, z) in Angstrom
    """
    bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10

    atomic_nums = _parse_array(lines, "Atomic numbers", int)
    cart_coords = _parse_array(lines, "Current cartesian coordinates", float)

    if atomic_nums is None or cart_coords is None:
        raise ValueError(
            "fchk file is missing 'Atomic numbers' or "
            "'Current cartesian coordinates' sections."
        )

    natoms = len(atomic_nums)
    coords_bohr = cart_coords.reshape(natoms, 3)
    coords_ang  = coords_bohr * bohr_to_ang

    coordinates_ang = [tuple(row) for row in coords_ang]
    atom_info       = [(int(atomic_nums[i]),) + coordinates_ang[i]
                       for i in range(natoms)]
    return coordinates_ang, atom_info


# ---------------------------------------------------------------------------
# Normalisation (delegates to nbo_read pipeline)
# ---------------------------------------------------------------------------

def _normalise_basis(raw_basis):
    """
    Apply the same two-stage normalisation used by nbo_read.load_basis_headless:
        1. convert_to_molden  (primitive Gaussian normalisation)
        2. normalize_by_self_overlap  (contracted-function normalisation)

    Returns final_norm_basis ready for compute_cube_data.
    Note: fchk MOs are already in the orthonormal MO basis, so the additional
    iterative_basis_modification step used for .47 files is NOT needed here.
    """
    import nbo_read as _nr
    
    basis = raw_basis

    from overlap_matrix import get_overlap_matrix as getSmat
    from bas_dict import dict_keys
    
    Smat  = getSmat(basis, dict_keys, normalize_primitives=True, diagonal_only=False)
    basis = _nr.normalize_basis_info(basis, Smat)
    Smat  = getSmat(basis, dict_keys, normalize_primitives=True, diagonal_only=False)
    basis = _nr.normalize_by_self_overlap(basis)

    
    
    return raw_basis#basis


# ---------------------------------------------------------------------------
# Public API — mirrors nbo_read
# ---------------------------------------------------------------------------

def load_basis_from_fchk(fchk_path):
    """
    Parse and normalise the basis set from a .fchk file.

    Returns
    -------
    final_norm_basis : list of basis-function dicts (same format as nbo_read)
    coordinates_ang  : list of (x, y, z) in Angstrom, one per atom
    atom_info        : list of (Z, x, y, z) in Angstrom, one per atom
    """
    with open(fchk_path, "r") as f:
        lines = f.read().splitlines()

    raw_basis       = _extract_basis_set(lines)
    coordinates_ang, atom_info = _extract_atoms(lines)
    
    # print(raw_basis[:])
    final_norm_basis = _normalise_basis(raw_basis)
    print(final_norm_basis[:])
    return   final_norm_basis, coordinates_ang, atom_info


def get_orbital_count_fchk(fchk_path):
    """
    Return (orbital_type_str, nbas, is_open_shell) for a .fchk file.

    orbital_type_str is always 'CMO' (canonical MOs from fchk).
    nbas is the number of basis functions.
    is_open_shell is True when Beta MO coefficients are present.
    """
    with open(fchk_path, "r") as f:
        lines = f.read().splitlines()

    nbas_val = _parse_scalar(lines, "Number of basis functions", int)
    if nbas_val is None:
        # fall back: count from basis extraction
        raw_basis = _extract_basis_set(lines)
        nbas_val  = len(raw_basis)

    # Open-shell: fchk has separate Alpha and Beta MO coefficient blocks
    has_beta = any(
        line.startswith("Beta MO coefficients") for line in lines
    )
    return "CMO", int(nbas_val), has_beta


def get_orbital_energies_and_occupations_fchk(fchk_path):
    """
    Return (ene_alpha, occ_alpha, ene_beta, occ_beta) from a .fchk file.
    All energies are in Hartree.  Beta arrays are None for closed-shell.
    """
    with open(fchk_path, "r") as f:
        lines = f.read().splitlines()

    ene_alpha = _parse_array(lines, "Alpha Orbital Energies", float)
    occ_alpha = _parse_array(lines, "Alpha Orbital occupancies", float)

    # Closed-shell fchk stores only Alpha sections even for a RHF
    ene_beta  = _parse_array(lines, "Beta Orbital Energies", float)
    occ_beta  = _parse_array(lines, "Beta Orbital occupancies", float)

    # If occupancies not present, try to derive from electron count
    if occ_alpha is None and ene_alpha is not None:
        n_elec_alpha = _parse_scalar(lines, "Number of alpha electrons", int)
        nbas         = len(ene_alpha)
        occ_alpha    = np.zeros(nbas)
        if n_elec_alpha:
            occ_alpha[:n_elec_alpha] = 1.0

    if occ_beta is None and ene_beta is not None:
        n_elec_beta = _parse_scalar(lines, "Number of beta electrons", int)
        nbas        = len(ene_beta)
        occ_beta    = np.zeros(nbas)
        if n_elec_beta:
            occ_beta[:n_elec_beta] = 1.0

    return (
        ene_alpha if ene_alpha is not None else np.array([]),
        occ_alpha if occ_alpha is not None else np.array([]),
        ene_beta,
        occ_beta,
    )


def load_cmos_from_fchk(fchk_path, orbital_indices, spin="alpha"):
    """
    Load CMO row vectors for requested 1-based orbital_indices from a .fchk.

    Parameters
    ----------
    fchk_path       : str
    orbital_indices : list of int (1-based)
    spin            : 'alpha' or 'beta'

    Returns
    -------
    List of 1-D numpy arrays, one per requested orbital.
    """
    with open(fchk_path, "r") as f:
        lines = f.read().splitlines()

    _, nbas, is_open = get_orbital_count_fchk(fchk_path)

    if spin.lower().startswith("b") and is_open:
        section = "Beta MO coefficients"
    else:
        section = "Alpha MO coefficients"

    mo_flat = _parse_array(lines, section, float)
    if mo_flat is None:
        raise ValueError(
            f"Section '{section}' not found in {fchk_path}"
        )

    expected = nbas * nbas
    if mo_flat.size < expected:
        raise ValueError(
            f"Expected {expected} MO coefficients, got {mo_flat.size}"
        )

    # fchk stores MOs row-by-row: row i = orbital i coefficients
    mo_matrix = mo_flat[:expected].reshape(nbas, nbas)
    return [mo_matrix[i - 1] for i in orbital_indices]


def compute_cube_data_fchk(fchk_path, orbital_indices, spin,
                            grid_quality, ext_dist, bohr_const):
    """
    Compute orbital grids directly from a .fchk file.

    Parameters match nbo_read.compute_cube_data() exactly, except that
    fchk_path replaces (basis_path, key_filepath).

    Returns the same list-of-dicts format so _load_computed_cubes in
    chemview.py can handle both sources identically.
    """
    from angular_funct import ang_res_lamda

    try:
        import electron_density_opt_omp as _cpp
        _use_cpp = True
    except ImportError:
        _use_cpp = False

    final_norm_basis, coordinates_ang, atom_info = load_basis_from_fchk(fchk_path)
    cmos = load_cmos_from_fchk(fchk_path, orbital_indices, spin)

    coord_bohr = np.array(coordinates_ang) / bohr_const
    ext_min    = coord_bohr.min(axis=0) - ext_dist
    ext_max    = coord_bohr.max(axis=0) + ext_dist
    ranges     = ext_max - ext_min
    spc        = ranges[int(np.argmax(ranges))] / (grid_quality - 1)
    nx = int(round(ranges[0] / spc)) + 1
    ny = int(round(ranges[1] / spc)) + 1
    nz = int(round(ranges[2] / spc)) + 1
    origin  = ext_min
    spacing = np.array([spc, spc, spc])

    x = np.arange(nx) * spc + origin[0]
    y = np.arange(ny) * spc + origin[1]
    z = np.arange(nz) * spc + origin[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points  = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    def _eval_python(cmo):
        density = np.zeros(len(points))
        for basis, c in zip(final_norm_basis, cmo):
            if abs(c) <= 1e-15:
                continue
            atom_c = coord_bohr[basis["CENTER"] - 1][:, np.newaxis]
            dx, dy, dz = points.T - atom_c
            r   = np.sqrt(dx**2 + dy**2 + dz**2)
            ang = ang_res_lamda(dx, dy, dz, basis["orb_val"])
            for coeff, zeta in zip(basis["coeffs"], basis["exps"]):
                density += np.round(c * coeff * ang * np.exp(-zeta * r**2), 99)
        return density.reshape(nx, ny, nz)

    def _eval_cpp(cmo):
        psi = _cpp.electron_density(
            final_norm_basis, coord_bohr, points, cmo, None)
        return psi.reshape(nx, ny, nz)

    _eval = _eval_cpp if _use_cpp else _eval_python

    base    = os.path.splitext(os.path.basename(fchk_path))[0]
    results = []
    for cmo, idx in zip(cmos, orbital_indices):
        grid = _eval(cmo)
        results.append({
            "index":      idx,
            "label":      f"{base}-{idx}",
            "grid":       grid,
            "nx": nx, "ny": ny, "nz": nz,
            "spacing":    spacing.copy(),
            "origin":     origin.copy(),
            "atom_info":  atom_info,
            "bohr_const": bohr_const,
        })
    return results


if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else input("FCHK file path: ")
    print(f"\n{'='*70}")
    print(f"Testing FCHK orthonormality: {path}")
    print('='*70)

    final_basis, coordinates_ang, atom_info = load_basis_from_fchk(path)
    nbas = len(final_basis)
    print(f"  Loaded {nbas} normalized basis functions")
    print(f"  Loaded {len(atom_info)} atoms")

    orbital_indices = list(range(1, nbas + 1))
    cmos = load_cmos_from_fchk(path, orbital_indices, spin='alpha')
    print(f"  Loaded {len(cmos)} alpha CMOs")

    from overlap_matrix import get_overlap_matrix as getSmat
    from bas_dict import dict_keys

    overlap = getSmat(final_basis, dict_keys, normalize_primitives=False, diagonal_only=False)
    print(f"  Overlap matrix shape: {overlap.shape}")
    print(f"  Overlap diagonal (first 10): {np.diag(overlap)[:min(10, nbas)]}")
    
    ortho_flat = _parse_array(open(path, 'r').read().splitlines(), "Orthonormal basis", float)
    if ortho_flat is not None:
        if ortho_flat.size != nbas * nbas:
            raise ValueError(
                f"Orthonormal basis section size mismatch: expected {nbas*nbas}, got {ortho_flat.size}"
            )
        ortho_basis = ortho_flat.reshape(nbas, nbas)
        print(f"  Orthonormal basis shape: {ortho_basis.shape}")
        ortho_product = ortho_basis.T @ ortho_basis
        overlap_ortho = np.linalg.inv(ortho_product)
        print(f"  inv(ortho_basis.T @ ortho_basis) computed successfully")
        print(f"  Orthonormal product max deviation from identity: {np.max(np.abs(ortho_product - np.eye(nbas))):.8e}")
        print(f"  Orthonormal overlap diagonal (first 10): {np.diag(overlap_ortho)[:min(10, nbas)]}")
        diff = overlap_ortho - overlap
        print(f"  Overlap comparison max abs diff: {np.max(np.abs(diff)):.8e}")
        print(f"  Overlap comparison Frobenius norm diff: {np.linalg.norm(diff):.8e}")
    else:
        ortho_basis = None
        overlap_ortho = None
        print("  Orthonormal basis section not found in FCHK file")

    cmat = np.column_stack(cmos)

    print(overlap_ortho )
    print("\nn0=000000000000000000000000000000000000000000")
    print( overlap)
    if overlap_ortho is not None:
        print("  Using overlap_ortho for CMO orthonormality test")
        use_overlap = overlap_ortho
    else:
        print("  Using final_basis overlap for CMO orthonormality test")
        use_overlap = overlap

    ortho_test = cmat.T @ use_overlap @ cmat
    print(f"  CMO overlap test (should be close to identity):")
    print(ortho_test)
    print(np.diag(ortho_test))

     