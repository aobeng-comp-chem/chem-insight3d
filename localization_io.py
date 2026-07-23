"""
localization_io.py
===================
Single entry point for orbital-localization code to pull what it needs out
of an NBO (.47/.31), Gaussian checkpoint (.fchk/.fck), or Molden (.molden)
source, without having to know which loader (nbo_read / fchk_read /
read_molden) or file-format quirks apply, and to actually run Pipek-Mezey
localization once it has them.

    get_localization_inputs(path, key_path=None, spin='alpha')
        -> (cmo_matrix, overlap_matrix, final_basis)

    get_fock_matrix(path, key_path=None, spin='alpha', cmo=None, overlap=None)
        -> fock_matrix

    localize_orbitals(cmo, overlap, fock, final_basis, space='occupied'|'virtual'|'range', ...)
        -> (localized_cmo, loc_energy)

    compute_localized_cube_data(path, spin='alpha', space='occupied', ...)
        -> dict of localized cube-grid data, ready for a viewer
"""

import os
import time

import numpy as np
import pandas as pd
from itertools import product

from overlap_matrix import get_overlap_matrix as _get_smat
from bas_dict import dict_keys


def _recognize_source_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".47", ".31"}:
        return "nbo"
    if ext in {".fchk", ".fck"}:
        return "fchk"
    if ext == ".molden":
        return "molden"
    return None


def get_localization_inputs(path, key_path=None, spin="alpha"):
    """
    Return (cmo_matrix, overlap_matrix, final_basis) for orbital localization.

    Parameters
    ----------
    path     : str
        NBO basis file (.47/.31), Gaussian checkpoint (.fchk/.fck), or
        Molden file (.molden).
    key_path : str, optional
        Required only for NBO sources — the paired key file (.31/.32/.33/...)
        holding the CMO coefficients. Ignored for fchk/molden, which store
        both the basis and the MOs in the same file.
    spin     : 'alpha' or 'beta'

    Returns
    -------
    cmo_matrix     : (nbas, norb) ndarray — columns are MOs, rows are AOs,
                      i.e. cmo_matrix[mu, i] is the coefficient of AO mu in
                      MO i.
    overlap_matrix : (nbas, nbas) ndarray — AO overlap matrix S.
    final_basis    : list of basis-function dicts, the shared schema used
                      by nbo_read / fchk_read / read_molden (N, CENTER,
                      type, orb_val, exps, coeffs, xcenter/ycenter/zcenter).
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        import nbo_read as _nr
        if key_path is None:
            raise ValueError(
                "NBO sources require key_path (the paired .31/.32/.33 "
                "orbital file holding the CMO coefficients)."
            )
        final_basis, _, _ = _nr.load_basis_headless(path)
        nbas = len(final_basis)
        cmo_rows = _nr.load_cmos_headless(key_path, list(range(1, nbas + 1)), spin=spin)

    elif source_type == "fchk":
        import fchk_read as _fr
        final_basis, _, _ = _fr.load_basis_from_fchk(path)
        nbas = len(final_basis)
        cmo_rows = _fr.load_cmos_from_fchk(path, list(range(1, nbas + 1)), spin=spin)

    elif source_type == "molden":
        import read_molden as _mr
        final_basis, _, _ = _mr.load_basis_from_molden(path)
        nbas = len(final_basis)
        cmo_rows = _mr.load_cmos_from_molden(path, list(range(1, nbas + 1)), spin=spin)

    else:
        raise ValueError(f"Unrecognized source file: {path}")

    # Loaders return one row per orbital (row i = MO i's AO coefficients);
    # localization code expects the standard AO x MO convention.
    cmo_matrix = np.asarray(cmo_rows, dtype=float).T

    overlap_matrix = _get_smat(
        final_basis, dict_keys, normalize_primitives=False, diagonal_only=False
    )

    return cmo_matrix, overlap_matrix, final_basis


def get_center_ranges(final_basis):
    """
    Group basis-function indices by their center (atom).

    Assumes final_basis is ordered so all basis functions of a given
    CENTER are contiguous (true for nbo_read/fchk_read/read_molden output).

    Parameters
    ----------
    final_basis : list of basis-function dicts (must have a 'CENTER' key).

    Returns
    -------
    list of {'bflo': int, 'bfhi': int} per center, in center order, where
    bflo/bfhi are inclusive 0-based indices into final_basis.
    """
    ranges = []
    start = 0
    for i in range(1, len(final_basis) + 1):
        if i == len(final_basis) or final_basis[i]["CENTER"] != final_basis[start]["CENTER"]:
            ranges.append({"bflo": start, "bfhi": i - 1})
            start = i
    return ranges


def get_fock_matrix(path, key_path=None, spin="alpha", cmo=None, overlap=None):
    """
    Return the AO-basis Fock matrix F for a source file.

    NBO .47 files store the Fock matrix directly (the $FOCK section) and it
    is simply read and returned as-is. fchk/molden sources don't store a
    Fock matrix, so it's rebuilt from the canonical orbital energies and MO
    coefficients:

        F = (S C) E (S C)^T = S C E C^T S

    where C is the AO x MO coefficient matrix, S is the AO overlap matrix,
    and E is the diagonal matrix of orbital energies. This holds because
    C^T S C = I (MO orthonormality in the AO metric) implies C^-1 = C^T S,
    so F_AO = S C F_MO C^-1 = S C E C^T S.

    Parameters
    ----------
    path, key_path, spin : same as get_localization_inputs.
    cmo, overlap : optional pre-computed cmo_matrix/overlap_matrix (e.g.
        from a prior get_localization_inputs call on the same source) to
        avoid re-parsing the file. Ignored for NBO sources, since the Fock
        matrix there is read directly rather than reconstructed.

    Returns
    -------
    fock_matrix : (nbas, nbas) ndarray, AO basis.
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        if os.path.splitext(path)[1].lower() != ".47":
            raise ValueError(
                "The Fock matrix is only stored in NBO .47 files (not .31) "
                "— point `path` at the .47 file."
            )
        import nbo_read as _nr
        final_basis, _, _ = _nr.load_basis_headless(path)
        nbas = len(final_basis)
        is_open, matrix_dict = _nr.process_47_file(path, nbas)
        key = ("FOCK_BETA" if spin.lower().startswith("b") else "FOCK_ALPHA") if is_open else "FOCK"
        fock = matrix_dict.get(key)
        if fock is None or not np.any(fock):
            raise ValueError(f"No $FOCK section found in {path}")
        return fock

    if source_type not in ("fchk", "molden"):
        raise ValueError(f"Unrecognized source file: {path}")

    if cmo is None or overlap is None:
        cmo, overlap, _ = get_localization_inputs(path, key_path=key_path, spin=spin)

    if source_type == "fchk":
        import fchk_read as _fr
        ene_alpha, _, ene_beta, _ = _fr.get_orbital_energies_and_occupations_fchk(path)
    else:
        import read_molden as _mr
        ene_alpha, _, ene_beta, _ = _mr.get_orbital_energies_and_occupations_molden(path)

    energies = ene_beta if (spin.lower().startswith("b") and ene_beta is not None) else ene_alpha
    if energies is None or len(energies) == 0:
        raise ValueError(f"No {spin} orbital energies found in {path}")

    E = np.diag(np.asarray(energies, dtype=float))
    return overlap @ cmo @ E @ cmo.T @ overlap



_CLOSED_SHELL_OCC_SCALE = {"nbo": 1.0, "fchk": 2.0, "molden": 1.0}


def _get_occupation_arrays(path, key_path=None):
    """
    Return (source_type, occ_alpha, occ_beta) occupation-number arrays for
    a source. occ_beta is None for closed-shell sources.
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        if key_path is None:
            raise ValueError(
                "NBO sources require key_path (the paired .31/.32/.33 "
                "orbital file holding the CMO coefficients)."
            )
        import nbo_read as _nr
        _, occ_alpha, _, occ_beta = _nr.get_orbital_energies_and_occupations(key_path)

    elif source_type == "fchk":
        import fchk_read as _fr
        _, occ_alpha, _, occ_beta = _fr.get_orbital_energies_and_occupations_fchk(path)

    elif source_type == "molden":
        import read_molden as _mr
        _, occ_alpha, _, occ_beta = _mr.get_orbital_energies_and_occupations_molden(path)

    else:
        raise ValueError(f"Unrecognized source file: {path}")

    if occ_alpha is None or len(occ_alpha) == 0:
        raise ValueError(f"No orbital occupation numbers found in {path}")

    occ_alpha = np.asarray(occ_alpha, dtype=float)
    occ_beta = (
        np.asarray(occ_beta, dtype=float)
        if (occ_beta is not None and len(occ_beta) > 0)
        else None
    )
    return source_type, occ_alpha, occ_beta


def get_electron_count(path, key_path=None):
    """
    Return the total number of electrons for a source file, derived from
    its orbital occupation numbers.

    Occupation-number scale differs by loader: fchk reports alpha/beta
    occupancies separately on a 0-1 (per-spin) scale, while molden/NBO
    closed-shell sources report a single alpha channel on a 0-2 (total)
    scale. Both are handled here.

    Parameters
    ----------
    path     : str, same as get_localization_inputs.
    key_path : str, optional — required for NBO sources (see
        get_localization_inputs); ignored for fchk/molden.
    """
    source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)

    if occ_beta is not None:
        return float(np.sum(occ_alpha) + np.sum(occ_beta))

    # Closed-shell, single alpha channel: the scale is a fixed property of
    # the loader (not the data) -- fchk reports per-spin (0-1) occupancies
    # with occ_beta=None, while molden/NBO closed-shell report a single
    # already-total (0-2) alpha channel.
    return float(np.sum(occ_alpha) * _CLOSED_SHELL_OCC_SCALE[source_type])


def get_num_occupied_orbitals(path, key_path=None, spin="alpha"):
    """
    Number of occupied (spatial) molecular orbitals.

    Closed-shell : n_occ = n_electrons / 2 (same for either spin channel,
        since alpha and beta orbitals are identical).
    Open-shell    : each spin-orbital holds exactly one electron, so n_occ
        for the requested spin is just that spin channel's own electron
        count (n_occ_alpha = n_alpha_electrons, n_occ_beta = n_beta_electrons)
        -- no halving.

    Raises if a closed-shell source reports an odd electron count, since
    n_occ = n_electrons/2 wouldn't be an integer.
    """
    source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)

    if occ_beta is None:
        # Closed-shell: same scale-normalization as get_electron_count.
        n_electrons = np.sum(occ_alpha) * _CLOSED_SHELL_OCC_SCALE[source_type]
        if round(n_electrons) % 2 != 0:
            raise ValueError(
                f"{path} has an odd electron count ({n_electrons:g}) for a "
                "closed-shell source -- this looks like an open-shell "
                "system reporting only one spin channel."
            )
        return int(round(n_electrons)) // 2

    occ = occ_beta if spin.lower().startswith("b") else occ_alpha
    return int(round(float(np.sum(occ))))


def _fock_diagonal(c, fock):
    """diag(c.T @ fock @ c) without forming the full n_sel x n_sel product."""
    return np.einsum('ij,ij->j', c, fock @ c)


import time

import numpy as np


def localize_orbitals(
    cmo,
    overlap,
    fock,
    basis,
    space="occupied",
    n_occ=None,
    orbital_range=None,
    seed=0,
):
    """
    Localize a subset of molecular orbitals with Pipek-Mezey localization
    using sequential Jacobi 2x2 rotations.

    This implementation vectorizes the atomic-population calculations.
    The orbital-pair loop remains sequential because each Jacobi rotation
    changes the orbitals used by subsequent rotations.

    Convergence is determined from the Pipek-Mezey objective

        L_PM = sum_i sum_A q_A(i)^2

    where

        q_A(i) = sum_{mu in A} C[mu, i] * (S C)[mu, i].

    Parameters
    ----------
    cmo : (nbas, norb) ndarray
        Full AO x MO coefficient matrix. Orbitals are assumed to be
        canonical and energy ordered.

    overlap : (nbas, nbas) ndarray
        AO overlap matrix S.

    fock : (nbas, nbas) ndarray
        AO Fock matrix F.

    basis : sequence of mappings
        Per-atom basis-function ranges. Each element must contain
        inclusive 'bflo' and 'bfhi' indices.

    space : {'occupied', 'virtual', 'range'}, default='occupied'
        Orbital subspace to localize.

    n_occ : int or None
        Number of occupied orbitals. Required for space='occupied'
        and space='virtual'.

    orbital_range : tuple[int, int] or None
        First and last MO numbers using 1-based inclusive indexing.
        Required for space='range'.

    seed : int or None, default=0
        Random seed controlling the orbital-pair sweep order.

    Returns
    -------
    sorted_c : (nbas, n_selected) ndarray
        Localized orbitals sorted by Fock expectation value.

    loc_energy : (n_selected,) ndarray
        Fock expectation values of the localized orbitals.
    """
    start_time = time.perf_counter()

    cmo = np.asarray(cmo)
    overlap = np.asarray(overlap)
    fock = np.asarray(fock)

    if cmo.ndim != 2:
        raise ValueError(
            f"cmo must be a two-dimensional array; got shape {cmo.shape}."
        )

    n_basis_fn, n_orbitals = cmo.shape

    if overlap.shape != (n_basis_fn, n_basis_fn):
        raise ValueError(
            f"overlap must have shape {(n_basis_fn, n_basis_fn)}; "
            f"got {overlap.shape}."
        )

    if fock.shape != (n_basis_fn, n_basis_fn):
        raise ValueError(
            f"fock must have shape {(n_basis_fn, n_basis_fn)}; "
            f"got {fock.shape}."
        )

    if len(basis) == 0:
        raise ValueError("basis must contain at least one atom.")

    # ---------------------------------------------------------------
    # Select the requested orbital subspace.
    # ---------------------------------------------------------------
    if space == "occupied":
        if n_occ is None:
            raise ValueError(
                "n_occ is required when space='occupied'."
            )

        lo, hi = 0, n_occ

    elif space == "virtual":
        if n_occ is None:
            raise ValueError(
                "n_occ is required when space='virtual'."
            )

        lo, hi = n_occ, n_orbitals

    elif space == "range":
        if orbital_range is None:
            raise ValueError(
                "orbital_range=(first, last) is required "
                "when space='range'."
            )

        first, last = orbital_range

        if not (1 <= first <= last <= n_orbitals):
            raise ValueError(
                f"orbital_range {orbital_range} "
                f"(1-based, inclusive) is out of bounds for "
                f"{n_orbitals} orbitals."
            )

        lo, hi = first - 1, last

    else:
        raise ValueError(
            f"Unknown orbital space: {space!r}; expected "
            "'occupied', 'virtual', or 'range'."
        )

    if not (0 <= lo < hi <= n_orbitals):
        raise ValueError(
            f"Orbital selection [{lo}, {hi}) is out of bounds "
            f"for {n_orbitals} orbitals."
        )

    n_sel = hi - lo
    natom = len(basis)

    # ---------------------------------------------------------------
    # Numerical controls.
    # ---------------------------------------------------------------
    gamma_tol = 1.0e-10
    coupling_tol = 1.0e-14

    objective_atol = 1.0e-12
    objective_rtol = 1.0e-10

    max_sweeps = 2000

    # ---------------------------------------------------------------
    # Prepare a grouped list of AO indices.
    #
    # atom_starts contains the starting position of each atom in the
    # grouped AO list. np.add.reduceat can therefore replace the loops
    # over atoms and basis functions.
    # ---------------------------------------------------------------
    atom_ranges = []
    atom_starts = np.empty(natom, dtype=np.intp)

    next_start = 0

    for atom_index, atom in enumerate(basis):
        bflo = int(atom["bflo"])
        bfhi = int(atom["bfhi"])

        if not (0 <= bflo <= bfhi < n_basis_fn):
            raise ValueError(
                f"Invalid basis-function range ({bflo}, {bfhi}) "
                f"for atom {atom_index}; valid basis indices are "
                f"0 through {n_basis_fn - 1}."
            )

        atom_starts[atom_index] = next_start

        indices = np.arange(
            bflo,
            bfhi + 1,
            dtype=np.intp,
        )

        atom_ranges.append(indices)
        next_start += indices.size

    grouped_ao_indices = np.concatenate(atom_ranges)

    # When the ranges already correspond to every AO in its normal order,
    # array indexing can be skipped entirely.
    ao_ranges_are_contiguous = (
        grouped_ao_indices.size == n_basis_fn
        and np.array_equal(
            grouped_ao_indices,
            np.arange(n_basis_fn),
        )
    )

    def reduce_by_atom(values):
        """
        Sum a one- or two-dimensional AO array over the basis functions
        belonging to each atom.
        """
        if values.shape[0] != n_basis_fn:
            raise ValueError(
                "The first dimension of values must equal the number "
                "of basis functions."
            )

        if ao_ranges_are_contiguous:
            grouped_values = values
        else:
            grouped_values = values[grouped_ao_indices, ...]

        return np.add.reduceat(
            grouped_values,
            atom_starts,
            axis=0,
        )

    def calculate_atomic_populations(coefficients, overlap_coefficients):
        """
        Return q[A, i] for every atom A and selected orbital i.
        """
        return reduce_by_atom(
            coefficients * overlap_coefficients
        )

    def pipek_mezey_objective(atomic_populations):
        """
        Calculate sum_A sum_i q[A, i]^2.
        """
        return float(
            np.einsum(
                "ai,ai->",
                atomic_populations,
                atomic_populations,
                optimize=True,
            )
        )

    # ---------------------------------------------------------------
    # Initialize the selected orbital space.
    # ---------------------------------------------------------------
    c = cmo[:, lo:hi].copy()
    sc = overlap @ c

    atomic_populations = calculate_atomic_populations(c, sc)

    prev_objective = pipek_mezey_objective(
        atomic_populations
    )

    current_objective = prev_objective
    objective_change = 0.0

    rng = np.random.default_rng(seed)

    # Preallocated work arrays avoid repeated allocations in the
    # orbital-pair loop.
    rotated_c_s = np.empty(n_basis_fn, dtype=c.dtype)
    rotated_c_t = np.empty(n_basis_fn, dtype=c.dtype)

    rotated_sc_s = np.empty(n_basis_fn, dtype=sc.dtype)
    rotated_sc_t = np.empty(n_basis_fn, dtype=sc.dtype)

    cross_ao = np.empty(
        n_basis_fn,
        dtype=np.result_type(c.dtype, sc.dtype),
    )

    cross_ao_tmp = np.empty_like(cross_ao)

    total_rotations = 0
    converged = False
    final_sweep = 0

    # ---------------------------------------------------------------
    # Sequential Jacobi sweeps.
    # ---------------------------------------------------------------
    for sweep in range(1, max_sweeps + 1):
        final_sweep = sweep
        rotations_this_sweep = 0

        random_s = rng.permutation(n_sel)

        # This preserves the pair ordering used by the original code:
        # s is randomized and t is visited in normal index order.
        #
        # The pairs are generated lazily rather than building a large
        # Python list.
        for s in random_s:
            for t in range(n_sel):
                if t == s:
                    continue

                # Atomic populations q_A(s) and q_A(t) are maintained
                # throughout the sweep, so they do not need to be
                # recalculated from the AO coefficients.
                qas = atomic_populations[:, s].copy()
                qat = atomic_populations[:, t].copy()

                # Compute the symmetrized AO cross-population:
                #
                # 0.5 * [
                #     C[:, t] * SC[:, s]
                #     + C[:, s] * SC[:, t]
                # ]
                np.multiply(
                    c[:, t],
                    sc[:, s],
                    out=cross_ao,
                )

                np.multiply(
                    c[:, s],
                    sc[:, t],
                    out=cross_ao_tmp,
                )

                cross_ao += cross_ao_tmp
                cross_ao *= 0.5

                # Vectorized reduction over atoms.
                qast = reduce_by_atom(cross_ao)

                population_difference = qas - qat

                ast = (
                    np.dot(qast, qast)
                    - 0.25
                    * np.dot(
                        population_difference,
                        population_difference,
                    )
                )

                bst = np.dot(
                    qast,
                    population_difference,
                )

                denominator = np.hypot(ast, bst)

                if denominator < coupling_tol:
                    continue

                cos_arg = np.clip(
                    -ast / denominator,
                    -1.0,
                    1.0,
                )

                gamma = (
                    0.25
                    * np.arccos(cos_arg)
                    * np.sign(bst)
                )

                if abs(gamma) <= gamma_tol:
                    continue

                cosg = np.cos(gamma)
                sing = np.sin(gamma)

                # Save these values because they are also needed for the
                # atomic-population update.
                cosg2 = cosg * cosg
                sing2 = sing * sing
                two_cos_sin = 2.0 * cosg * sing

                # ---------------------------------------------------
                # Rotate C.
                # ---------------------------------------------------
                np.multiply(
                    c[:, s],
                    cosg,
                    out=rotated_c_s,
                )

                rotated_c_s += c[:, t] * sing

                np.multiply(
                    c[:, t],
                    cosg,
                    out=rotated_c_t,
                )

                rotated_c_t -= c[:, s] * sing

                c[:, s] = rotated_c_s
                c[:, t] = rotated_c_t

                # ---------------------------------------------------
                # Rotate S C using the same Jacobi rotation.
                # ---------------------------------------------------
                np.multiply(
                    sc[:, s],
                    cosg,
                    out=rotated_sc_s,
                )

                rotated_sc_s += sc[:, t] * sing

                np.multiply(
                    sc[:, t],
                    cosg,
                    out=rotated_sc_t,
                )

                rotated_sc_t -= sc[:, s] * sing

                sc[:, s] = rotated_sc_s
                sc[:, t] = rotated_sc_t

                # ---------------------------------------------------
                # Update only the two affected population columns.
                #
                # q_s' = cos²(gamma) q_s
                #      + sin²(gamma) q_t
                #      + 2 cos(gamma) sin(gamma) q_st
                #
                # q_t' = sin²(gamma) q_s
                #      + cos²(gamma) q_t
                #      - 2 cos(gamma) sin(gamma) q_st
                # ---------------------------------------------------
                atomic_populations[:, s] = (
                    cosg2 * qas
                    + sing2 * qat
                    + two_cos_sin * qast
                )

                atomic_populations[:, t] = (
                    sing2 * qas
                    + cosg2 * qat
                    - two_cos_sin * qast
                )

                rotations_this_sweep += 1
                total_rotations += 1

        # Recompute the atomic populations from C and S C after each
        # complete sweep. This removes accumulated floating-point drift
        # from the incremental population updates.
        atomic_populations = calculate_atomic_populations(
            c,
            sc,
        )

        current_objective = pipek_mezey_objective(
            atomic_populations
        )

        objective_change = (
            current_objective - prev_objective
        )

        convergence_threshold = (
            objective_atol
            + objective_rtol
            * max(
                abs(prev_objective),
                abs(current_objective),
            )
        )

        if abs(objective_change) <= convergence_threshold:
            converged = True
            break

        # An entire sweep without a rotation is also converged.
        if rotations_this_sweep == 0:
            converged = True
            break

        prev_objective = current_objective

    # ---------------------------------------------------------------
    # Sort localized orbitals by Fock expectation value.
    # ---------------------------------------------------------------
    unsorted_energy = _fock_diagonal(c, fock)
    sort_energy_indices = np.argsort(unsorted_energy)

    sorted_c = c[:, sort_energy_indices]
    loc_energy = unsorted_energy[sort_energy_indices]

    elapsed_time = time.perf_counter() - start_time

    if converged:
        print(
            f"Localization converged after {final_sweep} sweeps."
        )
    else:
        print(
            f"Localization not converged after "
            f"{max_sweeps} sweeps."
        )

    print(
        f"Pipek-Mezey objective: "
        f"{current_objective:.15g}"
    )

    print(
        f"Objective change: {objective_change:.3e}"
    )

    print(
        f"Jacobi rotations: {total_rotations}"
    )

    print(
        f"Localization runtime: {elapsed_time:.6f} seconds"
    )

    print(
        f"Energy of localized orbitals:\n"
        f"{loc_energy}\n"
    )

    return sorted_c, loc_energy


def compute_localized_cube_data(
    path,
    spin="alpha",
    space="occupied",
    n_occ=None,
    orbital_range=None,
    seed=0,
    grid_quality=75,
    ext_dist=4.0,
    bohr_const=0.529177249,
):
    """
    Localize a subspace of MOs and compute cube grids for them, ready for
    a viewer to render -- the single entry point tying together
    get_localization_inputs / get_fock_matrix / localize_orbitals and the
    per-format compute_cube_data* functions.

    NBO sources always localize the sibling `.40` key file (the canonical
    AO-basis MOs), regardless of whichever key file is used elsewhere for
    picking un-localized orbitals; `path` must be the `.47` basis file
    (same requirement get_fock_matrix already has for the Fock matrix).
    fchk/molden sources localize whatever CMOs are in the source file
    itself.

    Parameters
    ----------
    path, spin, space, n_occ, orbital_range, seed : see localize_orbitals /
        get_localization_inputs.
    grid_quality, ext_dist, bohr_const : see compute_cube_data /
        compute_cube_data_fchk / compute_cube_data_molden.

    Returns
    -------
    dict with keys:
        cubes           : list of per-orbital cube-grid dicts (same shape
                           compute_cube_data* returns), labeled
                           "<base>_LOC_<space>-<n>" (no '.' before LOC --
                           os.path.splitext() is used elsewhere to recover
                           <base>, and a '.' there would be misparsed as a
                           file extension).
        final_basis     : basis-function list.
        atom_info       : list of (Z, x_ang, y_ang, z_ang).
        localized_cmo   : (nbas, n_selected) ndarray, AO x MO.
        energies        : (n_selected,) ndarray, Fock expectation values.
        occupations     : (n_selected,) ndarray, or None for space='range'
                           (ambiguous when the range straddles n_occ).
        overlap, fock   : AO-basis matrices used for the localization.
        orbital_indices : list(range(1, n_selected + 1)).
        n_occ           : the n_occ actually used (resolved if it was None).
        space           : echoed back, for convenience.
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        key_path = os.path.splitext(path)[0] + ".40"
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"NBO orbital localization requires a sibling .40 key file "
                f"(the canonical AO-basis MOs) -- expected {key_path}."
            )
        import nbo_read as _nr
        final_basis, coordinates_ang, atom_info = _nr.load_basis_headless(path)

    elif source_type == "fchk":
        key_path = None
        import fchk_read as _fr
        final_basis, coordinates_ang, atom_info = _fr.load_basis_from_fchk(path)

    elif source_type == "molden":
        key_path = None
        import read_molden as _mr
        final_basis, coordinates_ang, atom_info = _mr.load_basis_from_molden(path)

    else:
        raise ValueError(f"Unrecognized source file: {path}")

    cmo, overlap, final_basis = get_localization_inputs(path, key_path=key_path, spin=spin)
    fock = get_fock_matrix(path, key_path=key_path, spin=spin, cmo=cmo, overlap=overlap)
    basis_center = get_center_ranges(final_basis)

    _, _occ_alpha_probe, _occ_beta_probe = _get_occupation_arrays(path, key_path=key_path)
    is_open_shell = _occ_beta_probe is not None

    if space in ("occupied", "virtual") and n_occ is None:
        n_occ = get_num_occupied_orbitals(path, key_path=key_path, spin=spin)

    localized_cmo, loc_energy = localize_orbitals(
        cmo, overlap, fock, basis_center,
        space=space, n_occ=n_occ, orbital_range=orbital_range, seed=seed,
    )

    n_sel = localized_cmo.shape[1]
    orbital_indices = list(range(1, n_sel + 1))
    cmos_rows = list(localized_cmo.T)

    if space == "occupied":
        # Closed-shell orbitals hold 2 electrons each; open-shell spin-
        # orbitals (each spin solved independently) hold exactly 1.
        occupations = np.full(n_sel, 1.0 if is_open_shell else 2.0)
    elif space == "virtual":
        occupations = np.full(n_sel, 0.0)
    else:
        # A 'range' selection may straddle the occ/virt boundary, where
        # per-orbital occupation isn't well-defined here.
        occupations = None

    if source_type == "nbo":
        cubes = _nr.compute_cube_data(
            final_basis, coordinates_ang, atom_info,
            orbital_indices, path, spin,
            grid_quality, ext_dist, bohr_const,
            precomputed_cmos=cmos_rows,
        )
    elif source_type == "fchk":
        cubes = _fr.compute_cube_data_fchk(
            path, orbital_indices, spin, grid_quality, ext_dist, bohr_const,
            precomputed_cmos=cmos_rows,
        )
    else:
        cubes = _mr.compute_cube_data_molden(
            path, orbital_indices, spin, grid_quality, ext_dist, bohr_const,
            precomputed_cmos=cmos_rows,
        )

    base = os.path.splitext(os.path.basename(path))[0]
    for i, cube in enumerate(cubes, start=1):
        cube["label"] = f"{base}_LOC_{space}-{i}"

    return {
        "cubes": cubes,
        "final_basis": final_basis,
        "atom_info": atom_info,
        "localized_cmo": localized_cmo,
        "energies": loc_energy,
        "occupations": occupations,
        "overlap": overlap,
        "fock": fock,
        "orbital_indices": orbital_indices,
        "n_occ": n_occ,
        "space": space,
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity-check get_localization_inputs() against an "
                    "NBO (.47/.31), Gaussian (.fchk/.fck), or Molden (.molden) source."
    )
    parser.add_argument("path", help="Basis/orbital source file")
    parser.add_argument(
        "--key", dest="key_path", default=None,
        help="NBO key file (.31/.32/.33/.40/...) holding the CMO coefficients "
             "— required when path is a .47/.31 NBO basis file, ignored otherwise.",
    )
    parser.add_argument("--spin", default="alpha", choices=["alpha", "beta"])
    parser.add_argument(
        "--space", default="occupied", choices=["occupied", "virtual", "range"],
        help="Orbital subspace to localize (default: occupied).",
    )
    parser.add_argument(
        "--range", dest="orbital_range", default=None,
        help="Inclusive 1-based 'first-last' MO numbers, required when --space=range "
             "(e.g. '49-60').",
    )
    args = parser.parse_args()

    cmo, S, basis = get_localization_inputs(args.path, key_path=args.key_path, spin=args.spin)

    print(f"\n{'=' * 70}")
    print(f"Source type     : {_recognize_source_type(args.path)}")
    print(f"Source file     : {args.path}")
    if args.key_path:
        print(f"Key file        : {args.key_path}")
    print(f"Spin            : {args.spin}")
    print("=" * 70)
    print(f"Basis functions : {len(basis)}")
    print(f"CMO matrix      : {cmo.shape}  (AO x MO)")
    print(f"Overlap matrix  : {S.shape}")

    # Canonical MOs should be orthonormal in the AO metric: C^T S C == I.
    ortho     = cmo.T @ S @ cmo
    diag      = np.diag(ortho)
    off_diag  = ortho - np.diag(diag)
    print(f"\nC^T S C diagonal (first 10)   : {diag[:10].round(6)}")
    print(f"Max |diagonal - 1|            : {np.max(np.abs(diag - 1)):.3e}")
    print(f"Max |off-diagonal|            : {np.max(np.abs(off_diag)):.3e}")

    F = get_fock_matrix(args.path, key_path=args.key_path, spin=args.spin, cmo=cmo, overlap=S)
    print(f"\nFock matrix     : {F.shape}")
    # Back-transform to the MO basis: C^T F C should be ~diagonal, with the
    # diagonal equal to the canonical orbital energies.
    fock_mo  = cmo.T @ F @ cmo
    fock_off = fock_mo - np.diag(np.diag(fock_mo))
    print(f"C^T F C diagonal (first 10)   : {np.diag(fock_mo)[:10].round(6)}")
    print(f"Max |off-diagonal| of C^T F C : {np.max(np.abs(fock_off)):.3e}")

    basis_center = get_center_ranges(basis)
    n_occ = get_num_occupied_orbitals(args.path, key_path=args.key_path, spin=args.spin)
    print(f"\nOccupied orbitals for {args.spin} : {n_occ}")

    orbital_range = None
    if args.orbital_range:
        first, last = (int(x) for x in args.orbital_range.split("-"))
        orbital_range = (first, last)

    localize_orbitals(
        cmo, S, F, basis_center,
        space=args.space, n_occ=n_occ, orbital_range=orbital_range,
    )  # returns (sorted_c, loc_energy); loc_energy is already printed above
