import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import density_analysis
import fchk_read
import nbo_read
import overlap_matrix
import read_molden
from source_cache import ComputationCache, file_cache_key


class ComputationCacheTests(unittest.TestCase):
    def test_file_signature_invalidates_cached_value(self):
        cache = ComputationCache()
        calls = []

        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("first")
            path = handle.name

        try:
            def load():
                calls.append(None)
                return len(calls)

            key = ("value", file_cache_key(path))
            self.assertEqual(cache.get(key, load), 1)
            self.assertEqual(cache.get(key, load), 1)

            with open(path, "a") as handle:
                handle.write("-changed")

            changed_key = ("value", file_cache_key(path))
            self.assertNotEqual(key, changed_key)
            self.assertEqual(cache.get(changed_key, load), 2)
            self.assertEqual(len(calls), 2)
        finally:
            os.unlink(path)


class FchkCacheTests(unittest.TestCase):
    def setUp(self):
        fchk_read.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "sample.fchk")

    def tearDown(self):
        fchk_read.clear_source_cache()
        self.tempdir.cleanup()

    def test_basis_normalization_runs_once_and_invalidates_on_change(self):
        with open(self.path, "w") as handle:
            handle.write("basis-v1")

        normalized = [{"CENTER": 1, "orb_val": "s"}]
        with (
            mock.patch.object(fchk_read, "_extract_basis_set", return_value=[{}]) as extract,
            mock.patch.object(fchk_read, "_extract_atoms", return_value=([(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(fchk_read, "_normalise_basis", return_value=normalized) as normalize,
        ):
            first = fchk_read.load_basis_from_fchk(self.path)
            second = fchk_read.load_basis_from_fchk(self.path)
            self.assertIs(first, second)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(normalize.call_count, 1)

            with open(self.path, "a") as handle:
                handle.write("-changed")

            third = fchk_read.load_basis_from_fchk(self.path)
            self.assertIsNot(third, first)
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(normalize.call_count, 2)

    def test_cmo_matrix_is_parsed_once_for_multiple_subsets(self):
        with open(self.path, "w") as handle:
            handle.write(
                "Number of basis functions                 I              2\n"
                "Alpha MO coefficients                     R   N=          4\n"
                " 1.0 0.0 0.0 1.0\n"
            )

        with mock.patch.object(fchk_read, "_parse_array", wraps=fchk_read._parse_array) as parse:
            first = fchk_read.load_cmos_from_fchk(self.path, [1], "alpha")
            second = fchk_read.load_cmos_from_fchk(self.path, [2], "alpha")

        np.testing.assert_array_equal(first[0], [1.0, 0.0])
        np.testing.assert_array_equal(second[0], [0.0, 1.0])
        self.assertEqual(parse.call_count, 1)


    def test_occupation_derivation_reuses_cached_cmo_matrix(self):
        with open(self.path, "w") as handle:
            handle.write(
                "Number of basis functions                 I              2\n"
                "Alpha Orbital Energies                    R   N=          2\n"
                " -0.5 0.2\n"
                "Alpha MO coefficients                     R   N=          4\n"
                " 1.0 0.0 0.0 1.0\n"
                "Total SCF Density                         R   N=          3\n"
                " 1.0 0.0 1.0\n"
            )

        with (
            mock.patch.object(fchk_read, "_parse_array", wraps=fchk_read._parse_array) as parse,
            mock.patch.object(fchk_read, "get_ao_overlap_matrix", return_value=np.eye(2)),
        ):
            fchk_read.load_cmos_from_fchk(self.path, [1], "alpha")
            fchk_read.get_orbital_energies_and_occupations_fchk(self.path)

        mo_reads = [
            call for call in parse.call_args_list
            if len(call.args) > 1 and call.args[1] == "Alpha MO coefficients"
        ]
        self.assertEqual(len(mo_reads), 1)


    def test_final_overlap_is_computed_once(self):
        with open(self.path, "w") as handle:
            handle.write("overlap")

        expected = np.eye(1)
        with (
            mock.patch.object(fchk_read, "load_basis_from_fchk", return_value=([{}], [], [])),
            mock.patch.object(overlap_matrix, "get_overlap_matrix", return_value=expected) as get_overlap,
        ):
            first = fchk_read.get_ao_overlap_matrix(self.path)
            second = fchk_read.get_ao_overlap_matrix(self.path)

        self.assertIs(first, second)
        self.assertEqual(get_overlap.call_count, 1)


class MoldenCacheTests(unittest.TestCase):
    def setUp(self):
        read_molden.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "sample.molden")
        with open(self.path, "w") as handle:
            handle.write("molden-v1")

    def tearDown(self):
        read_molden.clear_source_cache()
        self.tempdir.cleanup()

    def test_full_parse_runs_once_and_invalidates_on_change(self):
        parsed = {"nbas": 1}
        with mock.patch.object(read_molden, "_parse_molden_uncached", return_value=parsed) as parse:
            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertEqual(parse.call_count, 1)

            with open(self.path, "a") as handle:
                handle.write("-changed")

            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertEqual(parse.call_count, 2)

    def test_basis_normalization_and_final_overlap_are_each_cached(self):
        parsed = {
            "raw_basis": [{}],
            "coordinates_ang": [(0, 0, 0)],
            "atom_info": [(1, 0, 0, 0)],
        }
        normalized = [{"CENTER": 1, "orb_val": "s"}]
        expected_overlap = np.eye(1)
        with (
            mock.patch.object(read_molden, "_parse_molden", return_value=parsed),
            mock.patch.object(read_molden, "_normalise_basis", return_value=normalized) as normalize,
            mock.patch.object(overlap_matrix, "get_overlap_matrix", return_value=expected_overlap) as get_overlap,
        ):
            first_basis = read_molden.load_basis_from_molden(self.path)
            second_basis = read_molden.load_basis_from_molden(self.path)
            first_overlap = read_molden.get_ao_overlap_matrix(self.path)
            second_overlap = read_molden.get_ao_overlap_matrix(self.path)

        self.assertIs(first_basis, second_basis)
        self.assertIs(first_overlap, second_overlap)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(get_overlap.call_count, 1)


class NboCacheTests(unittest.TestCase):
    def setUp(self):
        nbo_read.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_path = os.path.join(self.tempdir.name, "sample.40")
        with open(self.key_path, "w") as handle:
            handle.write("HEADER\nCMO\nCOMMENT\n1.0 0.0\n0.0 1.0\n")

    def tearDown(self):
        nbo_read.clear_source_cache()
        self.tempdir.cleanup()

    def test_full_cmo_matrix_is_loaded_once_for_multiple_subsets(self):
        with (
            mock.patch.object(nbo_read, "get_orbital_count", return_value=("CMO", 2, False)),
            mock.patch.object(nbo_read, "_detect_open_shell_key", return_value=False),
            mock.patch.object(nbo_read, "_single_block_open_shell_key", return_value=False),
            mock.patch.object(nbo_read, "_read_key_lines", wraps=nbo_read._read_key_lines) as read_lines,
        ):
            first = nbo_read.load_cmos_headless(self.key_path, [1], "alpha")
            second = nbo_read.load_cmos_headless(self.key_path, [2], "alpha")

        np.testing.assert_array_equal(first[0], [1.0, 0.0])
        np.testing.assert_array_equal(second[0], [0.0, 1.0])
        self.assertEqual(read_lines.call_count, 1)


class SpinDensityReuseTests(unittest.TestCase):
    def test_closed_shell_loads_only_alpha_and_cancels_spin_density(self):
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        alpha_cmo = np.ones((1, 1))
        spins_loaded = []
        captured = {}

        def localization_inputs(path, key_path=None, spin="alpha"):
            spins_loaded.append(spin)
            return alpha_cmo, np.eye(1), basis

        def spin_density(**kwargs):
            captured["denmat"] = kwargs["denmat"]
            return np.zeros(len(kwargs["points"]))

        with (
            mock.patch.object(read_molden, "load_basis_from_molden", return_value=(basis, [(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(density_analysis, "_get_occupation_arrays", return_value=("molden", np.array([2.0]), None)),
            mock.patch.object(density_analysis, "get_localization_inputs", side_effect=localization_inputs),
            mock.patch.object(density_analysis, "_build_uniform_grid", return_value=(np.zeros((1, 3)), (1, 1, 1), np.ones(3), np.zeros(3), np.zeros((1, 3)))),
            mock.patch.object(density_analysis, "spin_density_at_points", side_effect=spin_density),
        ):
            density_analysis.compute_spin_density_cube_data("sample.molden")

        self.assertEqual(spins_loaded, ["alpha"])
        np.testing.assert_array_equal(captured["denmat"], np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
