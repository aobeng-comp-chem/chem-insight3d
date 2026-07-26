#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct AtomRange {
    int bflo;
    int bfhi;
};

inline py::array_t<double> make_2d(const std::vector<double>& data, std::size_t rows, std::size_t cols) {
    py::array_t<double> out({rows, cols});
    auto r = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            r(i, j) = data[i * cols + j];
        }
    }
    return out;
}

inline py::array_t<double> make_1d(const std::vector<double>& data) {
    py::array_t<double> out({data.size()});
    auto r = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < data.size(); ++i) {
        r(i) = data[i];
    }
    return out;
}

std::pair<int, int> select_orbital_range(int n_orbitals, const std::string& space, int n_occ, const py::tuple& orbital_range) {
    if (space == "occupied") {
        if (n_occ <= 0) throw std::invalid_argument("n_occ must be positive for occupied space");
        return {0, n_occ};
    }
    if (space == "virtual") {
        if (n_occ <= 0) throw std::invalid_argument("n_occ must be positive for virtual space");
        return {n_occ, n_orbitals};
    }
    if (space == "range") {
        if (orbital_range.size() != 2) throw std::invalid_argument("orbital_range must be a 2-tuple");
        int first = orbital_range[0].cast<int>();
        int last = orbital_range[1].cast<int>();
        if (!(1 <= first && first <= last && last <= n_orbitals)) {
            throw std::invalid_argument("orbital_range is out of bounds");
        }
        return {first - 1, last};
    }
    throw std::invalid_argument("Unknown orbital space");
}

void validate_inputs(const py::array_t<double>& cmo_arr, const py::array_t<double>& overlap_arr, const py::array_t<double>& fock_arr) {
    if (cmo_arr.ndim() != 2) throw std::invalid_argument("cmo must be a 2D array");
    if (overlap_arr.ndim() != 2) throw std::invalid_argument("overlap must be a 2D array");
    if (fock_arr.ndim() != 2) throw std::invalid_argument("fock must be a 2D array");

    auto cmo = cmo_arr.unchecked<2>();
    auto overlap = overlap_arr.unchecked<2>();
    auto fock = fock_arr.unchecked<2>();

    if (overlap.shape(0) != cmo.shape(0) || overlap.shape(1) != cmo.shape(0)) {
        throw std::invalid_argument("overlap must be square with size equal to number of basis functions");
    }
    if (fock.shape(0) != cmo.shape(0) || fock.shape(1) != cmo.shape(0)) {
        throw std::invalid_argument("fock must be square with size equal to number of basis functions");
    }
}

std::vector<double> reduce_by_atom(const std::vector<double>& values, const std::vector<std::vector<int>>& atom_ranges, int n_basis_fn) {
    std::vector<double> reduced(atom_ranges.size(), 0.0);
    for (std::size_t atom_idx = 0; atom_idx < atom_ranges.size(); ++atom_idx) {
        double sum = 0.0;
        for (int idx : atom_ranges[atom_idx]) {
            sum += values[idx];
        }
        reduced[atom_idx] = sum;
    }
    return reduced;
}

// sum_A sum_i q[A,i]^2 -- the Pipek-Mezey objective, same convergence
// criterion the Python implementation uses (localization_io.py's
// pipek_mezey_objective). atomic_populations is natom*n_sel, flat.
inline double pipek_mezey_objective(const std::vector<double>& atomic_populations) {
    double total = 0.0;
    for (double v : atomic_populations) total += v * v;
    return total;
}

// np.sign() semantics: exactly 0.0 at x == 0, not +1/-1. bst == 0 means no
// net population imbalance for this pair, so the rotation must be a no-op.
inline double sign_of(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

}  // namespace

py::tuple localize_orbitals_cpp(
    py::array_t<double> cmo_arr,
    py::array_t<double> overlap_arr,
    py::array_t<double> fock_arr,
    py::list basis_list,
    const std::string& space = "occupied",
    int n_occ = 0,
    py::tuple orbital_range = py::tuple(),
    int seed = 0,
    py::array_t<int> sweep_orders = py::array_t<int>()
) {
    validate_inputs(cmo_arr, overlap_arr, fock_arr);

    auto cmo = cmo_arr.unchecked<2>();
    auto overlap = overlap_arr.unchecked<2>();
    auto fock = fock_arr.unchecked<2>();

    const int n_basis_fn = cmo.shape(0);
    const int n_orbitals = cmo.shape(1);

    std::vector<AtomRange> basis;
    basis.reserve(basis_list.size());
    for (py::handle item : basis_list) {
        py::dict atom = item.cast<py::dict>();
        basis.push_back({atom["bflo"].cast<int>(), atom["bfhi"].cast<int>()});
    }

    auto [lo, hi] = select_orbital_range(n_orbitals, space, n_occ, orbital_range);
    if (!(0 <= lo && lo < hi && hi <= n_orbitals)) {
        throw std::invalid_argument("Orbital selection is out of bounds");
    }

    const int n_sel = hi - lo;
    const int natom = static_cast<int>(basis.size());

    std::vector<std::vector<int>> atom_ranges;
    atom_ranges.reserve(natom);
    for (int atom_index = 0; atom_index < natom; ++atom_index) {
        const auto& atom = basis[atom_index];
        const int bflo = atom.bflo;
        const int bfhi = atom.bfhi;
        if (!(0 <= bflo && bflo <= bfhi && bfhi < n_basis_fn)) {
            throw std::invalid_argument("Invalid basis-function range");
        }
        std::vector<int> indices;
        indices.reserve(bfhi - bflo + 1);
        for (int idx = bflo; idx <= bfhi; ++idx) {
            indices.push_back(idx);
        }
        atom_ranges.push_back(std::move(indices));
    }

    std::vector<double> c(n_basis_fn * n_sel);
    std::vector<double> sc(n_basis_fn * n_sel);
    for (int i = 0; i < n_basis_fn; ++i) {
        for (int j = 0; j < n_sel; ++j) {
            c[i * n_sel + j] = cmo(i, lo + j);
        }
    }

    for (int i = 0; i < n_basis_fn; ++i) {
        for (int j = 0; j < n_sel; ++j) {
            double accum = 0.0;
            for (int k = 0; k < n_basis_fn; ++k) {
                accum += overlap(i, k) * cmo(k, lo + j);
            }
            sc[i * n_sel + j] = accum;
        }
    }

    std::vector<double> atomic_populations(natom * n_sel, 0.0);
    for (int s = 0; s < n_sel; ++s) {
        for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
            double sum = 0.0;
            for (int idx : atom_ranges[atom_idx]) {
                sum += c[idx * n_sel + s] * sc[idx * n_sel + s];
            }
            atomic_populations[atom_idx * n_sel + s] = sum;
        }
    }

    const double gamma_tol = 1.0e-10;
    const double coupling_tol = 1.0e-14;
    // Same convergence criteria as localization_io.localize_orbitals: a
    // combined absolute+relative tolerance on the Pipek-Mezey objective's
    // sweep-to-sweep change, or a sweep with zero rotations. max_sweeps is
    // a ceiling, not a target -- with real convergence checking it usually
    // stops far earlier, same as the Python version.
    const double objective_atol = 1.0e-12;
    const double objective_rtol = 1.0e-10;
    const int max_sweeps = 2000;

    std::vector<std::vector<int>> sweep_orders_by_step;
    const bool have_sweep_orders = sweep_orders.ndim() == 2 && sweep_orders.shape(0) > 0 && sweep_orders.shape(1) == n_sel;
    if (have_sweep_orders) {
        auto orders = sweep_orders.unchecked<2>();
        sweep_orders_by_step.reserve(orders.shape(0));
        for (std::size_t sweep_idx = 0; sweep_idx < static_cast<std::size_t>(orders.shape(0)); ++sweep_idx) {
            std::vector<int> order(n_sel);
            for (int i = 0; i < n_sel; ++i) {
                order[i] = orders(sweep_idx, i);
            }
            sweep_orders_by_step.push_back(std::move(order));
        }
    }

    std::vector<int> order(n_sel);
    for (int i = 0; i < n_sel; ++i) order[i] = i;

    std::vector<double> cross_ao(n_basis_fn);
    std::vector<double> cross_ao_tmp(n_basis_fn);
    std::vector<double> rotated_c_s(n_basis_fn);
    std::vector<double> rotated_c_t(n_basis_fn);
    std::vector<double> rotated_sc_s(n_basis_fn);
    std::vector<double> rotated_sc_t(n_basis_fn);

    double prev_objective = pipek_mezey_objective(atomic_populations);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        if (have_sweep_orders && static_cast<std::size_t>(sweep) < sweep_orders_by_step.size()) {
            order = sweep_orders_by_step[sweep];
        } else {
            std::mt19937 rng(static_cast<unsigned>(seed + sweep));
            std::shuffle(order.begin(), order.end(), rng);
        }
        int rotations_this_sweep = 0;
        for (int s : order) {
            for (int t = 0; t < n_sel; ++t) {
                if (t == s) continue;

                std::vector<double> qas(natom);
                std::vector<double> qat(natom);
                for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
                    qas[atom_idx] = atomic_populations[atom_idx * n_sel + s];
                    qat[atom_idx] = atomic_populations[atom_idx * n_sel + t];
                }

                for (int mu = 0; mu < n_basis_fn; ++mu) {
                    cross_ao[mu] = 0.5 * (c[mu * n_sel + t] * sc[mu * n_sel + s] + c[mu * n_sel + s] * sc[mu * n_sel + t]);
                }

                std::vector<double> qast(natom);
                for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
                    double sum = 0.0;
                    for (int idx : atom_ranges[atom_idx]) {
                        sum += cross_ao[idx];
                    }
                    qast[atom_idx] = sum;
                }

                double ast = 0.0;
                double bst = 0.0;
                for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
                    const double diff = qas[atom_idx] - qat[atom_idx];
                    ast += qast[atom_idx] * qast[atom_idx] - 0.25 * diff * diff;
                    bst += qast[atom_idx] * diff;
                }

                const double denominator = std::hypot(ast, bst);
                if (denominator < coupling_tol) continue;

                const double cos_arg = std::max(-1.0, std::min(1.0, -ast / denominator));
                const double gamma = 0.25 * std::acos(cos_arg) * sign_of(bst);
                if (std::abs(gamma) <= gamma_tol) continue;

                const double cosg = std::cos(gamma);
                const double sing = std::sin(gamma);
                const double cosg2 = cosg * cosg;
                const double sing2 = sing * sing;
                const double two_cos_sin = 2.0 * cosg * sing;

                for (int mu = 0; mu < n_basis_fn; ++mu) {
                    rotated_c_s[mu] = cosg * c[mu * n_sel + s] + sing * c[mu * n_sel + t];
                    rotated_c_t[mu] = cosg * c[mu * n_sel + t] - sing * c[mu * n_sel + s];
                    rotated_sc_s[mu] = cosg * sc[mu * n_sel + s] + sing * sc[mu * n_sel + t];
                    rotated_sc_t[mu] = cosg * sc[mu * n_sel + t] - sing * sc[mu * n_sel + s];
                }

                for (int mu = 0; mu < n_basis_fn; ++mu) {
                    c[mu * n_sel + s] = rotated_c_s[mu];
                    c[mu * n_sel + t] = rotated_c_t[mu];
                    sc[mu * n_sel + s] = rotated_sc_s[mu];
                    sc[mu * n_sel + t] = rotated_sc_t[mu];
                }

                for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
                    const double q_s_new = cosg2 * qas[atom_idx] + sing2 * qat[atom_idx] + two_cos_sin * qast[atom_idx];
                    const double q_t_new = sing2 * qas[atom_idx] + cosg2 * qat[atom_idx] - two_cos_sin * qast[atom_idx];
                    atomic_populations[atom_idx * n_sel + s] = q_s_new;
                    atomic_populations[atom_idx * n_sel + t] = q_t_new;
                }

                ++rotations_this_sweep;
            }
        }

        // Recompute atomic_populations from c/sc once per sweep, same as
        // the Python version: removes floating-point drift accumulated by
        // many incremental analytic updates in a row.
        for (int s = 0; s < n_sel; ++s) {
            for (int atom_idx = 0; atom_idx < natom; ++atom_idx) {
                double sum = 0.0;
                for (int idx : atom_ranges[atom_idx]) {
                    sum += c[idx * n_sel + s] * sc[idx * n_sel + s];
                }
                atomic_populations[atom_idx * n_sel + s] = sum;
            }
        }

        const double current_objective = pipek_mezey_objective(atomic_populations);
        const double objective_change = current_objective - prev_objective;
        const double convergence_threshold =
            objective_atol + objective_rtol * std::max(std::abs(prev_objective), std::abs(current_objective));

        if (std::abs(objective_change) <= convergence_threshold) break;
        if (rotations_this_sweep == 0) break;

        prev_objective = current_objective;
    }

    std::vector<double> energies(n_sel, 0.0);
    for (int i = 0; i < n_sel; ++i) {
        double energy = 0.0;
        for (int mu = 0; mu < n_basis_fn; ++mu) {
            for (int nu = 0; nu < n_basis_fn; ++nu) {
                energy += c[mu * n_sel + i] * fock(mu, nu) * c[nu * n_sel + i];
            }
        }
        energies[i] = energy;
    }

    // Sort by Fock expectation value, same as the Python version's
    // sorted_c/loc_energy contract.
    std::vector<int> sort_idx(n_sel);
    for (int i = 0; i < n_sel; ++i) sort_idx[i] = i;
    std::sort(sort_idx.begin(), sort_idx.end(),
              [&](int a, int b) { return energies[a] < energies[b]; });

    std::vector<double> sorted_c(static_cast<std::size_t>(n_basis_fn) * n_sel);
    std::vector<double> sorted_energies(n_sel);
    for (int i = 0; i < n_basis_fn; ++i) {
        for (int j = 0; j < n_sel; ++j) {
            sorted_c[i * n_sel + j] = c[i * n_sel + sort_idx[j]];
        }
    }
    for (int j = 0; j < n_sel; ++j) sorted_energies[j] = energies[sort_idx[j]];

    return py::make_tuple(make_2d(sorted_c, n_basis_fn, n_sel), make_1d(sorted_energies));
}

PYBIND11_MODULE(localization_native, m) {
    m.def("localize_orbitals_cpp", &localize_orbitals_cpp, py::arg("cmo"), py::arg("overlap"), py::arg("fock"), py::arg("basis"), py::arg("space") = "occupied", py::arg("n_occ") = 0, py::arg("orbital_range") = py::tuple(), py::arg("seed") = 0, py::arg("sweep_orders") = py::array_t<int>());
}
