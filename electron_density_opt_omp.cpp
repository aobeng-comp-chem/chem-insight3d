#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;



inline double powi(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i) result *= x;
    return result;
}

inline double solid_harmonic(double dx, double dy, double dz, const std::string& orb_val) {
    // s and p
    if (orb_val == "s")  return 1.0;
    if (orb_val == "px") return dx;
    if (orb_val == "py") return dy;
    if (orb_val == "pz") return dz;

    // d
    if (orb_val == "d0")   return dz*dz - 0.5*dx*dx - 0.5*dy*dy;
    if (orb_val == "dc1")  return 1.73205080756888 * dy * dz;
    if (orb_val == "ds1")  return 1.73205080756888 * dx * dz;
    if (orb_val == "dc2")  return 0.866025403784439 * (dx*dx - dy*dy);
    if (orb_val == "ds2")  return 1.73205080756888 * dx * dy;

    // f
    if (orb_val == "f0")   return dz*dz*dz - 1.5*dx*dx*dz - 1.5*dy*dy*dz;
    if (orb_val == "fc1")  return -0.612372435695794 * dx*dx*dx + 2.44948974278318 * dx*dz*dz - 0.612372435695794 * dx*dy*dy;
    if (orb_val == "fs1")  return -0.612372435695794 * dy*dy*dy + 2.44948974278318 * dy*dz*dz - 0.612372435695794 * dx*dx*dy;
    if (orb_val == "fc2")  return 1.93649167310371 * dx*dx*dz - 1.93649167310371 * dy*dy*dz;
    if (orb_val == "fs2")  return 3.87298334620742 * dx * dy * dz;
    if (orb_val == "fc3")  return 0.790569415042095 * dx*dx*dx - 2.37170824512628 * dx*dy*dy;
    if (orb_val == "fs3")  return -0.790569415042095 * dy*dy*dy + 2.37170824512628 * dx*dx*dy;

    // g
    if (orb_val == "g0")   return powi(dz,4) + 0.375*powi(dx,4) + 0.375*powi(dy,4) + 0.75*dx*dx*dy*dy - 3.0*dx*dx*dz*dz - 3.0*dy*dy*dz*dz;
    if (orb_val == "gc1")  return 3.16227766016838 * dx * powi(dz,3) - 2.37170824512628 * powi(dx,3) * dz - 2.37170824512628 * dx * dy*dy * dz;
    if (orb_val == "gs1")  return 3.16227766016838 * dy * powi(dz,3) - 2.37170824512628 * powi(dy,3) * dz - 2.37170824512628 * dx*dx * dy * dz;
    if (orb_val == "gc2")  return 0.559016994374947 * powi(dy,4) - 0.559016994374947 * powi(dx,4) + 3.35410196624968 * dx*dx * dz*dz - 3.35410196624968 * dy*dy * dz*dz;
    if (orb_val == "gs2")  return -1.11803398874989 * dx * powi(dy,3) - 1.11803398874989 * powi(dx,3) * dy + 6.70820393249937 * dx * dy * dz*dz;
    if (orb_val == "gc3")  return 2.09165006633519 * powi(dx,3) * dz - 6.27495019900557 * dx * dy*dy * dz;
    if (orb_val == "gs3")  return -2.09165006633519 * powi(dy,3) * dz + 6.27495019900557 * dx*dx * dy * dz;
    if (orb_val == "gc4")  return 0.739509972887452 * powi(dx,4) + 0.739509972887452 * powi(dy,4) - 4.43705983732471 * dx*dx * dy*dy;
    if (orb_val == "gs4")  return 2.95803989154981 * dx*dx*dx * dy - 2.95803989154981 * dx * dy*dy*dy;

    // h
    if (orb_val == "h0")   return powi(dz,5) + 1.875*powi(dx,4)*dz + 1.875*powi(dy,4)*dz - 5.0*dx*dx*dz*dz*dz - 5.0*dy*dy*dz*dz*dz + 3.75*dx*dx*dy*dy*dz;
    if (orb_val == "hc1")  return 0.484122918275927 * powi(dx,5) + 0.484122918275927 * dx*powi(dy,4) + 0.968245836551854 * powi(dx,3)*dy*dy + 3.87298334620742 * dx*powi(dz,4)
        - 5.80947501931113 * powi(dx,3)*dz*dz - 5.80947501931113 * dx*dy*dy*dz*dz;
    if (orb_val == "hs1")  return 0.484122918275927 * powi(dy,5) + 0.484122918275927 * powi(dx,4)*dy + 0.968245836551854 * dx*dx*powi(dy,3) + 3.87298334620742 * dy*powi(dz,4)
        - 5.80947501931113 * dx*dx*dy*dz*dz - 5.80947501931113 * powi(dy,3)*dz*dz;
    if (orb_val == "hc2")  return 2.56173769148990 * powi(dy,4)*dz + 5.12347538297980 * dx*dx*dz*dz*dz - 2.56173769148990 * powi(dx,4)*dz - 5.12347538297980 * dy*dy*dz*dz*dz;
    if (orb_val == "hs2")  return 10.2469507659596 * dx*dy*dz*dz*dz - 5.12347538297980 * dx*powi(dy,3)*dz - 5.12347538297980 * powi(dx,3)*dy*dz;
    if (orb_val == "hc3")  return -0.522912516583797 * powi(dx,5) + 1.04582503316759 * powi(dx,3)*dy*dy + 4.18330013267038 * powi(dx,3)*dz*dz + 1.56873754975139 * dx*powi(dy,4)
        - 12.5499003980111 * dx*dy*dy*dz*dz;
    if (orb_val == "hs3")  return 0.522912516583797 * powi(dy,5) - 1.04582503316759 * dx*dx*powi(dy,3) - 4.18330013267038 * dy*dy*dz*dz*dz - 1.56873754975139 * powi(dx,4)*dy
        + 12.5499003980111 * dx*dx*dy*dz*dz;
    if (orb_val == "hc4")  return 2.21852991866236 * dx*dx*dz*dz*dz + 2.21852991866236 * dy*dy*dz*dz*dz - 13.3111795119741 * dx*dx*dy*dy*dz;
    if (orb_val == "hs4")  return 8.87411967464942 * dx*dy*dz*dz*dz - 8.87411967464942 * dx*dx*dy*dz*dz;
    if (orb_val == "hc5")  return 0.701560760020114 * powi(dx,5) + 3.50780380010057 * dx*powi(dy,4) - 7.01560760020114 * powi(dx,3)*dy*dy;
    if (orb_val == "hs5")  return 0.701560760020114 * powi(dy,5) + 3.50780380010057 * powi(dx,4)*dy - 7.01560760020114 * dx*dx*powi(dy,3);

    // i
    if (orb_val == "i0")   return powi(dz,6) - 0.3125*powi(dx,6) - 0.3125*powi(dy,6) + 5.625*powi(dx,4)*dz*dz + 5.625*powi(dy,4)*dz*dz
        - 0.9375*powi(dx,4)*dy*dy - 0.9375*powi(dy,4)*dx*dx - 7.5*dx*dx*dz*dz*dz*dz - 7.5*dy*dy*dz*dz*dz*dz + 11.25*dx*dx*dy*dy*dz*dz;
    if (orb_val == "ic1")  return 2.86410980934740*powi(dx,5)*dz + 4.58257569495584*dx*powi(dz,5) - 11.4564392373896*powi(dx,3)*dz*dz*dz
        + 2.86410980934740*dx*powi(dy,4)*dz + 5.72821961869480*dx*dx*dy*dy*dz - 11.4564392373896*dx*dy*dy*dz*dz*dz;
    if (orb_val == "is1")  return 2.86410980934740*powi(dy,5)*dz + 4.58257569495584*dy*powi(dz,5) - 11.4564392373896*powi(dy,3)*dz*dz*dz
        + 2.86410980934740*powi(dx,4)*dy*dz + 5.72821961869480*dx*dx*dy*dy*dz - 11.4564392373896*dx*dx*dy*dz*dz*dz;
    if (orb_val == "ic2")  return 0.452855523318420*powi(dx,6) - 0.452855523318420*powi(dy,6) + 0.452855523318420*powi(dx,4)*dy*dy
        + 7.24568837309472*dx*dx*dz*dz*dz*dz + 7.24568837309472*dy*dy*dz*dz*dz*dz - 0.452855523318420*powi(dx,2)*powi(dy,4)
        - 7.24568837309472*powi(dx,4)*dz*dz - 7.24568837309472*powi(dy,4)*dz*dz;
    if (orb_val == "is2")  return 0.905711046636840*dx*powi(dy,5) + 0.905711046636840*powi(dx,5)*dy + 1.81142209327368*powi(dx,3)*powi(dy,3)
        + 14.4913767461894*dx*dy*powi(dz,4) - 14.4913767461894*dx*powi(dy,3)*dz*dz - 14.4913767461894*powi(dx,3)*dy*dz*dz;
    if (orb_val == "ic3")  return 7.24568837309472*powi(dx,3)*dz*dz*dz - 2.71713313991052*powi(dx,5)*dz + 5.43426627982104*powi(dx,3)*dy*dy*dz
        + 8.15139941973156*dx*powi(dy,4)*dz - 21.7370651192842*dx*dy*dy*dz*dz*dz;
    if (orb_val == "is3")  return 2.71713313991052*powi(dy,5)*dz - 7.24568837309472*powi(dy,3)*dz*dz*dz + 21.7370651192842*dx*dx*dy*dz*dz*dz
        - 5.43426627982104*dx*dx*powi(dy,3)*dz - 8.15139941973156*powi(dx,4)*dy*dz;
    if (orb_val == "ic4")  return -0.496078370824611*powi(dx,6) - 0.496078370824611*powi(dy,6) + 4.96078370824611*powi(dx,4)*dz*dz
        + 4.96078370824611*powi(dy,4)*dz*dz + 2.48039185412305*powi(dx,2)*powi(dy,4) + 2.48039185412305*powi(dx,4)*powi(dy,2)
        - 29.7647022494766*dx*dx*dy*dy*dz*dz;
    if (orb_val == "is4")  return 1.98431348329844*dx*powi(dy,5) - 1.98431348329844*powi(dx,5)*dy + 19.8431348329844*dx*dy*powi(dz,4)
        - 19.8431348329844*dx*dx*dy*dz*dz*dz;
    if (orb_val == "ic5")  return 2.32681380862329*powi(dx,5)*dz + 11.6340690431164*dx*powi(dy,4)*dz - 23.2681380862329*powi(dx,3)*dy*dy*dz;
    if (orb_val == "is5")  return 2.32681380862329*powi(dy,5)*dz + 11.6340690431164*powi(dx,4)*dy*dz - 23.2681380862329*dx*dx*powi(dy,3)*dz;
    if (orb_val == "ic6")  return 0.671693289381396*powi(dx,6) - 0.671693289381396*powi(dy,6) + 10.0753993407209*powi(dx,2)*powi(dy,4) - 10.0753993407209*powi(dx,4)*powi(dy,2);
    if (orb_val == "is6")  return 4.03015973628838*dx*powi(dy,5) + 4.03015973628838*powi(dx,5)*dy - 13.4338657876279*powi(dx,3)*powi(dy,3);

    // j
    if (orb_val == "j0")   return powi(dz,7) + 13.125*powi(dx,4)*powi(dz,3) + 13.125*powi(dy,4)*powi(dz,3)
        - 10.5*dx*dx*powi(dz,5) - 10.5*dy*dy*powi(dz,5) - 2.1875*powi(dx,6)*dz - 2.1875*powi(dy,6)*dz
        + 26.25*dx*dx*dy*dy*powi(dz,3) - 6.5625*dx*dx*powi(dy,4)*dz - 6.5625*powi(dx,4)*dy*dy*dz;
    if (orb_val == "jc1")  return -0.413398642353842*powi(dx,7) + 9.92156741649221*powi(dx,5)*dz*dz + 5.29150262212918*dx*powi(dz,6)
        - 19.8431348329844*powi(dx,3)*dz*dz*dz*dz - 1.24019592706153*powi(dx,5)*dy*dy - 1.24019592706153*dx*powi(dy,6)
        - 0.413398642353842*dx*powi(dy,6) + 9.92156741649221*dx*powi(dy,4)*dz*dz + 19.8431348329844*dx*dx*dy*dy*dz*dz*dz - 19.8431348329844*dx*dy*dy*dz*dz*dz;
    if (orb_val == "js1")  return -0.413398642353842*powi(dy,7) + 9.92156741649221*powi(dy,5)*dz*dz + 5.29150262212918*dy*powi(dz,6)
        - 19.8431348329844*powi(dy,3)*dz*dz*dz*dz - 1.24019592706153*powi(dy,5)*dx*dx - 1.24019592706153*dy*powi(dx,6)
        - 0.413398642353842*dy*powi(dx,6) + 9.92156741649221*dy*powi(dx,4)*dz*dz + 19.8431348329844*dx*dx*dy*dy*dz*dz*dz - 19.8431348329844*dx*dx*dy*dz*dz*dz;
    if (orb_val == "jc2")  return 16.2018517460197*powi(dx,4)*powi(dz,3) + 9.72111104761179*dx*dx*powi(dz,5) + 3.03784720237868*powi(dx,6)*dz
        - 16.2018517460197*powi(dy,4)*powi(dz,3) - 9.72111104761179*dy*dy*powi(dz,5) - 3.03784720237868*powi(dy,6)*dz
        + 3.03784720237868*dx*dx*powi(dy,4)*dz - 3.03784720237868*powi(dx,4)*dy*dy*dz;
    if (orb_val == "js2")  return 19.4422220952236*dx*dy*powi(dz,5) + 12.1513888095147*dx*dx*dy*dy*dz*dz*dz + 6.07569440475737*dx*powi(dy,3)*dz*dz
        + 6.07569440475737*powi(dx,3)*dy*dz*dz - 32.4037034920393*dx*dy*dz*dz*dz*dz*dz - 32.4037034920393*dx*dx*dy*dz*dz*dz;
    if (orb_val == "jc3")  return 0.429616471402110*powi(dx,7) + 11.4564392373896*powi(dx,3)*dz*dz*dz*dz - 0.429616471402110*dx*powi(dy,6)
        - 1.28884941420633*powi(dx,5)*dz*dz - 2.14808235701055*powi(dx,3)*dy*dy*dz*dz - 8.59232942804220*dx*dy*dy*dz*dz*dz
        + 17.1846588560844*dx*dx*dy*dz*dz*dz + 25.7769882841266*dx*powi(dy,4)*dz*dz - 34.3693177121688*powi(dx,3)*dy*dz*dz;
    if (orb_val == "js3")  return -0.429616471402110*powi(dy,7) + 0.429616471402110*dy*powi(dx,6) + 1.28884941420633*powi(dy,5)*dz*dz
        + 2.14808235701055*dx*dx*powi(dy,3)*dz*dz + 8.59232942804220*dx*dx*dy*dz*dz*dz - 11.4564392373896*powi(dy,3)*dz*dz*dz*dz
        + 34.3693177121688*dx*dx*dy*dz*dz*dz - 17.1846588560844*dx*powi(dy,4)*dz*dz - 25.7769882841266*powi(dx,4)*dy*dz*dz;
    if (orb_val == "jc4")  return 9.49917759598167*powi(dx,4)*powi(dz,3) + 9.49917759598167*powi(dy,4)*powi(dz,3) - 2.84975327879450*powi(dx,6)*dz
        - 2.84975327879450*powi(dy,6)*dz + 14.2487663939725*dx*dx*dy*dy*dz*dz*dz + 14.2487663939725*dx*dy*dy*dz*dz*dz - 56.9950655758900*dx*dx*dy*dz*dz*dz;
    if (orb_val == "js4")  return 11.3990131151780*dx*dy*powi(dz,5) + 37.9967103839267*dx*dx*dy*dz*dz*dz - 11.3990131151780*dx*dx*dy*dz*dz*dz - 37.9967103839267*dx*dy*dz*dz*dz*dz*dz;
    if (orb_val == "jc5")  return -0.474958879799083*powi(dx,7) + 5.69950655758900*powi(dx,5)*dz*dz + 4.27462991819175*dx*powi(dz,6)
        + 2.37479439899542*powi(dx,3)*dy*dy*dz - 2.37479439899542*dx*dy*dy*dz*dz*dz + 28.4975327879450*dx*dx*dy*dz*dz*dz - 56.9950655758900*powi(dx,3)*dy*dz*dz;
    if (orb_val == "js5")  return -0.474958879799083*powi(dy,7) + 5.69950655758900*powi(dy,5)*dz*dz + 4.27462991819175*dy*powi(dz,6)
        + 2.37479439899542*dx*dx*powi(dy,3)*dz - 2.37479439899542*dx*dx*dy*dz*dz*dz + 28.4975327879450*dx*dx*dy*dz*dz*dz - 56.9950655758900*dx*dx*powi(dy,3)*dz;
    if (orb_val == "jc6")  return 2.42182459624970*powi(dx,6)*dz - 2.42182459624970*powi(dy,6)*dz + 36.3273689437454*dx*dx*dy*dz*dz*dz - 36.3273689437454*dx*dy*dy*dz*dz*dz;
    if (orb_val == "js6")  return 14.5309475774982*dx*powi(dy,5)*dz + 14.5309475774982*powi(dx,5)*dy*dz - 48.4364919249939*powi(dx,3)*powi(dy,3)*dz;
    if (orb_val == "jc7")  return 0.647259849287749*powi(dx,7) + 22.6540947250712*powi(dx,3)*powi(dy,4) - 4.53081894501425*dx*powi(dy,6) - 13.5924568350427*powi(dx,5)*dy*dy;
    if (orb_val == "js7")  return -0.647259849287749*powi(dy,7) + 4.53081894501425*powi(dx,6)*dy + 13.5924568350427*dx*dx*powi(dy,5) - 22.6540947250712*powi(dx,4)*powi(dy,3);

    // Fallback
    return 0.0;
}


py::array_t<double> electron_density(
    py::list data,                   // List of dicts (basis functions)
    py::array_t<double> coordinates, // shape: (n_atoms, 3)
    py::array_t<double> points,      // shape: (n_points, 3)
    std::vector<double> cmo,         // MO coefficients, size: n_basis
    py::object /*ang_res_lambda*/    // Not used anymore, for interface compatibility
) {
    // Convert coordinates to C++ array
    auto coords = coordinates.unchecked<2>();
    //size_t n_atoms = coords.shape(0);

    // Pre-convert all basis set info to C++ arrays/vectors for thread safety
    size_t n_basis = data.size();
    std::vector<int> centers(n_basis);
    std::vector<std::string> orb_vals(n_basis);
    std::vector<std::vector<double>> exps_list(n_basis), coeffs_list(n_basis);

    for (size_t i = 0; i < n_basis; ++i) {
        py::dict basis = data[i];
        centers[i] = basis["CENTER"].cast<int>() - 1;
        orb_vals[i] = basis["orb_val"].cast<std::string>();
        exps_list[i] = basis["exps"].cast<std::vector<double>>();
        coeffs_list[i] = basis["coeffs"].cast<std::vector<double>>();
    }

    // Prepare points
    auto pts = points.unchecked<2>();
    size_t n_points = pts.shape(0);

    py::array_t<double> result(n_points);
    auto res = result.mutable_unchecked<1>();

    // Parallelize over grid points
    #pragma omp parallel for schedule(dynamic)
    for (size_t ipt = 0; ipt < n_points; ++ipt) {
        double px = pts(ipt, 0);
        double py_ = pts(ipt, 1);
        double pz = pts(ipt, 2);
        double val = 0.0;

        for (size_t i = 0; i < n_basis; ++i) {
            double c = cmo[i];
            if (std::abs(c) < 1e-15) continue;

            int center = centers[i];
            double cx = coords(center, 0);
            double cy = coords(center, 1);
            double cz = coords(center, 2);

            double dx = px - cx;
            double dy = py_ - cy;
            double dz = pz - cz;
            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > 200*200) continue;

            const std::vector<double>& exps = exps_list[i];
            double zeta_small = *std::min_element(exps.begin(), exps.end());
            if (std::exp(-zeta_small * r2) < 1e-15) continue;

            double angular_part = solid_harmonic(dx, dy, dz, orb_vals[i]);
            const std::vector<double>& coeffs = coeffs_list[i];

            double bas_res = 0.0;
            for (size_t j = 0; j < coeffs.size(); ++j) {
                bas_res += coeffs[j] * std::exp(-exps[j] * r2);
            }

            val += c * bas_res * angular_part;
        }
        res(ipt) = val;
    }

    return result;
}

PYBIND11_MODULE(electron_density_opt_omp, m) {
    m.def("electron_density", &electron_density,
          py::arg("data"),
          py::arg("coordinates"),
          py::arg("points"),
          py::arg("cmo"),
          py::arg("ang_res_lambda") = py::none(), // Placeholder for compatibility
          "Vectorized electron density with solid harmonics and OpenMP");
}




