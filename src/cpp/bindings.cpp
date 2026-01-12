/**
    bindings.cpp
    Purpose: Python bindings for PALMS (Image and Point Cloud solvers).
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "linewiseAffineMS.h"

namespace py = pybind11;

// --- 1. Image Solver (Dense/Raster) ---
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
linewise_solver_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> f_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> a_data_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> b_data_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dir_in,
    double gamma_s,
    double eta_s,
    py::array_t<double, py::array::f_style | py::array::forcecast> c_lin_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> s_lin_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> c_const_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> s_const_in,
    int nr_threads
) {
    py::buffer_info f_buf = f_in.request();
    py::buffer_info dir_buf = dir_in.request();
    
    if (dir_buf.size != 2) {
        throw std::runtime_error("Direction vector must have length 2");
    }

    int m = f_buf.shape[0];
    int n = f_buf.shape[1];
    int nr_channels = (f_buf.ndim == 3) ? f_buf.shape[2] : 1;

    std::vector<ssize_t> shape;
    if (f_buf.ndim == 3) shape = {m, n, nr_channels};
    else shape = {m, n};

    auto u_out = py::array_t<double, py::array::f_style>(shape);
    auto a_out = py::array_t<double, py::array::f_style>(shape);
    auto b_out = py::array_t<double, py::array::f_style>(shape);

    ArmadilloConverter(
        static_cast<double*>(f_buf.ptr),
        static_cast<double*>(a_data_in.request().ptr),
        static_cast<double*>(b_data_in.request().ptr),
        static_cast<double*>(c_lin_in.request().ptr),
        static_cast<double*>(s_lin_in.request().ptr),
        static_cast<double*>(c_const_in.request().ptr),
        static_cast<double*>(s_const_in.request().ptr),
        static_cast<double*>(u_out.request().ptr),
        static_cast<double*>(a_out.request().ptr),
        static_cast<double*>(b_out.request().ptr),
        static_cast<double*>(dir_buf.ptr),
        gamma_s, eta_s,
        m, n, nr_channels, nr_threads
    );

    // FIX: Use std::make_tuple (C++ syntax), not std.make_tuple (Python syntax)
    return std::make_tuple(u_out, a_out, b_out);
}

// --- 2. Point Cloud Solver (Sparse/Stripes) ---
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
sparse_solver_py(
    int num_points, 
    int nr_channels,
    // Dummy inputs for shape/compatibility (content ignored if stripes provided)
    py::array_t<double, py::array::f_style> u_in,
    py::array_t<double, py::array::f_style> a_in,
    py::array_t<double, py::array::f_style> b_in,
    // Stripes: List of (indices, data) tuples
    std::vector<std::pair<py::array_t<size_t>, py::array_t<double>>> stripes_u,
    std::vector<std::pair<py::array_t<size_t>, py::array_t<double>>> stripes_a,
    std::vector<std::pair<py::array_t<size_t>, py::array_t<double>>> stripes_b,
    double gamma_s,
    double eta_s,
    py::array_t<double> c_lin,
    py::array_t<double> s_lin,
    py::array_t<double> c_const,
    py::array_t<double> s_const
) {
    // Outputs (N x 1 x C)
    auto u_out = py::array_t<double, py::array::f_style>({num_points, 1, nr_channels});
    auto a_out = py::array_t<double, py::array::f_style>({num_points, 1, nr_channels});
    auto b_out = py::array_t<double, py::array::f_style>({num_points, 1, nr_channels});

    // Wrap in Armadillo Cubes
    cube u_out_cube(static_cast<double*>(u_out.request().ptr), num_points, 1, nr_channels, false, true);
    cube a_out_cube(static_cast<double*>(a_out.request().ptr), num_points, 1, nr_channels, false, true);
    cube b_out_cube(static_cast<double*>(b_out.request().ptr), num_points, 1, nr_channels, false, true);

    // Helper to convert Python stripes to C++ Stripe objects
    auto convert_stripes = [&](const auto& py_stripes) {
        std::vector<Stripe> cpp_stripes;
        cpp_stripes.reserve(py_stripes.size());
        
        for (const auto& s : py_stripes) {
            // Indices
            auto idx_buf = s.first.request();
            
            // Cast to uvec::elem_type* to satisfy Armadillo on ARM64 macOS
            uvec indices(reinterpret_cast<uvec::elem_type*>(idx_buf.ptr), idx_buf.size, false, true);
            
            // Data (Must be transposed on Python side to be Channels x N_stripe here)
            auto dat_buf = s.second.request();
            mat data(static_cast<double*>(dat_buf.ptr), nr_channels, idx_buf.size, false, true);
            
            cpp_stripes.emplace_back(data, indices);
        }
        return cpp_stripes;
    };

    std::vector<Stripe> L_udata = convert_stripes(stripes_u);
    std::vector<Stripe> L_adata = convert_stripes(stripes_a);
    std::vector<Stripe> L_bdata = convert_stripes(stripes_b);

    // Rotation Matrices
    mat C_linear(static_cast<double*>(c_lin.request().ptr), c_lin.shape(0), c_lin.shape(1), false, true);
    mat S_linear(static_cast<double*>(s_lin.request().ptr), s_lin.shape(0), s_lin.shape(1), false, true);
    mat C_const(static_cast<double*>(c_const.request().ptr), c_const.shape(0), c_const.shape(1), false, true);
    mat S_const(static_cast<double*>(s_const.request().ptr), s_const.shape(0), s_const.shape(1), false, true);

    // Run Solver
    RunSolverOnStripes(u_out_cube, a_out_cube, b_out_cube, nr_channels, 
                       L_udata, L_adata, L_bdata, gamma_s, eta_s, 
                       C_linear, S_linear, C_const, S_const);

    // FIX: Use std::make_tuple
    return std::make_tuple(u_out, a_out, b_out);
}

PYBIND11_MODULE(palms_cpp, m) {
    m.doc() = "Python bindings for PALMS";
    
    m.def("linewise_solver", &linewise_solver_py, "Dense Image Solver",
          py::arg("f"), py::arg("a_data"), py::arg("b_data"), 
          py::arg("dir"), py::arg("gamma"), py::arg("eta"),
          py::arg("c_lin"), py::arg("s_lin"), py::arg("c_const"), py::arg("s_const"),
          py::arg("nr_threads") = 1);
          
    m.def("sparse_solver", &sparse_solver_py, "Sparse Point Cloud Solver");
}