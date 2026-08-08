// Shared serialization + loading for oracle-golden test data.
//
// One text file per system at tests/golden/data/<name>.golden. Every numeric
// quantity is written at full double precision (%.17g, exact round-trip) so the
// frozen data reproduces the generating code bit-for-bit. Both the generator
// (tests/gen_golden.cc) and the tests include this header, so the writer and
// reader uses the same format.
//
// File format (whitespace/newline separated tokens; '#' line = comment):
//   system  <label>
//   reference <rhf|uhf>
//   nbf <int> / n_docc <int> / n_alpha <int> / n_beta <int>
//   scalar  <name> <value>
//   matrix  <name> <rows> <cols>   <rows*cols values, row-major>
//   vector  <name> <n>             <n values>
//   list    <name> <n>             <n values>            (a raw std::vector<double>, e.g. the ERI vector)
//   tensor4 <name> <d0> <d1> <d2> <d3>   <product values, index order i,j,k,l>

#ifndef SCF_TESTS_GOLDEN_IO
#define SCF_TESTS_GOLDEN_IO

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace golden {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;

// Single source of truth for the golden systems
struct GoldenSystem {
    std::string name;       // file stem, e.g. "h2o_sto-3g"
    std::string label;      // bench/reference.json key, e.g. "h2o/sto-3g"
    std::string xyz;        // path relative to repo root
    std::string basis;
    int mult;
    int charge;
    std::string reference;  // "rhf" or "uhf"
};

// Edit golden systems here
inline const std::vector<GoldenSystem>& golden_systems() {
    static const std::vector<GoldenSystem> systems = {
        {"h2o_sto-3g",          "h2o/sto-3g",          "input/h2o.xyz",          "sto-3g", 1, 0, "rhf"},
        {"ch3_sto-3g",          "ch3/sto-3g",          "input/ch3.xyz",          "sto-3g", 2, 0, "uhf"},
        {"h2_stretched_sto-3g", "h2_stretched/sto-3g", "input/h2_stretched.xyz", "sto-3g", 1, 0, "rhf"},
    };
    return systems;
}

inline std::string data_path(const std::string& name) {
    return std::string(SCF_REPO_ROOT) + "/tests/golden/data/" + name + ".golden";
}

// Writers (used by gen_golden)

inline void write_num(std::ostream& os, double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    os << buf;
}

inline void write_scalar(std::ostream& os, const std::string& name, double v) {
    os << "scalar " << name << " ";
    write_num(os, v);
    os << "\n";
}

inline void write_matrix(std::ostream& os, const std::string& name, const Matrix& M) {
    // Row-major format
    os << "matrix " << name << " " << M.rows() << " " << M.cols() << "\n";
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (j) os << " ";
            write_num(os, M(i, j));
        }
        os << "\n";
    }
}

inline void write_vector(std::ostream& os, const std::string& name, const Vector& v) {
    os << "vector " << name << " " << v.size() << "\n";
    for (int i = 0; i < v.size(); ++i) {
        if (i) os << " ";
        write_num(os, v(i));
    }
    os << "\n";
}

inline void write_list(std::ostream& os, const std::string& name, const std::vector<double>& v) {
    os << "list " << name << " " << v.size() << "\n";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << " ";
        write_num(os, v[i]);
    }
    os << "\n";
}

inline void write_tensor4(std::ostream& os, const std::string& name, const Tensor4D& t) {
    const int d0 = static_cast<int>(t.dimension(0));
    const int d1 = static_cast<int>(t.dimension(1));
    const int d2 = static_cast<int>(t.dimension(2));
    const int d3 = static_cast<int>(t.dimension(3));
    os << "tensor4 " << name << " " << d0 << " " << d1 << " " << d2 << " " << d3 << "\n";
    bool first = true;
    for (int i = 0; i < d0; ++i)
    for (int j = 0; j < d1; ++j)
    for (int k = 0; k < d2; ++k)
    for (int l = 0; l < d3; ++l) {
        if (!first) os << " ";

        // Prints all out in a single line
        write_num(os, t(i, j, k, l));
        first = false;
    }
    os << "\n";
}


// Reader (used by tests)

class Golden {
public:
    std::string system;
    std::string reference;
    int nbf = 0, n_docc = 0, n_alpha = 0, n_beta = 0;

    // Public methods to access stored golden data
    double scalar(const std::string& name) const { return get(scalars_, name); }

    Matrix mat(const std::string& name) const {
        const Entry& e = get_entry(matrices_, name);
        Matrix M(e.dims[0], e.dims[1]);
        int idx = 0;
        for (int i = 0; i < e.dims[0]; ++i)
            for (int j = 0; j < e.dims[1]; ++j)
                M(i, j) = e.data[idx++];   // row-major, matching the writer
        return M;
    }

    Vector vec(const std::string& name) const {
        const Entry& e = get_entry(vectors_, name);
        Vector v(e.dims[0]);
        for (int i = 0; i < e.dims[0]; ++i) v(i) = e.data[i];
        return v;
    }

    std::vector<double> list(const std::string& name) const {
        return get_entry(lists_, name).data;
    }

    Tensor4D tensor(const std::string& name) const {
        const Entry& e = get_entry(tensors_, name);
        Tensor4D t(e.dims[0], e.dims[1], e.dims[2], e.dims[3]);
        int idx = 0;
        for (int i = 0; i < e.dims[0]; ++i)
        for (int j = 0; j < e.dims[1]; ++j)
        for (int k = 0; k < e.dims[2]; ++k)
        for (int l = 0; l < e.dims[3]; ++l)
            t(i, j, k, l) = e.data[idx++];
            return t;
    }

    bool has_scalar(const std::string& name) const { return scalars_.count(name) > 0; }

private:
    // For each structured data, dimensions is stored also
    struct Entry { std::vector<int> dims; std::vector<double> data; };

    // A map for each data type
    std::map<std::string, double> scalars_;
    std::map<std::string, Entry>  matrices_, vectors_, lists_, tensors_;

    static double get(const std::map<std::string, double>& m, const std::string& name) {
        auto it = m.find(name);
        if (it == m.end()) throw std::runtime_error("golden: missing scalar '" + name + "'");
        return it->second;
    }
    static const Entry& get_entry(const std::map<std::string, Entry>& m, const std::string& name) {
        auto it = m.find(name);
        if (it == m.end()) throw std::runtime_error("golden: missing quantity '" + name + "'");
        return it->second;
    }

    friend Golden load_golden(const std::string& name);
};

inline Golden load_golden(const std::string& name) {
    const std::string path = data_path(name);
    std::ifstream in(path);
    if (!in) throw std::runtime_error("golden: cannot open " + path +
                                       " (run ./build/tests/gen_golden to generate)");

    Golden g;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;

        if (tag == "system")         { ls >> g.system; }
        else if (tag == "reference") { ls >> g.reference; }
        else if (tag == "nbf")       { ls >> g.nbf; }
        else if (tag == "n_docc")    { ls >> g.n_docc; }
        else if (tag == "n_alpha")   { ls >> g.n_alpha; }
        else if (tag == "n_beta")    { ls >> g.n_beta; }

        // Parse the data into the map of each type
        else if (tag == "scalar") {    
            std::string qname; double v; ls >> qname >> v; // quantity name
            g.scalars_[qname] = v;
        }
        else if (tag == "matrix" || tag == "vector" || tag == "list" || tag == "tensor4") {
            std::string qname; ls >> qname;
            Golden::Entry e;

            // Calculate the total element count
            int ndim = (tag == "matrix") ? 2 : (tag == "tensor4") ? 4 : 1;
            for (int d = 0; d < ndim; ++d) { int x; ls >> x; e.dims.push_back(x); }
            size_t count = 1;
            for (int d : e.dims) count *= static_cast<size_t>(d);
            e.data.reserve(count);

            // Most confusing part here:
            // Reads from `in` (file stream), not `ls` (line stream)
            // Current line is the header only, while the data lives on the next lines
            // Use `in` to skip any whitespace to read in per token
            for (size_t i = 0; i < count; ++i) { double v; in >> v; e.data.push_back(v); }
            // After this, the file pointer is at the end of the data section
            // consume the rest of the data line's newline
            std::getline(in, line);

            if (tag == "matrix")       g.matrices_[qname] = std::move(e);
            else if (tag == "vector")  g.vectors_[qname]  = std::move(e);
            else if (tag == "list")    g.lists_[qname]    = std::move(e);
            else                       g.tensors_[qname]  = std::move(e);
        }
        else {
            throw std::runtime_error("golden: unknown tag '" + tag + "' in " + path);
        }
    }
    return g;
}

} // namespace golden

#endif
