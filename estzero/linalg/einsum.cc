#include <algorithm>
#include <array>
#include <bitset>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "est_gemm.h"
#include "einsum.h"
#include "odometer.h"
#include "tensor.h"

namespace {

// Labels are single ASCII characters, so per-label tables are plain arrays/bitsets indexed by the character
// instead of hash maps/sets (no heap allocation, no hashing).
constexpr int NUM_LABELS = 128;
using LabelSet = std::bitset<NUM_LABELS>;

unsigned char to_ascii(char c) { return static_cast<unsigned char>(c); }

LabelSet labels_of(std::string_view term) {
    LabelSet set;
    for (char c : term) set.set(to_ascii(c));
    return set;
}

// Einsum plan for one pairwise contraction. NT = number of terms (inputs + output), the same NT as Odometer<NT>:
// 2 (1 input, 1 output) or 3 (2 inputs, the GEMM-style case).
template <int NT>
struct EinsumPlan {
    static_assert(NT == 2 || NT == 3, "EinsumPlan: NT must be 2 (1 input) or 3 (2 inputs)");

    // String-only: built once per distinct subscript string and cached (see plan_for)
    std::array<std::string, NT> terms;             // input terms, then the output term last (spaces removed)
    std::array<std::vector<Index>, NT> term_cols;  // per term: odometer column of each axis' label
    Index ncols = 0;                               // distinct labels, numbered in order of first appearance

    // try_gemm's string-only decisions (NT == 3). Not eligible for a label repeated within an operand
    // (i.e. "piqi") or a batch label (in A, B and C).
    bool gemm_eligible = false;
    std::vector<Index> left_axes, summed_axes, right_axes; // axes of A kept, axes of A/B summed, axes of B kept
    std::vector<Index> permA, permB;               // (left, summed) for A, (summed, right) for B
    std::vector<Index> perm_out;                   // GEMM output (left labels + right labels) -> requested order
};

// The shape-dependent part. Rebuilt on every call from the operands (it's a handful of integer ops).
template <int NT>
struct Binding {
    std::array<Index, NUM_LABELS> extents;     // one extent per column (label); only the first ncols are set
    std::vector<Index> result_shape;

    bool use_gemm = false;                     // gemm_eligible, a GEMM scheme, and above the FLOP guard
    Index left_size = 1, summed_size = 1, right_size = 1;
    std::vector<Index> combined_shape;         // GEMM output shape before the final permutation
};

// Validates a subscript string (the shape-independent gates) and splits it into its terms, inputs first and
// the output last. Shared by the pairwise plans and the 3+ operand chain.
std::vector<std::string> parse_terms(std::string_view raw) {
    std::string subscripts;
    for (char c : raw) if (c != ' ') subscripts += c;

    for (char c : subscripts) {
        if (to_ascii(c) >= NUM_LABELS) {
            throw std::invalid_argument("Subscript labels must be ASCII characters: " + std::string(raw));
        }
    }

    // Exactly one arrow check
    const size_t arrow_pos = subscripts.find("->");
    if (arrow_pos == std::string::npos || subscripts.find("->", arrow_pos + 2) != std::string::npos) {
        throw std::invalid_argument("Subscripts must contain exactly one '->': " + std::string(raw));
    }
    const std::string_view lhs = std::string_view(subscripts).substr(0, arrow_pos);
    const std::string_view rhs = std::string_view(subscripts).substr(arrow_pos + 2);

    // Empty term check (i.e., ",jk" or "ij,"). An empty RHS denotes a scalar result.
    std::vector<std::string> terms;
    for (size_t start = 0;;) {
        const size_t comma = lhs.find(',', start);
        const std::string_view term = lhs.substr(start, comma == std::string_view::npos ? lhs.npos : comma - start);
        if (term.empty()) {
            throw std::invalid_argument("Subscripts contain empty terms: " + std::string(raw));
        }
        terms.emplace_back(term);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }

    // Output index validation (RHS indices must be unique and a subset of LHS)
    LabelSet lhs_labels;
    for (const std::string& term : terms) lhs_labels |= labels_of(term);
    LabelSet rhs_labels;
    for (char c : rhs) {
        const auto l = to_ascii(c);
        if (rhs_labels[l]) {
            throw std::invalid_argument(std::string("Output subscript contains duplicate index '") + c + "'.");
        }
        if (!lhs_labels[l]) {
            throw std::invalid_argument(std::string("Output subscript contains index '") + c + "' not found in input terms.");
        }
        rhs_labels.set(l);
    }

    terms.emplace_back(rhs); // The last term is the rhs
    return terms;
}

void check_operand_count(size_t n_operands, size_t n_input_terms) {
    if (n_operands != n_input_terms) {
        throw std::invalid_argument("Provided " + std::to_string(n_operands) +
                                    " operands, but subscript expects " + std::to_string(n_input_terms) + " terms.");
    }
}

// Builds the string-only part of try_gemm (see EinsumPlan).
void plan_gemm_strings(EinsumPlan<3>& plan) {
    const std::string& A = plan.terms[0];
    const std::string& B = plan.terms[1];
    const std::string& C = plan.terms[2]; // requested output order, used for the final permutation

    const LabelSet a_set = labels_of(A), b_set = labels_of(B), c_set = labels_of(C);

    // Repeated label in a single operand (i.e. "piqi"): a set smaller than the term means a duplicate
    if (A.size() != a_set.count() || B.size() != b_set.count()) return;

    // Batch index check (index in A is not in B and C, or same for B)
    // This doesn't occur in this codebase, but just decline for simplcity here, not doing GEMM-supported batch
    for (char c : A) {
        const auto l = to_ascii(c);
        if (b_set[l] && c_set[l]) return;
        if (!c_set[l] && !b_set[l]) return;
    }
    for (char c : B) {
        const auto l = to_ascii(c);
        if (!c_set[l] && !a_set[l]) return;
    }

    // Walk A once: kept-left axes (C has it), summed axes (C doesn't have it).
    // The summed labels are the same for both A and B, so B finds its summed axes by A's summed labels.
    std::string summed_labels;
    for (Index i = 0; i < std::ssize(A); ++i) {
        if (c_set[to_ascii(A[i])]) plan.left_axes.push_back(i);
        else { plan.summed_axes.push_back(i); summed_labels += A[i]; }
    }
    for (Index i = 0; i < std::ssize(B); ++i) {
        if (c_set[to_ascii(B[i])]) plan.right_axes.push_back(i);
    }

    plan.permA = plan.left_axes;
    plan.permA.insert(plan.permA.end(), plan.summed_axes.begin(), plan.summed_axes.end());
    for (char c : summed_labels) plan.permB.push_back(B.find(c));
    plan.permB.insert(plan.permB.end(), plan.right_axes.begin(), plan.right_axes.end());

    // combined_labels: left_axes + right_axes (what GEMM produces); perm_out reorders it into C
    std::string combined_labels;
    for (Index i : plan.left_axes)  combined_labels += A[i];
    for (Index i : plan.right_axes) combined_labels += B[i];
    for (char c : C) plan.perm_out.push_back(combined_labels.find(c));

    plan.gemm_eligible = true;
}

// The string-only part of the plan.
template <int NT>
EinsumPlan<NT> make_plan(std::string_view raw) {
    std::vector<std::string> terms = parse_terms(raw);
    check_operand_count(NT - 1, terms.size() - 1);

    EinsumPlan<NT> plan;
    std::move(terms.begin(), terms.end(), plan.terms.begin());

    // One column per distinct label, numbered in order of first appearance across the input terms
    std::array<Index, NUM_LABELS> char_to_col;
    char_to_col.fill(-1);
    for (int t = 0; t < NT - 1; ++t) {
        for (char c : plan.terms[t]) {
            Index& col = char_to_col[to_ascii(c)];
            if (col < 0) col = plan.ncols++;
        }
    }
    for (int t = 0; t < NT; ++t) {
        for (char c : plan.terms[t]) plan.term_cols[t].push_back(char_to_col[to_ascii(c)]);
    }

    if constexpr (NT == 3) plan_gemm_strings(plan);
    return plan;
}

// Validates the operands against the terms (ndim, one extent per label), then binds the result shape and, for GEMM, the sizes.
template <int NT>
Binding<NT> bind_operands(const EinsumPlan<NT>& plan, Operands operands) {
    Binding<NT> b;
    std::fill_n(b.extents.begin(), plan.ncols, -1);

    for (int t = 0; t < NT - 1; ++t) {
        const std::string& term = plan.terms[t];
        const auto& shape = operands[t]->shape();
        if (shape.size() != term.size()) {
            throw std::invalid_argument("Operand " + std::to_string(t) + " has " + std::to_string(shape.size()) +
                                        " dimensions, but subscript '" + term + "' expects " +
                                        std::to_string(term.size()) + ".");
        }
        for (size_t a = 0; a < term.size(); ++a) {
            Index& extent = b.extents[plan.term_cols[t][a]];
            if (extent < 0) {
                extent = shape[a];
            } else if (extent != shape[a]) {
                throw std::invalid_argument(std::string("Dimension mismatch for index '") + term[a] +
                                            "': expected size " + std::to_string(extent) +
                                            ", but got " + std::to_string(shape[a]) + ".");
            }
        }
    }

    b.result_shape.reserve(plan.term_cols.back().size());
    for (Index col : plan.term_cols.back()) b.result_shape.push_back(b.extents[col]);

    if constexpr (NT == 3) {
        if (plan.gemm_eligible && gs != GEMMScheme::ODO) {
            const auto& A_shape = operands[0]->shape();
            const auto& B_shape = operands[1]->shape();
            for (Index i : plan.left_axes)   b.left_size *= A_shape[i];
            for (Index i : plan.summed_axes) b.summed_size *= A_shape[i]; // same if taken from B
            for (Index i : plan.right_axes)  b.right_size *= B_shape[i];

            // Guard on the contraction's total work (FLOPs ~ left*summed*right) instead of raw element count.
            b.use_gemm = b.left_size * b.summed_size * b.right_size >= gemm_threshold;
            if (b.use_gemm) {
                b.combined_shape.reserve(plan.left_axes.size() + plan.right_axes.size());
                for (Index i : plan.left_axes)  b.combined_shape.push_back(A_shape[i]);
                for (Index i : plan.right_axes) b.combined_shape.push_back(B_shape[i]);
            }
        }
    }
    return b;
}

// Transparent hash, so the caches can be searched by std::string_view without building a std::string
struct SubscriptHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

template <typename T>
using SubscriptCache = std::unordered_map<std::string, T, SubscriptHash, std::equal_to<>>;

// The plan cache: one parse per distinct subscript string with thread_local
template <int NT>
const EinsumPlan<NT>& plan_for(std::string_view subscripts) {
    thread_local SubscriptCache<EinsumPlan<NT>> cache;
    if (auto it = cache.find(subscripts); it != cache.end()) return it->second;
    return cache.emplace(std::string(subscripts), make_plan<NT>(subscripts)).first->second;
}

bool try_gemm(const EinsumPlan<3>& plan, Binding<3>& b, Operands operands, Tensor& result) {
    if (!b.use_gemm) return false;

    // Explicitly pass as a span. The template argument resolves before the std::span overload, so it tried
    // implicit conversion: `static_cast<long>(std::vector<long>)` -> error
    Tensor M = operands[0]->permute(std::span<const Index>(plan.permA)).reshape(b.left_size, b.summed_size);
    Tensor N = operands[1]->permute(std::span<const Index>(plan.permB)).reshape(b.summed_size, b.right_size);

    // Check if the permutation already match the correct ordering to avoid additional allocation/copy
    bool correct_perm = true;
    for (int i=0; i != result.ndim(); i++) {
        if (plan.perm_out[i] != i) {
            // Permutation doesn't match, need to allocate and copy
            correct_perm = false; break;
        }
    }

    if (correct_perm) {
        // Use the result tensor directly if the permutation matches
        call_gemm(gs, static_cast<int>(b.left_size), static_cast<int>(b.right_size), static_cast<int>(b.summed_size),
                  M.data_ptr(), N.data_ptr(), result.data_ptr());
        return true;
    }

    // Empty gemm output without needing for 0-initialization (beta=0.0)
    Tensor gemm_out = Tensor::empty(std::move(b.combined_shape));

    call_gemm(gs, static_cast<int>(b.left_size), static_cast<int>(b.right_size), static_cast<int>(b.summed_size),
              M.data_ptr(), N.data_ptr(), gemm_out.data_ptr());

    // Reorder combined's axes into the caller's actual requested order
    result.assign(gemm_out.permute(std::span<const Index>(plan.perm_out)));
    return true;
}

// Generic odometer route when GEMM is declined
template <int NT>
void odometer_contract(const EinsumPlan<NT>& plan, const Binding<NT>& b, Operands operands, Tensor& result) {
    // One odometer axis ("column") per distinct label
    // Same row-wise innermost strategy seen in Tensor ops: odometer only walks ncols-1 axes
    Odometer<NT> odo(plan.ncols - 1);

    Index total_comb_outer = 1;
    for (Index col = 0; col < plan.ncols - 1; ++col) {
        odo.dims[col] = b.extents[col];
        total_comb_outer *= b.extents[col];
    }

    // stride_table[col][term] = how much term's flat offset moves per +1 of the label in column `col`. Each
    // term's entries are a relabeled copy of its own strides.
    // The stride table is still built the same, because indicating `plan.ncols - 1` above already ensures
    // that it walks from the ncols-1 axis (see `odometer.h`: `for (Index axis = naxes - 1 ...`)
    for (int term = 0; term < NT - 1; ++term) {
        const auto& strides = operands[term]->strides();
        for (size_t axis = 0; axis < plan.term_cols[term].size(); ++axis) {
            odo.stride_table[plan.term_cols[term][axis]][term] += strides[axis];
        }
    }
    const auto& result_strides = result.strides();
    for (size_t axis = 0; axis < plan.term_cols.back().size(); ++axis) {
        odo.stride_table[plan.term_cols.back()[axis]][NT - 1] += result_strides[axis];
    }

    for (Index c = 0; c < total_comb_outer; ++c) {
        const Index last = plan.ncols - 1; // innermost global axis index
        for (Index j = 0; j != b.extents[last]; ++j) { // derives from b.extents, odo.dims wasn't built to `last`
            double product = 1.0;
            for (int term = 0; term < NT - 1; ++term) {
                product *= operands[term]->flat_index(odo.flat[term] + j * odo.stride_table[last][term]);
            }
            result.flat_index(odo.flat[NT - 1] + j * odo.stride_table[last][NT - 1]) += product; 
        }

        odo.next();
    }
}

// One contraction of 1 (NT = 2) or 2 (NT = 3) operands, given its plan.
template <int NT>
Tensor pairwise_einsum(const EinsumPlan<NT>& plan, Operands operands) {
    Binding<NT> b = bind_operands(plan, operands);
    Tensor result = Tensor::empty(std::move(b.result_shape)); // constructor already computes correct natural strides

    if constexpr (NT == 3) {
        if (try_gemm(plan, b, operands, result)) return result; // modifies `result` directly
    }

    // Odometer path still needs result to be 0-initialized
    result.fill(0.0);
    odometer_contract<NT>(plan, b, operands, result);
    return result;
}

// Builds the pairwise subscripts of the left-to-right chain for 3+ operands. Each intermediate keeps exactly the
// labels still needed later (by a remaining input or the output), in sorted order; the last step produces the
// output in its exact requested order. String-only, so it is cached per subscript like the plans.
std::vector<std::string> separate_chain(std::string_view raw) {
    const std::vector<std::string> terms = parse_terms(raw);
    const size_t n_inputs = terms.size() - 1;

    std::vector<std::string> chain;
    std::string current = terms[0];
    for (size_t next = 1; next < n_inputs; ++next) {
        std::string next_str;
        if (next == n_inputs - 1) {
            next_str = terms.back();
        } else {
            LabelSet needed_later; // labels of the remaining inputs and the output
            for (size_t t = next + 1; t < terms.size(); ++t) needed_later |= labels_of(terms[t]);
            const LabelSet keep = (labels_of(current) | labels_of(terms[next])) & needed_later;
            for (int l = 0; l < NUM_LABELS; ++l) if (keep[l]) next_str += static_cast<char>(l);
        }
        chain.push_back(current + "," + terms[next] + "->" + next_str);
        current = std::move(next_str);
    }
    return chain;
}

const std::vector<std::string>& chain_for(std::string_view subscripts, size_t n_operands) {
    thread_local SubscriptCache<std::vector<std::string>> cache;
    auto it = cache.find(subscripts);
    if (it == cache.end()) {
        it = cache.emplace(std::string(subscripts), separate_chain(subscripts)).first;
    }
    check_operand_count(n_operands, it->second.size() + 1); // a chain of k steps contracts k + 1 inputs
    return it->second;
}

} // namespace

Tensor einsum(std::string_view subscripts, Operands operands) {
    switch (operands.size()) {
        case 0: throw std::invalid_argument("einsum needs at least one operand: " + std::string(subscripts));
        case 1: return pairwise_einsum(plan_for<2>(subscripts), operands);
        case 2: return pairwise_einsum(plan_for<3>(subscripts), operands);
        default: break;
    }

    // 3 or more operands: contract left to right, each step validated and cached as its own pairwise plan
    const std::vector<std::string>& chain = chain_for(subscripts, operands.size());
    Tensor output = *operands[0];
    for (size_t next = 1; next < operands.size(); ++next) {
        const std::array<const Tensor*, 2> pair{ &output, operands[next] };
        output = pairwise_einsum(plan_for<3>(chain[next - 1]), pair);
    }
    return output;
}
