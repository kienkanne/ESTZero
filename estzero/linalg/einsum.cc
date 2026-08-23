#include <string>
#include <vector>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

#include "einsum.h"
#include "odometer.h"

struct ParsedSubscripts {
    std::vector<std::string> terms_strings;
    std::unordered_map<char, int> char_to_dim;
    std::unordered_map<char, int> char_to_col;
};

// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    
    // For each token in the range created by split using delimiter
    for (const auto& token : str | std::views::split(delimiter)) {
        tokens.emplace_back(token.begin(), token.end());
    }
    return tokens;
}

// Helper to remove all spaces from a string
std::string remove_spaces(const std::string& str) {
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
}

ParsedSubscripts parse_subscripts(const std::string& raw_subscripts, const std::vector<Tensor>& operands) {
    std::string subscripts = remove_spaces(raw_subscripts);

    // Gate 1: Exactly one arrow check
    size_t arrow_pos = subscripts.find("->");
    if (arrow_pos == std::string::npos || subscripts.find("->", arrow_pos + 2) != std::string::npos) {
        throw std::invalid_argument("Subscripts must contain exactly one '->': " + raw_subscripts);
    }

    std::string lhs = subscripts.substr(0, arrow_pos);
    std::string rhs = subscripts.substr(arrow_pos + 2);

    // Gate 2: Empty term check (i.e., ",jk" or "ij,"). An empty RHS denotes a scalar result.
    if (lhs.empty()) {
        throw std::invalid_argument("Subscripts contain empty terms: " + raw_subscripts);
    }

    std::vector<std::string> terms_strings = split(lhs, ',');
    for (const auto& term : terms_strings) {
        if (term.empty()) {
            throw std::invalid_argument("Subscripts contain empty terms: " + raw_subscripts);
        }
    }
    // The last term is the rhs
    terms_strings.push_back(rhs);

    // Gate 3: Operand count match check
    if (terms_strings.size()-1 != operands.size()) {
        throw std::invalid_argument("Provided " + std::to_string(operands.size()) + 
                                    " operands, but subscript expects " + std::to_string(terms_strings.size()-1) + " terms.");
    }

    // Gate 4: Output index validation (RHS indices must be a subset of LHS)
    std::unordered_set<char> lhs_chars;
    for (size_t term_index = 0; term_index < terms_strings.size() - 1; ++term_index) {
        const auto& term = terms_strings[term_index];
        for (char c : term) lhs_chars.insert(c);
    }

    std::unordered_set<char> rhs_chars;
    for (char c : rhs) {
        if (rhs_chars.count(c)) {
            throw std::invalid_argument(std::string("Output subscript contains duplicate index '") + c + "'.");
        }
        rhs_chars.insert(c);
    }

    for (char c : rhs_chars) {
        if (!lhs_chars.count(c)) {
            throw std::invalid_argument(std::string("Output subscript contains index '") + c + "' not found in input terms.");
        }
    }

    // Gate 5: Operand dimensionality match check
    for (size_t i = 0; i < terms_strings.size()-1; ++i) {
        if (operands[i].ndim() != static_cast<int>(terms_strings[i].length())) {
            throw std::invalid_argument("Operand " + std::to_string(i) + " has " + 
                                        std::to_string(operands[i].ndim()) + " dimensions, but subscript '" + 
                                        terms_strings[i] + "' expects " + std::to_string(terms_strings[i].length()) + ".");
        }
    }

    // Gate 6: Dimension size consistency across shared letters
    // Map from character to its corresponding dimension size
    std::unordered_map<char, int> char_to_dim;
    std::unordered_map<char, int> char_to_col;
    size_t next_col = 0;
    for (size_t i = 0; i < terms_strings.size()-1; ++i) {
        const auto& term = terms_strings[i];
        const auto& shape = operands[i].shape;
        for (size_t a = 0; a < term.length(); ++a) {
            char c = term[a];
            int size = shape[a];
            if (char_to_dim.count(c)) {
                if (char_to_dim[c] != size) {
                    throw std::invalid_argument(std::string("Dimension mismatch for index '") + c + 
                                                "': expected size " + std::to_string(char_to_dim[c]) + 
                                                ", but got " + std::to_string(size) + ".");
                }
            } else {
                char_to_dim[c] = size;
                char_to_col[c] = next_col++;
            }
        }
    }

    return {terms_strings, char_to_dim, char_to_col};
}


bool try_gemm(const std::vector<std::string>& terms_strings, const std::vector<Tensor>& operands, Tensor& result) {
    if (operands.size() != 2) return false;

    // Add a guard on minimum tensor size to avoid doing large overhead for very small systems (go to odometer instead)
    for (const auto& t : operands) {
        if (t.size() < 1024) { return false; }
    }

    const std::string& A = terms_strings[0];
    const std::string& B = terms_strings[1];
    const std::string& C = terms_strings[2]; // requested output order, used for final permutation

    // Check for repeated indices in a single operand (i.e. "piqi")
    auto has_repeat = [](const std::string& s) {
        std::unordered_set<char> seen;
        for (char c : s) if (!seen.insert(c).second) return true;
        return false;
    };
    if (has_repeat(A) || has_repeat(B)) return false;

    std::unordered_set<char> a_set(A.begin(), A.end());
    std::unordered_set<char> b_set(B.begin(), B.end());
    std::unordered_set<char> c_set(C.begin(), C.end());

    // Batch index check (index in A is not in B and C, or same for B)
    // This doesn't occur in this codebase, but just decline for simplcity here, not doing GEMM-supported batch
    for (char c : A) {
        if (b_set.count(c) && c_set.count(c)) return false;              
        if (!c_set.count(c) && !b_set.count(c)) return false;
    }
    for (char c : B) {
        if (!c_set.count(c) && !a_set.count(c)) return false;
    }

    // Walk A once: kept-left axes (C have it) summed axes (C doesn't have it).
    // summed_labels must is the same for both A and B, so create one here and B reuses it
    std::vector<int> left_axes, summed_axes_A;
    std::string summed_labels;
    for (int i = 0; i < static_cast<int>(A.size()); ++i) {
        if (c_set.count(A[i])) left_axes.push_back(i);
        else { summed_axes_A.push_back(i); summed_labels += A[i]; }
    }

    std::vector<int> right_axes;
    for (int i = 0; i < static_cast<int>(B.size()); ++i) {
        if (c_set.count(B[i])) right_axes.push_back(i);
    }
    std::vector<int> summed_axes_B(summed_labels.size());
    for (size_t k = 0; k < summed_labels.size(); ++k) {
        summed_axes_B[k] = static_cast<int>(B.find(summed_labels[k])); // reuse
    }

    std::vector<int> permA = left_axes;
    permA.insert(permA.end(), summed_axes_A.begin(), summed_axes_A.end());
    std::vector<int> permB = summed_axes_B;
    permB.insert(permB.end(), right_axes.begin(), right_axes.end());

    int left_size = 1; for (int i : left_axes) left_size *= operands[0].shape[i];
    int summed_size = 1; for (int i : summed_axes_A) summed_size *= operands[0].shape[i]; // only takes A to compute, same if takes B
    int right_size = 1; for (int i : right_axes) right_size *= operands[1].shape[i];

    Tensor M = operands[0].permute(permA).reshape(left_size, summed_size);
    Tensor N = operands[1].permute(permB).reshape(summed_size, right_size);
    Matrix result_mat = M.as_matrix(left_size, summed_size) * N.as_matrix(summed_size, right_size);

    // combined_labels: left_axes + right_axes (what GEMM produced)
    std::string combined_labels;
    std::vector<int> combined_shape;
    for (int i : left_axes)  { combined_labels += A[i]; combined_shape.push_back(operands[0].shape[i]); }
    for (int i : right_axes) { combined_labels += B[i]; combined_shape.push_back(operands[1].shape[i]); }

    Tensor combined = mat_to_tensor(result_mat).reshape(combined_shape);

    // Reorder combined's axes into the caller's actual requested order
    std::vector<int> perm_out(C.size());
    for (size_t i = 0; i < C.size(); ++i) {
        perm_out[i] = static_cast<int>(combined_labels.find(C[i]));
    }
    result = combined.permute(perm_out);
    return true;
}

Tensor pairwise_einsum(const std::string& subscripts, const std::vector<Tensor>& operands) {
    auto [terms_strings, char_to_dim, char_to_col] = parse_subscripts(subscripts, operands);

    std::vector<int> result_shape;
    for (const char& c : terms_strings.back()) {
        result_shape.push_back(char_to_dim[c]);
    }
    Tensor result(result_shape); // constructor already computes correct natural strides

    // GEMM Route
    bool gemm = try_gemm(terms_strings, operands, result); // modify `Tensor result` directly
    if (gemm) { return result; }

    const int n_labels = static_cast<int>(char_to_dim.size());
    std::vector<int> dim_sizes(n_labels);
    for (const auto& [label, col] : char_to_col) {
        dim_sizes[col] = char_to_dim.at(label);
    }

    size_t total_combinations = 1;
    for (int d : dim_sizes) {
        total_combinations *= static_cast<size_t>(d);
    }

    // terms_strides[term_idx][col] = how much term `term_idx`'s flat offset moves per +1 of 
    // the global label assigned to column `col`. "term" here means every operand AND the result
    //
    // Each row is a basically a relabeled copy of each tensor's (operands) own `.strides`.
    // The loop below just maps them onto this canonical table that contain all indices/labels.
    // The table is filled with 0 by default, so any index not owned by an operand is just 0,
    // meaning that when that index moves the flat index of that operand just stays
    std::vector<std::vector<int>> terms_strides(
        terms_strings.size(), std::vector<int>(char_to_dim.size(), 0));

    // BUG FIX: Accumulate (not overwrite). a label can appear more than once in the same term. When that 
    // happens, every axis tied to the label must move together, so the flat-offset delta per +1 label is 
    // the sum of those axes strides.
    for (int operand_idx = 0; operand_idx < operands.size(); ++operand_idx) {
        const auto& term_string = terms_strings[operand_idx];

        for (int axis = 0; axis < term_string.size(); ++axis) {
            int col = char_to_col.at(term_string[axis]);
            terms_strides[operand_idx][col] += operands[operand_idx].strides[axis];
    }}

    // Same fix for result row. Technically, parse_subscripts' Gate 4 rejects a duplicate output index, 
    // so applying it here is for consistency only.
    for (int axis = 0; axis < terms_strings.back().size(); ++axis) {
        int col = char_to_col.at(terms_strings.back()[axis]);
        terms_strides.back()[col] += result.strides[axis];
    }

    // global_indices[col]: current value of the label at that column, over all label 
    // combinations (odometer() walks every one exactly once)
    std::vector<int> global_indices(n_labels, 0);
    std::vector<int> terms_flats(terms_strings.size(), 0);
    for (size_t c = 0; c < total_combinations; ++c) {
        double product = 1.0;

        for (size_t operand_idx = 0;
            operand_idx < operands.size();
            ++operand_idx) {
            product *= operands[operand_idx].flat_index(
                terms_flats[operand_idx]);
        }

        result.flat_index(terms_flats.back()) += product;

        odometer(global_indices, terms_flats, dim_sizes, terms_strides);
    }

    return result;
}


std::string multi_union(const std::vector<std::string>& terms_strings) {
    std::string combined;

    for (const auto& s : terms_strings) {
        combined += s;
    }
    std::sort(combined.begin(), combined.end());

    // Remove duplicate characters to get the unique set union
    auto last = std::unique(combined.begin(), combined.end());
    combined.erase(last, combined.end());

    return combined;
}

std::string str_intersect(std::string term1, std::string term2) {
    std::sort(term1.begin(), term1.end());
    std::sort(term2.begin(), term2.end());

    std::string result;
    // 2. Perform the set operation on the sorted copies
    std::set_intersection(term1.begin(), term1.end(), 
            term2.begin(), term2.end(), std::back_inserter(result));

    return result;
}

std::string get_output_string(int output_idx, std::vector<std::string> terms_strings, 
                            std::string term1_str, std::string term2_str){
    std::string term12_union = multi_union({term1_str, term2_str});

    std::vector<std::string> remaining_terms(terms_strings.begin()+output_idx, terms_strings.end());
    std::string remaining_union = multi_union(remaining_terms);

    return str_intersect(term12_union, remaining_union);  
}


Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands) {
    if (operands.size() == 1 || operands.size() == 2) {
        return pairwise_einsum(subscripts, operands);
    }

    // Handle cases with 3 or more operands only
    auto [terms_strings, char_to_dim, char_to_col] = parse_subscripts(subscripts, operands);

    Tensor output = operands[0];
    std::string output_str = terms_strings[0];

    for (size_t next_operand = 1; next_operand < operands.size(); ++next_operand) {
        std::string next_output_str;

        if (next_operand == operands.size() - 1) {
            next_output_str = terms_strings.back();   // exact requested order, not a sorted intersection
        } else {
            const size_t output_idx = next_operand + 1;
            next_output_str = get_output_string(output_idx, terms_strings, output_str, terms_strings[next_operand]);
        }

        const size_t output_idx = next_operand + 1;
        output = pairwise_einsum(
            output_str + "," + terms_strings[next_operand] + "->" + next_output_str,
            {output, operands[next_operand]});
        output_str = next_output_str;
    }
    return output;
}
