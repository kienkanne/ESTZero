#include <string>
#include <vector>
#include <ranges>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

#include "einsum.h"

struct CharInfo {
    // Maps char to column index, this is crucial for building
    // the offset table (see below) for the einsum operation
    size_t col; 
    int dim_size;
};

struct ParsedSubscripts {
    std::vector<std::string> terms;
    std::unordered_map<char, CharInfo> char_to_dim;
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

    std::vector<std::string> terms = split(lhs, ',');
    for (const auto& term : terms) {
        if (term.empty()) {
            throw std::invalid_argument("Subscripts contain empty terms: " + raw_subscripts);
        }
    }
    // The last term is the rhs
    terms.push_back(rhs);

    // Gate 3: Operand count match check
    if (terms.size()-1 != operands.size()) {
        throw std::invalid_argument("Provided " + std::to_string(operands.size()) + 
                                    " operands, but subscript expects " + std::to_string(terms.size()-1) + " terms.");
    }

    // Gate 4: Output index validation (RHS indices must be a subset of LHS)
    std::unordered_set<char> lhs_chars;
    for (size_t term_index = 0; term_index < terms.size() - 1; ++term_index) {
        const auto& term = terms[term_index];
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
    for (size_t i = 0; i < terms.size()-1; ++i) {
        if (operands[i].ndim() != static_cast<int>(terms[i].length())) {
            throw std::invalid_argument("Operand " + std::to_string(i) + " has " + 
                                        std::to_string(operands[i].ndim()) + " dimensions, but subscript '" + 
                                        terms[i] + "' expects " + std::to_string(terms[i].length()) + ".");
        }
    }

    // Gate 6: Dimension size consistency across shared letters
    // Map from character to its corresponding dimension size
    std::unordered_map<char, CharInfo> char_to_dim;
    size_t next_col = 0;
    for (size_t i = 0; i < terms.size()-1; ++i) {
        const auto& term = terms[i];
        const auto& shape = operands[i].shape;
        for (size_t a = 0; a < term.length(); ++a) {
            char c = term[a];
            int size = shape[a];
            if (char_to_dim.count(c)) {
                if (char_to_dim[c].dim_size != size) {
                    throw std::invalid_argument(std::string("Dimension mismatch for index '") + c + 
                                                "': expected size " + std::to_string(char_to_dim[c].dim_size) + 
                                                ", but got " + std::to_string(size) + ".");
                }
            } else {
                char_to_dim[c] = CharInfo{next_col++, size};
            }
        }
    }

    return {terms, char_to_dim};
}


// Helper function to increment multi-dimensional indices and update offsets
void move_idx(
    std::vector<int>& indices,
    const std::vector<int>& dimensions,
    size_t& result_offset,
    const std::vector<size_t>& result_strides,
    std::vector<size_t>& operand_offsets,
    const std::vector<std::vector<size_t>>& operand_strides) {

    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        if (indices[i] < dimensions[i]-1) {
            indices[i]++;

            result_offset += result_strides[i];
            for (size_t a = 0; a < operand_offsets.size(); ++a) {
                operand_offsets[a] += operand_strides[a][i];
            }
            return;

        } else {
            indices[i] = 0;
            
            result_offset -= static_cast<size_t>(dimensions[i] - 1) * result_strides[i];
            for (size_t a = 0; a < operand_offsets.size(); ++a) {
                operand_offsets[a] -= static_cast<size_t>(dimensions[i] - 1) * operand_strides[a][i];
            }
        }
    }
}

Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands) {
    auto [terms, char_to_dim] = parse_subscripts(subscripts, operands);

    std::vector<int> result_shape;
    for (const char& c : terms.back()) {
        result_shape.push_back(char_to_dim[c].dim_size);
    }

    Tensor result(result_shape);
    std::vector<size_t> result_strides(char_to_dim.size(), 0);
    size_t result_stride = 1;
    for (int axis = static_cast<int>(terms.back().size()) - 1; axis >= 0; --axis) {
        const auto& info = char_to_dim.at(terms.back()[axis]);
        result_strides[info.col] = result_stride;
        result_stride *= static_cast<size_t>(info.dim_size);
    }

    std::vector<std::vector<size_t>> operands_strides;
    for (size_t operand_index = 0; operand_index < operands.size(); ++operand_index) {
        std::vector<size_t> strides(char_to_dim.size(), 0);
        size_t stride = 1;

        // Previous bug: Cols automatically inferred from char order enumerating the term,
        // but this breaks when the same char appears in multiple terms, as it should map
        // to the same column in the result.
        for (int axis = static_cast<int>(terms[operand_index].size()) - 1; axis >= 0; --axis) {
            const auto& info = char_to_dim.at(terms[operand_index][axis]);
            strides[info.col] = stride;
            stride *= static_cast<size_t>(operands[operand_index].shape[axis]);
        }
        operands_strides.push_back(std::move(strides));
    }

    std::vector<int> dim_sizes(char_to_dim.size(), 0);
    for (const auto& [label, info] : char_to_dim) {
        dim_sizes[info.col] = info.dim_size;
    }

    std::vector<int> indices(char_to_dim.size(), 0);
    size_t total_combinations = 1;
    for (const auto& [label, info] : char_to_dim) {
        total_combinations *= static_cast<size_t>(info.dim_size);
    }

    // Initialize offsets
    size_t result_offset = 0;
    std::vector<size_t> operand_offsets(operands.size(), 0);

    for (size_t combination = 0; combination < total_combinations; ++combination) {
        double product = 1.0;

        for (size_t operand_index = 0; operand_index < operands.size(); ++operand_index) {
            product *= operands[operand_index].flat_index(operand_offsets[operand_index]);
        }

        result.flat_index(result_offset) += product;
        move_idx(
            indices,
            dim_sizes,
            result_offset,
            result_strides,
            operand_offsets,
            operands_strides);
    }

    return result;
}
