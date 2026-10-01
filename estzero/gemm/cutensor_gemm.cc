#include "tensor.h"
#include "est_gemm.h"

// Guarded by EST_HAVE_CUTENSOR (set in CMakeLists.txt, alongside the -lcutensor link) so a
// CPU only build doesn't have to load this in. Mirrors cublas_gemm.cc's toggle pattern.
#ifdef EST_HAVE_CUTENSOR

#include <cuda_runtime.h>
#include <cutensor.h>
#include <unordered_set>
#include <vector>

namespace {

// Similar to cublas_gemm.cc's get_handle().
cutensorHandle_t get_handle() {
    static cutensorHandle_t handle = [] {
        cutensorHandle_t h;
        cutensorCreate(&h);
        return h;
    }();
    return handle;
}

}  // namespace

bool try_cutensor(const std::vector<std::string>& terms_strings, Operands operands,
                   Tensor& result) {
    if (operands.size() != 2) return false;

    const std::string& A_str = terms_strings[0];
    const std::string& B_str = terms_strings[1];
    const std::string& C_str = terms_strings[2]; // caller's requested output order

    // Declines if there are repeated indices like try_gemm does (e.g. "piqi->pq"), 
    // fall back to the odometer walk.
    auto has_repeat = [](const std::string& s) {
        std::unordered_set<char> seen;
        for (char c : s) if (!seen.insert(c).second) return true;
        return false;
    };
    if (has_repeat(A_str) || has_repeat(B_str)) return false;

    // For simplicity, force the data to be contiguous first, providing row-major data to give cuTensor.
    // TODO: pass  non-contiguous Tensor::strides straight through, skipping this copy
    Tensor A = operands[0]->contiguous();
    Tensor B = operands[1]->contiguous();

    const uint32_t numModesA = static_cast<uint32_t>(A_str.size());
    const uint32_t numModesB = static_cast<uint32_t>(B_str.size());
    const uint32_t numModesC = static_cast<uint32_t>(C_str.size());

    std::vector<int32_t> modeA(A_str.begin(), A_str.end());
    std::vector<int32_t> modeB(B_str.begin(), B_str.end());
    std::vector<int32_t> modeC(C_str.begin(), C_str.end());

    // Extent is similar to shape
    std::vector<int64_t> extentA(A.shape().begin(), A.shape().end());
    std::vector<int64_t> extentB(B.shape().begin(), B.shape().end());

    // The extent can be built and requested directly from cuTensor without an extra step of permuting
    // with our own Tensor, which `try_gemm` had to do.
    std::vector<int64_t> extentC(numModesC);
    std::vector<Index> result_shape(numModesC);
    for (uint32_t i = 0; i < numModesC; ++i) {
        size_t posA = A_str.find(C_str[i]);
        Index dim = (posA != std::string::npos) ? A.shape()[posA] : B.shape()[B_str.find(C_str[i])];
        extentC[i] = dim;
        result_shape[i] = dim;
    }

    std::vector<int64_t> strideA(A.strides().begin(), A.strides().end());
    std::vector<int64_t> strideB(B.strides().begin(), B.strides().end());
    std::vector<int64_t> strideC(numModesC);

    { // Strides of C have to be computed manually, as the default by cuTensor is column major
        int64_t running = 1;
        for (int i = static_cast<int>(numModesC) - 1; i >= 0; --i) {
            strideC[i] = running;
            running *= extentC[i];
        }
    }

    cutensorHandle_t handle = get_handle();
    const uint32_t kAlignment = 128; // matches cuTensor example

    cutensorTensorDescriptor_t descA, descB, descC;
    cutensorCreateTensorDescriptor(handle, &descA, numModesA, extentA.data(), strideA.data(),
                                CUDA_R_64F, kAlignment);
    cutensorCreateTensorDescriptor(handle, &descB, numModesB, extentB.data(), strideB.data(),
                                CUDA_R_64F, kAlignment);
    cutensorCreateTensorDescriptor(handle, &descC, numModesC, extentC.data(), strideC.data(),
                                CUDA_R_64F, kAlignment);

    cutensorOperationDescriptor_t opDesc;
    cutensorCreateContraction(handle, &opDesc,
                            descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                            descB, modeB.data(), CUTENSOR_OP_IDENTITY,
                            descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                            descC, modeC.data(),
                            CUTENSOR_COMPUTE_DESC_64F);

    cutensorPlanPreference_t planPref;
    cutensorCreatePlanPreference(handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);

    uint64_t worksize = 0;
    cutensorEstimateWorkspaceSize(handle, opDesc, planPref, CUTENSOR_WORKSPACE_DEFAULT, &worksize);

    cutensorPlan_t plan;
    cutensorCreatePlan(handle, &plan, opDesc, planPref, worksize);

    const size_t sizeA = A.size();
    const size_t sizeB = B.size();
    size_t sizeC = 1;
    for (int64_t e : extentC) sizeC *= static_cast<size_t>(e);

    double *d_A, *d_B, *d_C;
    void* d_work = nullptr;
    cudaMalloc((void**)&d_A, sizeA * sizeof(double));
    cudaMalloc((void**)&d_B, sizeB * sizeof(double));
    cudaMalloc((void**)&d_C, sizeC * sizeof(double));
    if (worksize > 0) cudaMalloc(&d_work, worksize);

    cudaMemcpy(d_A, A.data_ptr(), sizeA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_ptr(), sizeB * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1.0, beta = 0.0;
    cutensorContract(handle, plan, &alpha, d_A, d_B, &beta, d_C, d_C, d_work, worksize, 0);

    std::vector<double> result_data(sizeC);
    cudaMemcpy(result_data.data(), d_C, sizeC * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (d_work) cudaFree(d_work);
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(planPref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descA);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descC);

    result = Tensor(result_shape, result_data);
    return true;
}

#else

#include <stdexcept>

bool try_cutensor(const std::vector<std::string>&, Operands, Tensor&) {
    throw std::runtime_error(
        "cutensor_gemm: this build was compiled without EST_HAVE_CUTENSOR -- the cuTENSOR block in "
        "CMakeLists.txt is commented out (clean CPU-only build). Un-comment it to use "
        "--gemm-scheme cutensor.");
}

#endif
