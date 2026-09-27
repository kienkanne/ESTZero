#include <vector>
#include "tensor.h"
#include "timer.h"
#include "est_gemm.h"
#include "buffers.h"
#include "dispatch.h"
#include "triples_legacy.h"

// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-adapted/RHF form, where the six
// permutations are written out explicitly. Wvvvo/Wovoo are raw Iprqs slices, unlike the UHF
// version (triples_spin.cc) which uses DEprqs.
// This is a direct line-by-line port of `t3c_ijk` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
Tensor t3c_ijk_spat(const Tensor& t2, VikingsBufferSpat& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    const int v2 = v * v;
    const size_t v3 = static_cast<size_t>(v2) * v;

    Tensor& t3 = vb.t3;
    t3.fill(0);

    // Similar idea in UHF where t2 is already contiguous and W needs packing. However, we don't accumulate directly into
    // t3 because the output shape varies in the RHF scheme, unlike uhf. For example, t2(c,e) * W(ba,e).T = (cba), not
    // abc, and this varies with all contractions. Therefore, a raw term must be permutated. They also can't share the same
    // vb.gemm_out for accumulation, so it is instead reused sequentially as scratch
    auto vvvo_term = [&](const Tensor& t2_slice, const Tensor& W_pack) {
        const double* A = t2_slice.data_ptr();
        const double* B = W_pack.data_ptr();
        call_gemm(gs, v, v2, v, A, B, vb.gemm_out.data(), 1.0, 0.0, /*transA*/ false, /*transB*/ true);
        return Tensor({v, v, v}, vb.gemm_out);
    };

    auto ovoo_term = [&](const Tensor& t2_slice, const Tensor& W_pack) {
        const double* A = t2_slice.data_ptr();
        const double* B = W_pack.data_ptr();
        call_gemm(gs, v2, v, o, A, B, vb.gemm_out.data(), 1.0, 0.0, /*transA*/ true, /*transB*/ false);
        return Tensor({v, v, v}, vb.gemm_out);
    };

    // Device-resident counterparts
    auto vvvo_term_gpu = [&](const double* d_t2_slice, const double* d_W) {
        cublas_direct_dgemm(v, v2, v, 1.0, 0.0, d_t2_slice, d_W, vb.d_gemm_out, false, true);
        to_host(vb.gemm_out.data(), vb.d_gemm_out, v3);
        return Tensor({v, v, v}, vb.gemm_out);
    };
    auto ovoo_term_gpu = [&](const double* d_t2_slice, const double* d_W) {
        cublas_direct_dgemm(v2, v, o, 1.0, 0.0, d_t2_slice, d_W, vb.d_gemm_out, true, false);
        to_host(vb.gemm_out.data(), vb.d_gemm_out, v3);
        return Tensor({v, v, v}, vb.gemm_out);
    };

    // Permutation logic to accumulate the raw output GEMM into t3. (bac) to (abc) would be (1, 0, 2)
    auto add_term = [&](const Tensor& raw, double sign, int pos_a, int pos_b, int pos_c) {
        t3 += sign * raw.permute({pos_a, pos_b, pos_c});
    };

    // Comment shows the old einsum string, then the raw (physical) axis order the GEMM actually produces
    if (gs == GEMMScheme::CUBLAS_DGEMM) {
        add_term(vvvo_term_gpu(vb.d_t2 + (k * o + j) * v2, vb.d_Wvvvo_i), 1.0, 2, 1, 0); // "bae,ce->abc" (c,b,a)
        add_term(vvvo_term_gpu(vb.d_t2 + (j * o + k) * v2, vb.d_Wvvvo_i), 1.0, 2, 0, 1); // "cae,be->abc" (b,c,a)
        add_term(vvvo_term_gpu(vb.d_t2 + (j * o + i) * v2, vb.d_Wvvvo_k), 1.0, 1, 0, 2); // "ace,be->abc" (b,a,c)
        add_term(vvvo_term_gpu(vb.d_t2 + (i * o + j) * v2, vb.d_Wvvvo_k), 1.0, 0, 1, 2); // "bce,ae->abc" (a,b,c)
        add_term(vvvo_term_gpu(vb.d_t2 + (i * o + k) * v2, vb.d_Wvvvo_j), 1.0, 0, 2, 1); // "cbe,ae->abc" (a,c,b)
        add_term(vvvo_term_gpu(vb.d_t2 + (k * o + i) * v2, vb.d_Wvvvo_j), 1.0, 1, 2, 0); // "abe,ce->abc" (c,a,b)

        add_term(ovoo_term_gpu(vb.d_t2 + i * o * v2, vb.d_Wovoo_jk), -1.0, 0, 1, 2); // "mc,mab->abc" (a,b,c)
        add_term(ovoo_term_gpu(vb.d_t2 + i * o * v2, vb.d_Wovoo_kj), -1.0, 0, 2, 1); // "mb,mac->abc" (a,c,b)
        add_term(ovoo_term_gpu(vb.d_t2 + k * o * v2, vb.d_Wovoo_ij), -1.0, 1, 2, 0); // "mb,mca->abc" (c,a,b)
        add_term(ovoo_term_gpu(vb.d_t2 + k * o * v2, vb.d_Wovoo_ji), -1.0, 2, 1, 0); // "ma,mcb->abc" (c,b,a)
        add_term(ovoo_term_gpu(vb.d_t2 + j * o * v2, vb.d_Wovoo_ki), -1.0, 2, 0, 1); // "ma,mbc->abc" (b,c,a)
        add_term(ovoo_term_gpu(vb.d_t2 + j * o * v2, vb.d_Wovoo_ik), -1.0, 1, 0, 2); // "mc,mba->abc" (b,a,c)

    } else {
        add_term(vvvo_term(t2.select(0, k).select(0, j), vb.Wvvvo_i), 1.0, 2, 1, 0); // "bae,ce->abc" (c,b,a)
        add_term(vvvo_term(t2.select(0, j).select(0, k), vb.Wvvvo_i), 1.0, 2, 0, 1); // "cae,be->abc" (b,c,a)
        add_term(vvvo_term(t2.select(0, j).select(0, i), vb.Wvvvo_k), 1.0, 1, 0, 2); // "ace,be->abc" (b,a,c)
        add_term(vvvo_term(t2.select(0, i).select(0, j), vb.Wvvvo_k), 1.0, 0, 1, 2); // "bce,ae->abc" (a,b,c)
        add_term(vvvo_term(t2.select(0, i).select(0, k), vb.Wvvvo_j), 1.0, 0, 2, 1); // "cbe,ae->abc" (a,c,b)
        add_term(vvvo_term(t2.select(0, k).select(0, i), vb.Wvvvo_j), 1.0, 1, 2, 0); // "abe,ce->abc" (c,a,b)

        add_term(ovoo_term(t2.select(0, i), vb.Wovoo_jk), -1.0, 0, 1, 2); // "mc,mab->abc" (a,b,c)
        add_term(ovoo_term(t2.select(0, i), vb.Wovoo_kj), -1.0, 0, 2, 1); // "mb,mac->abc" (a,c,b)
        add_term(ovoo_term(t2.select(0, k), vb.Wovoo_ij), -1.0, 1, 2, 0); // "mb,mca->abc" (c,a,b)
        add_term(ovoo_term(t2.select(0, k), vb.Wovoo_ji), -1.0, 2, 1, 0); // "ma,mcb->abc" (c,b,a)
        add_term(ovoo_term(t2.select(0, j), vb.Wovoo_ki), -1.0, 2, 0, 1); // "ma,mbc->abc" (b,c,a)
        add_term(ovoo_term(t2.select(0, j), vb.Wovoo_ik), -1.0, 1, 0, 2); // "mc,mba->abc" (b,a,c)
    }

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}

// Loops over occupied (i,j,k), builds one (v,v,v) connected-T3 batch (t3c_ijk_spat above) at a
// time, and contracts it into the x1/x2 accumulators. RHF/spatial-orbital formulation, needs both
// Iprqs and DEprqs (unlike triples_spin.cc's UHF formulation, which only needs DEprqs).
// This is a line-by-line direct port from `t_vikings` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples_spat(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V) {
    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape[0]});
    for (int i = 0; i != FN.shape[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    auto vb  = VikingsBufferSpat(t1, t2, o, v);
    Tensor& x1 = vb.x1; Tensor& x2 = vb.x2; Tensor& t3 = vb.t3;
    Tensor& t3_diff = vb.t3_diff; Tensor& t3_comb1 = vb.t3_comb1;
    const int v2 = v * v;

    // Non-copying views
    Tensor Wvvvo = Iprqs.slice(V, V, V, O);
    Tensor Wovoo = Iprqs.slice(O, V, O, O);

    Tensor Wvovv = Iprqs.slice(V, O, V, V);
    Tensor Wooov = Iprqs.slice(O, O, O, V);
    Tensor DEoovv = DEprqs.slice(O, O, V, V);

    const int v3 = v2 * v;
    const bool gpu_direct = (gs == GEMMScheme::CUBLAS_DGEMM);

    for (int i = 0; i < o; ++i) {
        vb.Wvvvo_i.assign(Wvvvo.select(3, i));
        if (gpu_direct) to_device(vb.d_Wvvvo_i, vb.Wvvvo_i.data_ptr(), v3);

    for (int j = 0; j < o; ++j) {
        vb.Wvvvo_j.assign(Wvvvo.select(3, j));
        vb.Wovoo_ij.assign(Wovoo.select(2, i).select(2, j));
        vb.Wovoo_ji.assign(Wovoo.select(2, j).select(2, i));

        if (gpu_direct) {
            to_device(vb.d_Wvvvo_j, vb.Wvvvo_j.data_ptr(), v3);
            to_device(vb.d_Wovoo_ij, vb.Wovoo_ij.data_ptr(), o*v);
            to_device(vb.d_Wovoo_ji, vb.Wovoo_ji.data_ptr(), o*v);
        }

    for (int k = 0; k < o; ++k) {
        vb.Wvvvo_k.assign(Wvvvo.select(3, k));
        vb.Wovoo_jk.assign(Wovoo.select(2, j).select(2, k));
        vb.Wovoo_kj.assign(Wovoo.select(2, k).select(2, j));
        vb.Wovoo_ik.assign(Wovoo.select(2, i).select(2, k));
        vb.Wovoo_ki.assign(Wovoo.select(2, k).select(2, i));

        if (gpu_direct) {
            to_device(vb.d_Wvvvo_k, vb.Wvvvo_k.data_ptr(), v3);
            to_device(vb.d_Wovoo_jk, vb.Wovoo_jk.data_ptr(), o*v);
            to_device(vb.d_Wovoo_kj, vb.Wovoo_kj.data_ptr(), o*v);
            to_device(vb.d_Wovoo_ik, vb.Wovoo_ik.data_ptr(), o*v);
            to_device(vb.d_Wovoo_ki, vb.Wovoo_ki.data_ptr(), o*v);
        }

        timing::Scoped _t("  Build T3");

        t3.assign(t3c_ijk_spat_old(t2, vb, o, v, F_occ, F_virt, i, j, k));
        t3_diff.assign(t3 - t3.swapaxes(0, 2));
        t3_comb1.assign(2.0 * t3 - t3.swapaxes(1, 2) - t3.swapaxes(0, 2));

        if (gpu_direct) {
            // Same idea as UHF's x1/x2 GPU branch. t3_diff/t3_comb1 were just built on host, one H2D upload each iteration,
            // Then four W slices to device. d_x1/d_x2 accumulate in place
            to_device(vb.d_t3_diff, t3_diff.data_ptr(), v3);
            to_device(vb.d_t3_comb1, t3_comb1.data_ptr(), v3);

            Tensor DEoovv_jk = DEoovv.select(0, j).select(0, k).contiguous();
            to_device(vb.d_DEoovv_jk, DEoovv_jk.data_ptr(), v2);
            cublas_direct_dgemm(v, 1, v2, 1.0, 1.0, vb.d_t3_diff, vb.d_DEoovv_jk,
                                 vb.d_x1 + i * v, false, false);

            Tensor Wvovv_k = Wvovv.select(1, k).contiguous();
            to_device(vb.d_Wvovv_k, Wvovv_k.data_ptr(), v3);
            cublas_direct_dgemm(v, v, v2, 1.0, 1.0, vb.d_t3_comb1, vb.d_Wvovv_k,
                                 vb.d_x2 + (i * o + j) * v2, false, true);

            Tensor Wooov_jk = Wooov.select(0, j).select(0, k).contiguous();
            to_device(vb.d_Wooov_jk, Wooov_jk.data_ptr(), o*v);
            cublas_direct_dgemm(o, v2, v, -1.0, 1.0, vb.d_Wooov_jk, vb.d_t3_comb1,
                                 vb.d_x2 + i * o * v2, false, true);

            Tensor Fov_k = Fov.select(0, k);
            to_device(vb.d_Fov_k, Fov_k.data_ptr(), v);
            cublas_direct_dgemm(v2, 1, v, 1.0, 1.0, vb.d_t3_diff, vb.d_Fov_k,
                                 vb.d_x2 + (i * o + j) * v2, false, false);

        } else {
            // Same GEMM-shape reasoning as the UHF branch above, just no 0.25 factor here.
            Tensor DEoovv_jk = DEoovv.select(0, j).select(0, k).contiguous();
            call_gemm(gs, v, 1, v2, t3_diff.data_ptr(), DEoovv_jk.data_ptr(),
                      x1.select(0, i).data_ptr(), 1.0, 1.0, false, false);

            Tensor Wvovv_k = Wvovv.select(1, k).contiguous();
            call_gemm(gs, v, v, v2, t3_comb1.data_ptr(), Wvovv_k.data_ptr(),
                      x2.select(0, i).select(0, j).data_ptr(), 1.0, 1.0, false, true);

            Tensor Wooov_jk = Wooov.select(0, j).select(0, k).contiguous();
            call_gemm(gs, o, v2, v, Wooov_jk.data_ptr(), t3_comb1.data_ptr(),
                      x2.select(0, i).data_ptr(), -1.0, 1.0, false, true);

            Tensor Fov_k = Fov.select(0, k);
            call_gemm(gs, v2, 1, v, t3_diff.data_ptr(), Fov_k.data_ptr(),
                      x2.select(0, i).select(0, j).data_ptr(), 1.0, 1.0, false, false);
        }
    }}}

    if (gpu_direct) {
        to_host(x1.data_ptr(), vb.d_x1, x1.size());
        to_host(x2.data_ptr(), vb.d_x2, x2.size());
    }

    double et = 2.0 * (t1 * x1).sum();
    Tensor t2_comb = 4.0 * t2 - 2.0 * t2.swapaxes(2, 3);
    et += (t2_comb * x2).sum();
    return et;
}
