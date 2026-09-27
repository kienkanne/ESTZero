#include <vector>
#include "tensor.h"
#include "timer.h"
#include "est_gemm.h"
#include "buffers.h"
#include "dispatch.h"
#include "triples_legacy.h"


// This is a direct line-by-line port of `t3c_ijk_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
Tensor t3c_ijk_spin(const Tensor& t2, VikingsBufferSpin& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    const int v2 = v * v;
    const size_t v3 = static_cast<size_t>(v2) * v;

    Tensor& t3 = vb.t3;
    t3.fill(0);

    // t2[idx,idx,:,:] (v,v) is already contiguous because it's the first index being sliced (largest stride), so no
    // packing is needed (just pass in the data_ptr that is already offsetted). Wvvvo[:,:,:,idx] is a strided slice (last
    // index) and it non-contiguous, so use .copy() to pack it for GEMM.
    //
    // The GEMM notation is t2(a,d) * W(bc,d).T = (abc)
    // The sizes and leading dimension are passed into cblas, so the shape can be recognized (v^2 is bc). The transpose then
    // guarantees the output shape is abc, no permutation needed.
    Tensor abc, abc2;

    if (gs == GEMMScheme::CUBLAS_DGEMM) {
        // t2 and W are already on device (vb.d_t2 uploaded once in VikingsBufferSpin's constructor, and vb.d_W* are
        // refreshed by to_device() calls in the loops. cublas_direct_dgemm directly uses them, and skipping all the
        // malloc/H2D/D2H the normal cublas_dgemm does. Only the two to_host() calls below finally leave the deivce.
        auto add_vvvo_term = [&](double sign, const double* d_t2_slice, const double* d_W, bool first) {
            cublas_direct_dgemm(v, v2, v, sign, first ? 0.0 : 1.0, d_t2_slice, d_W, vb.d_gemm_out,
                                 /*transA*/ false, /*transB*/ true);
        };
        add_vvvo_term(1.0,  vb.d_t2 + (i * o + j) * v2, vb.d_Wvvvo_k, true);  // "ad,bcd->abc" (a,b,c)
        add_vvvo_term(-1.0, vb.d_t2 + (k * o + j) * v2, vb.d_Wvvvo_i, false); // "ad,bcd->abc" (a,b,c)
        add_vvvo_term(-1.0, vb.d_t2 + (i * o + k) * v2, vb.d_Wvvvo_j, false); // "ad,bcd->abc" (a,b,c)
        to_host(vb.gemm_out.data(), vb.d_gemm_out, v3);
        abc = Tensor({v, v, v}, vb.gemm_out);

        auto add_ovoo_term = [&](double sign, const double* d_t2_slice, const double* d_W, bool first) {
            cublas_direct_dgemm(v2, v, o, sign, first ? 0.0 : 1.0, d_t2_slice, d_W, vb.d_gemm_out,
                                 /*transA*/ true, /*transB*/ false);
        };
        add_ovoo_term(1.0,  vb.d_t2 + i * o * v2, vb.d_Wovoo_jk, true);  // "lab,lc->abc" (a,b,c)
        add_ovoo_term(-1.0, vb.d_t2 + j * o * v2, vb.d_Wovoo_ik, false); // "lab,lc->abc" (a,b,c)
        add_ovoo_term(-1.0, vb.d_t2 + k * o * v2, vb.d_Wovoo_ji, false); // "lab,lc->abc" (a,b,c)
        to_host(vb.gemm_out.data(), vb.d_gemm_out, v3);
        abc2 = Tensor({v, v, v}, vb.gemm_out);

    } else {
        auto add_vvvo_term = [&](double sign, const Tensor& t2_slice, const Tensor& W_pack, bool first) {
            const double* A = t2_slice.data_ptr();
            const double* B = W_pack.data_ptr();
            const double beta = first ? 0.0 : 1.0;
            call_gemm(gs, v, v2, v, A, B, vb.gemm_out.data(), sign, beta, /*transA*/ false, /*transB*/ true);
        };
        add_vvvo_term(1.0,  t2.select(0, i).select(0, j), vb.Wvvvo_k, true);  // "ad,bcd->abc" (a,b,c)
        add_vvvo_term(-1.0, t2.select(0, k).select(0, j), vb.Wvvvo_i, false); // "ad,bcd->abc" (a,b,c)
        add_vvvo_term(-1.0, t2.select(0, i).select(0, k), vb.Wvvvo_j, false); // "ad,bcd->abc" (a,b,c)
        abc = Tensor({v, v, v}, vb.gemm_out);

        // Very much similar idea here, but t2 is sliced once and W is sliced twice instead. The transpose trick is the
        // same to get the desired output shape (abc)
        auto add_ovoo_term = [&](double sign, const Tensor& t2_slice, const Tensor& W_pack, bool first) {
            const double* A = t2_slice.data_ptr();
            const double* B = W_pack.data_ptr();
            const double beta = first ? 0.0 : 1.0;
            call_gemm(gs, v2, v, o, A, B, vb.gemm_out.data(), sign, beta, /*transA*/ true, /*transB*/ false);
        };
        add_ovoo_term(1.0,  t2.select(0, i), vb.Wovoo_jk, true);  // "lab,lc->abc" (a,b,c)
        add_ovoo_term(-1.0, t2.select(0, j), vb.Wovoo_ik, false); // "lab,lc->abc" (a,b,c)
        add_ovoo_term(-1.0, t2.select(0, k), vb.Wovoo_ji, false); // "lab,lc->abc" (a,b,c)
        abc2 = Tensor({v, v, v}, vb.gemm_out);
    }

    t3.assign(abc);                   // + abc[a,b,c]
    t3 -= abc.swapaxes(0, 1).copy(); // - abc[b,a,c]
    t3 -= abc.swapaxes(0, 2).copy(); // - abc[c,b,a]

    t3 -= abc2;                               // - abc2[a,b,c]
    t3 += abc2.swapaxes(0, 2).copy();// + abc2[c,b,a]
    t3 += abc2.swapaxes(1, 2).copy();// + abc2[a,c,b]

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}

    return t3 / denom;
}

// This is a line-by-line direct port from `t_vikings_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples_spin(const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V) {
    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape[0]});
    for (int i = 0; i != FN.shape[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    auto vb  = VikingsBufferSpin(t1, t2, o, v);
    Tensor& x1 = vb.x1; Tensor& x2 = vb.x2; Tensor& t3 = vb.t3;

    // Non-copying views
    Tensor Wvvvo = DEprqs.slice(V, V, V, O);
    Tensor Wovoo = DEprqs.slice(O, V, O, O);

    Tensor Woovv = DEprqs.slice(O, O, V, V);
    Tensor Wvovv = DEprqs.slice(V, O, V, V);
    Tensor Wooov = DEprqs.slice(O, O, O, V);

    const int v2 = v * v;
    const int v3 = v2 * v;

    // .assign() writes values into the buffer's already allocated storage in place
    const bool gpu_direct = (gs == GEMMScheme::CUBLAS_DGEMM);

    for (int i = 0; i < o; ++i) {
        vb.Wvvvo_i.assign(Wvvvo.select(3, i));
        if (gpu_direct) to_device(vb.d_Wvvvo_i, vb.Wvvvo_i.data_ptr(), v3);

    for (int j = 0; j < o; ++j) {
        vb.Wvvvo_j.assign(Wvvvo.select(3, j));
        vb.Wovoo_ji.assign(Wovoo.select(2, j).select(2, i));

        if (gpu_direct) {
            to_device(vb.d_Wvvvo_j, vb.Wvvvo_j.data_ptr(), v3);
            to_device(vb.d_Wovoo_ji, vb.Wovoo_ji.data_ptr(), o*v);
        }

    for (int k = 0; k < o; ++k) {
        vb.Wvvvo_k.assign(Wvvvo.select(3, k));
        vb.Wovoo_jk.assign(Wovoo.select(2, j).select(2, k));
        vb.Wovoo_ik.assign(Wovoo.select(2, i).select(2, k));

        if (gpu_direct) {
            to_device(vb.d_Wvvvo_k, vb.Wvvvo_k.data_ptr(), v3);
            to_device(vb.d_Wovoo_jk, vb.Wovoo_jk.data_ptr(), o*v);
            to_device(vb.d_Wovoo_ik, vb.Wovoo_ik.data_ptr(), o*v);
        }

        timing::Scoped _t("  Build T3");

        t3.assign(t3c_ijk_spin_old(t2, vb, o, v, F_occ, F_virt, i, j, k));

        // t3 (a,b,c) is contiguous (freshly built), and the other operands are also contiguous (slicing only the
        // first indices from the left). The output is fed directly into x2 with the correct slicing, which is
        // also contiguous, so it's safe to call data_ptr().

        if (gpu_direct) {
            // t3 was just built on host, so it needs one H2D upload per (i,j,k) - well this could be optimized further
            // by just doing the arithmetic on device as well, not just GEMM (TODO).
            to_device(vb.d_t3, t3.data_ptr(), v3);

            Tensor Woovv_jk = Woovv.select(0, j).select(0, k).contiguous();
            to_device(vb.d_Woovv_jk, Woovv_jk.data_ptr(), v2);
            cublas_direct_dgemm(v, 1, v2, 0.25, 1.0, vb.d_t3, vb.d_Woovv_jk,
                                 vb.d_x1 + i * v, false, false);

            Tensor Fov_k = Fov.select(0, k);
            to_device(vb.d_Fov_k, Fov_k.data_ptr(), v);
            cublas_direct_dgemm(v2, 1, v, 0.25, 1.0, vb.d_t3, vb.d_Fov_k,
                                 vb.d_x2 + (i * o + j) * v2, false, false);

            Tensor Wvovv_k = Wvovv.select(1, k).contiguous();
            to_device(vb.d_Wvovv_k, Wvovv_k.data_ptr(), v3);
            cublas_direct_dgemm(v, v, v2, 0.25, 1.0, vb.d_t3, vb.d_Wvovv_k,
                                 vb.d_x2 + (i * o + j) * v2, false, true);

            Tensor Wooov_jk = Wooov.select(0, j).select(0, k).contiguous();
            to_device(vb.d_Wooov_jk, Wooov_jk.data_ptr(), o*v);
            cublas_direct_dgemm(o, v2, v, -0.25, 1.0, vb.d_Wooov_jk, vb.d_t3,
                                 vb.d_x2 + i * o * v2, false, true);

        } else {
            // x1(i,:) += 0.25 * t3 as (a,(bc)) times Woovv_jk (bc,).
            Tensor Woovv_jk = Woovv.select(0, j).select(0, k).contiguous();
            call_gemm(gs, v, 1, v2, t3.data_ptr(), Woovv_jk.data_ptr(),
                      x1.select(0, i).data_ptr(), 0.25, 1.0, false, false);

            // x2(i,j,:,:) += 0.25 * t3 as ((ab),c) times Fov_k (c,).
            Tensor Fov_k = Fov.select(0, k);
            call_gemm(gs, v2, 1, v, t3.data_ptr(), Fov_k.data_ptr(),
                      x2.select(0, i).select(0, j).data_ptr(), 0.25, 1.0, false, false);

            // x2(i,j,a,d) += 0.25 * t3 (a,(bc)) @ Wvovv_k (d,(bc))^T.
            Tensor Wvovv_k = Wvovv.select(1, k).contiguous();
            call_gemm(gs, v, v, v2, t3.data_ptr(), Wvovv_k.data_ptr(),
                      x2.select(0, i).select(0, j).data_ptr(), 0.25, 1.0, false, true);

            // x2(i,m,a,b) -= 0.25 * Wooov_jk (m,d) @ t3 ((ab),d)^T.
            Tensor Wooov_jk = Wooov.select(0, j).select(0, k).contiguous();
            call_gemm(gs, o, v2, v, Wooov_jk.data_ptr(), t3.data_ptr(),
                      x2.select(0, i).data_ptr(), -0.25, 1.0, false, true);
        }
    }}}

    if (gpu_direct) {
        to_host(x1.data_ptr(), vb.d_x1, x1.size());
        to_host(x2.data_ptr(), vb.d_x2, x2.size());
    }

    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3);

    double et = (t1 * x1).sum();
    et += 0.25 * (t2 * x2).sum();
    return et;
}
