#include <vector>
#include "tensor.h"
#include "einsum.h"
#include "timer.h"
#include "est_gemm.h"
#include "cc_triples.h"

// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-orbital/UHF form, called once per
// (i,j,k) by calc_pert_triples below.
// This is a direct line-by-line port of `t3c_ijk_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
//
// Legacy version wired through einsum. The route routed into live code is below where GEMM is called
// directly. The repeated einsum calls in PertT makes the overhead cost the dominate.
Tensor t3c_ijk_spin_old(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    Tensor abc = einsum("ad,bcd->abc", {t2.select(0, i).select(0, j), Wvvvo.select(3, k)});
    abc -= einsum("ad,bcd->abc", {t2.select(0, k).select(0, j), Wvvvo.select(3, i)});
    abc -= einsum("ad,bcd->abc", {t2.select(0, i).select(0, k), Wvvvo.select(3, j)});
    Tensor t3 = abc - abc.swapaxes(0, 1) - abc.swapaxes(0, 2);

    abc = einsum("lab,lc->abc", {t2.select(0, i), Wovoo.select(2, j).select(2, k)});
    abc -= einsum("lab,lc->abc", {t2.select(0, j), Wovoo.select(2, i).select(2, k)});
    abc -= einsum("lab,lc->abc", {t2.select(0, k), Wovoo.select(2, j).select(2, i)});
    t3 -= (abc - abc.swapaxes(0, 2) - abc.swapaxes(1, 2));

    const int n_virt = t3.shape[0];
    Tensor denom({n_virt, n_virt, n_virt});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < n_virt; ++a) {
    for (int b = 0; b < n_virt; ++b) {
    for (int c = 0; c < n_virt; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}


// Directly call GEMM for contractions, bypassing einsum.
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


// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-adapted/RHF form, where the six permutations 
// are written out explicitly. Wvvvo/Wovoo are raw Iprqs slices, unlike the UHF version which uses DEprqs
// This is a direct line-by-line port of `t3c_ijk` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
Tensor t3c_ijk_spat_old(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    Tensor t3 = einsum("bae,ce->abc", {Wvvvo.select(3, i), t2.select(0, k).select(0, j)});
    t3 += einsum("cae,be->abc", {Wvvvo.select(3, i), t2.select(0, j).select(0, k)});
    t3 += einsum("ace,be->abc", {Wvvvo.select(3, k), t2.select(0, j).select(0, i)});
    t3 += einsum("bce,ae->abc", {Wvvvo.select(3, k), t2.select(0, i).select(0, j)});
    t3 += einsum("cbe,ae->abc", {Wvvvo.select(3, j), t2.select(0, i).select(0, k)});
    t3 += einsum("abe,ce->abc", {Wvvvo.select(3, j), t2.select(0, k).select(0, i)});

    t3 -= einsum("mc,mab->abc", {Wovoo.select(2, j).select(2, k), t2.select(0, i)});
    t3 -= einsum("mb,mac->abc", {Wovoo.select(2, k).select(2, j), t2.select(0, i)});
    t3 -= einsum("mb,mca->abc", {Wovoo.select(2, i).select(2, j), t2.select(0, k)});
    t3 -= einsum("ma,mcb->abc", {Wovoo.select(2, j).select(2, i), t2.select(0, k)});
    t3 -= einsum("ma,mbc->abc", {Wovoo.select(2, k).select(2, i), t2.select(0, j)});
    t3 -= einsum("mc,mba->abc", {Wovoo.select(2, i).select(2, k), t2.select(0, j)});

    const int n_virt = t3.shape[0];
    Tensor denom({n_virt, n_virt, n_virt});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < n_virt; ++a) {
    for (int b = 0; b < n_virt; ++b) {
    for (int c = 0; c < n_virt; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}


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

    // Device-resident counterparts: same GEMM shapes, but A comes from vb.d_t2 by pointer offset (same
    // flattening t2.select(0,idx)... uses on host) and B from the resident vb.d_W* buffer. Unlike UHF's
    // add_vvvo_term/add_ovoo_term, beta is always 0 here (each raw term is independent, no on-device
    // accumulation possible before the permute below), so vb.d_gemm_out/vb.gemm_out are just reused
    // sequentially as scratch -- one to_host() per term, same cadence as the CPU path's vb.gemm_out reuse.
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

// Loops over occupied (i,j,k), builds one (v,v,v) connected-T3 batch (t3c_ijk_so/t3c_ijk_spat above) at a time, 
// and contracts it into the x1/x2 accumulators, meaning that the full O(o^3 v^3) T3 is never created, unlike 
// calc_pert_triples_noddy below. RHF needs both Iprqs and DEprqs, while UHF only uses DEprqs.
// This is a line-by-line direct port from `t_vikings_so` / `t_vikings` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V, bool spin) {
    timing::Scoped _t("Perbutative Triples");

    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape[0]});
    for (int i = 0; i != FN.shape[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    if (spin) {
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

        t3 = t3c_ijk_spin(t2, vb, o, v, F_occ, F_virt, i, j, k);

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
    } else {
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

        t3 = t3c_ijk_spat(t2, vb, o, v, F_occ, F_virt, i, j, k);
        t3_diff  = t3 - t3.swapaxes(0, 2);
        t3_comb1 = 2.0 * t3 - t3.swapaxes(1, 2) - t3.swapaxes(0, 2);

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
}


// "Noddy" implementation for the perbutative triples in CCSD(T). This is a direct line-by-line
// port from Psi4Numpy's spin-orbital CCSD_T module, which cites Crawford's programming project #6.
// It creates the full O(o^3 v^3) T3 tensor. It's kept for A-B comparison against calc_pert_triples
// above, not called from cc_driver.h. UHF/spin-orbital only. References:
//
// https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306
// https://github.com/psi4/psi4numpy/blob/master/Coupled-Cluster/Spin_Orbitals/CCSD/CCSD_T.py
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V) {
    timing::Scoped _t("Perbutative Triples");

    // Disconnected triples
    Tensor tmp = einsum("ia,jkbc->ijkabc", {t1, Iprqs.slice(O, O, V, V)});
    Tensor t3d = tmp.copy();

    t3d -= tmp.swapaxes(3, 4); // ijkbac
    t3d -= tmp.swapaxes(3, 5); // ijkcba
    t3d -= tmp.swapaxes(0, 1); // jikabc
    t3d -= tmp.swapaxes(0, 2); // kjiabc

    t3d += tmp.swapaxes(0, 1).swapaxes(3, 4); // jikbac
    t3d += tmp.swapaxes(0, 1).swapaxes(3, 5); // jikcba
    t3d += tmp.swapaxes(0, 2).swapaxes(3, 4); // kjibac
    t3d += tmp.swapaxes(0, 2).swapaxes(3, 5); // kjicba

    // Connected triples
    tmp =  einsum("jkae,eibc->ijkabc", {t2, Iprqs.slice(V, O, V, V)});
    tmp -= einsum("imbc,majk->ijkabc", {t2, Iprqs.slice(O, V, O, O)});
    Tensor t3c = tmp.copy();

    t3c -= tmp.swapaxes(3, 4); // ijkbac
    t3c -= tmp.swapaxes(3, 5); // ijkcba
    t3c -= tmp.swapaxes(0, 1); // jikabc
    t3c -= tmp.swapaxes(0, 2); // kjiabc

    t3c += tmp.swapaxes(0, 1).swapaxes(3, 4); // jikbac
    t3c += tmp.swapaxes(0, 1).swapaxes(3, 5); // jikcba
    t3c += tmp.swapaxes(0, 2).swapaxes(3, 4); // kjibac
    t3c += tmp.swapaxes(0, 2).swapaxes(3, 5); // kjicba

    tmp = t3d + t3c;
    Tensor F_diag({FN.shape[0]});
    for (int i=0; i != FN.shape[0]; ++i) {
        F_diag(i) = FN(i, i);
    }
    Tensor F_occ = F_diag.slice(O); Tensor F_virt = F_diag.slice(V);

    const int n_virt = nbf * 2 - n_occ;
    Tensor Dijkabc({n_occ, n_occ, n_occ, n_virt, n_virt, n_virt});

    for (int i=0; i<n_occ; ++i)  for (int j=0; j<n_occ; ++j)  for (int k=0; k<n_occ; ++k)
    for (int a=0; a<n_virt; ++a) for (int b=0; b<n_virt; ++b) for (int c=0; c<n_virt; ++c) {
        Dijkabc(i,j,k,a,b,c) = F_occ(i) + F_occ(j) + F_occ(k)
                            - F_virt(a) - F_virt(b) - F_virt(c);
    }
    tmp /= Dijkabc;

    return (1.0/36) * (t3c * tmp).sum();
}
