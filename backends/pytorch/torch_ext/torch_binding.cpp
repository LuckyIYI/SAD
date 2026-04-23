#include <torch/library.h>
#include <torch/extension.h>

#include "torch_binding.h"

TORCH_LIBRARY(sad, ops) {
  ops.def("render_sad(Tensor cand0, Tensor cand1, Tensor sites, float inv_scale_sq, int width, int height) -> Tensor");
  ops.def("render_sad_padded(Tensor cand0, Tensor cand1, Tensor sites, float inv_scale_sq, int width, int height, int cand_width, int cand_height) -> Tensor");
  ops.def("init_candidates(int width, int height, int site_count, int seed) -> (Tensor, Tensor)");
  ops.def("vpt_pass(Tensor cand0, Tensor cand1, Tensor sites, float inv_scale_sq, int width, int height, int jump, int radius_probes, float radius_scale, int inject_count, int seed) -> (Tensor, Tensor)");
  ops.def("render_sad_backward(Tensor cand0, Tensor cand1, Tensor sites, Tensor grad_output, float inv_scale_sq, int width, int height) -> Tensor");
  ops.def("clear_buffer(Tensor buffer, int count) -> Tensor");
  ops.def("pack_candidate_sites(Tensor sites, Tensor packed, int site_count) -> Tensor");
  ops.def("update_candidates_compact(Tensor cand0_in, Tensor cand1_in, Tensor cand0_out, Tensor cand1_out, Tensor packed_sites, Tensor hilbert_order, Tensor hilbert_pos, float inv_scale_sq, int site_count, int step, int step_high, float radius_scale, int radius_probes, int inject_count, int hilbert_probes, int hilbert_window, int cand_downscale, int target_width, int target_height, int cand_width, int cand_height) -> Tensor");
  ops.def("jfa_clear(Tensor cand0, Tensor cand1, int cand_width, int cand_height) -> Tensor");
  ops.def("jfa_seed(Tensor cand0, Tensor sites, int site_count, int cand_downscale, int cand_width, int cand_height) -> Tensor");
  ops.def("jfa_flood(Tensor cand0_in, Tensor cand0_out, Tensor sites, float inv_scale_sq, int site_count, int step_size, int cand_downscale, int cand_width, int cand_height) -> Tensor");
  ops.def("compute_gradients_tiled(Tensor cand0, Tensor cand1, Tensor target, Tensor rendered, Tensor mask, Tensor sites, Tensor grad_pos_x, Tensor grad_pos_y, Tensor grad_log_tau, Tensor grad_radius, Tensor grad_color_r, Tensor grad_color_g, Tensor grad_color_b, Tensor grad_dir_x, Tensor grad_dir_y, Tensor grad_log_aniso, Tensor removal_delta, float inv_scale_sq, int site_count, int compute_removal, float ssim_weight, int cand_width, int cand_height) -> Tensor");
  ops.def("tau_diffuse(Tensor cand0, Tensor cand1, Tensor sites, Tensor grad_raw, Tensor grad_in, Tensor grad_out, int site_count, float lambda, int cand_downscale, int cand_width, int cand_height) -> Tensor");
  ops.def("adam_update(Tensor sites, Tensor adam, Tensor grad_pos_x, Tensor grad_pos_y, Tensor grad_log_tau, Tensor grad_radius, Tensor grad_color_r, Tensor grad_color_g, Tensor grad_color_b, Tensor grad_dir_x, Tensor grad_dir_y, Tensor grad_log_aniso, float lr_pos, float lr_tau, float lr_radius, float lr_color, float lr_dir, float lr_aniso, float beta1, float beta2, float eps, int t, int width, int height) -> Tensor");
  ops.def("compute_site_stats_tiled(Tensor cand0, Tensor cand1, Tensor target, Tensor mask, Tensor sites, Tensor mass, Tensor energy, Tensor err_w, Tensor err_wx, Tensor err_wy, Tensor err_wxx, Tensor err_wxy, Tensor err_wyy, float inv_scale_sq, int site_count, int cand_width, int cand_height) -> Tensor");
  ops.def("compute_densify_score_pairs(Tensor sites, Tensor mass, Tensor energy, Tensor pairs, int site_count, float min_mass, float score_alpha, int pair_count) -> Tensor");
  ops.def("compute_prune_score_pairs(Tensor sites, Tensor removal_delta, Tensor pairs, int site_count, float delta_norm, int pair_count) -> Tensor");
  ops.def("radix_sort_pairs(Tensor pairs, int max_key_exclusive) -> Tensor");
  ops.def("write_split_indices(Tensor pairs, Tensor indices, int num_to_split) -> Tensor");
  ops.def("split_sites(Tensor sites, Tensor adam, Tensor split_indices, Tensor mass, Tensor err_w, Tensor err_wx, Tensor err_wy, Tensor err_wxx, Tensor err_wxy, Tensor err_wyy, int current_site_count, int num_to_split, Tensor target) -> Tensor");
  ops.def("prune_sites(Tensor sites, Tensor indices, int count) -> Tensor");
  ops.def("build_hilbert_pairs(Tensor sites, Tensor pairs, int site_count, int padded_count, int width, int height, int bits) -> Tensor");
  ops.def("write_hilbert_order(Tensor pairs, Tensor order, Tensor pos, int site_count) -> Tensor");
  ops.def("init_gradient_weighted(Tensor sites, Tensor target, Tensor mask, int site_count, float init_log_tau, float init_radius, float gradient_alpha) -> Tensor");

#if defined(METAL_KERNEL)
  ops.impl("render_sad", torch::kMPS, &renderVoronoi);
  ops.impl("render_sad_padded", torch::kMPS, &renderVoronoiPadded);
  // No tensor inputs -> register catchall so dispatcher can call it.
  ops.impl("init_candidates", &initCandidates);
  ops.impl("vpt_pass", torch::kMPS, &vptPass);
  ops.impl("render_sad_backward", torch::kMPS, &renderVoronoiBackward);
  ops.impl("clear_buffer", torch::kMPS, &clearBuffer);
  ops.impl("pack_candidate_sites", torch::kMPS, &packCandidateSites);
  ops.impl("update_candidates_compact", torch::kMPS, &updateCandidatesCompact);
  ops.impl("jfa_clear", torch::kMPS, &jfaClear);
  ops.impl("jfa_seed", torch::kMPS, &jfaSeed);
  ops.impl("jfa_flood", torch::kMPS, &jfaFlood);
  ops.impl("compute_gradients_tiled", torch::kMPS, &computeGradientsTiled);
  ops.impl("tau_diffuse", torch::kMPS, &tauDiffuse);
  ops.impl("adam_update", torch::kMPS, &adamUpdate);
  ops.impl("compute_site_stats_tiled", torch::kMPS, &computeSiteStatsTiled);
  ops.impl("compute_densify_score_pairs", torch::kMPS, &computeDensifyScorePairs);
  ops.impl("compute_prune_score_pairs", torch::kMPS, &computePruneScorePairs);
  ops.impl("radix_sort_pairs", torch::kMPS, &radixSortPairs);
  ops.impl("write_split_indices", torch::kMPS, &writeSplitIndices);
  ops.impl("split_sites", torch::kMPS, &splitSites);
  ops.impl("prune_sites", torch::kMPS, &pruneSites);
  ops.impl("build_hilbert_pairs", torch::kMPS, &buildHilbertPairs);
  ops.impl("write_hilbert_order", torch::kMPS, &writeHilbertOrder);
  ops.impl("init_gradient_weighted", torch::kMPS, &initGradientWeighted);
#endif

#if defined(CUDA_KERNEL)
  ops.impl("render_sad", torch::kCUDA, &renderVoronoi);
  ops.impl("render_sad_padded", torch::kCUDA, &renderVoronoiPadded);
  // No tensor inputs -> register catchall so dispatcher can call it.
  ops.impl("init_candidates", &initCandidates);
  ops.impl("vpt_pass", torch::kCUDA, &vptPass);
  ops.impl("render_sad_backward", torch::kCUDA, &renderVoronoiBackward);
  ops.impl("clear_buffer", torch::kCUDA, &clearBuffer);
  ops.impl("pack_candidate_sites", torch::kCUDA, &packCandidateSites);
  ops.impl("update_candidates_compact", torch::kCUDA, &updateCandidatesCompact);
  ops.impl("jfa_clear", torch::kCUDA, &jfaClear);
  ops.impl("jfa_seed", torch::kCUDA, &jfaSeed);
  ops.impl("jfa_flood", torch::kCUDA, &jfaFlood);
  ops.impl("compute_gradients_tiled", torch::kCUDA, &computeGradientsTiled);
  ops.impl("tau_diffuse", torch::kCUDA, &tauDiffuse);
  ops.impl("adam_update", torch::kCUDA, &adamUpdate);
  ops.impl("compute_site_stats_tiled", torch::kCUDA, &computeSiteStatsTiled);
  ops.impl("compute_densify_score_pairs", torch::kCUDA, &computeDensifyScorePairs);
  ops.impl("compute_prune_score_pairs", torch::kCUDA, &computePruneScorePairs);
  ops.impl("radix_sort_pairs", torch::kCUDA, &radixSortPairs);
  ops.impl("write_split_indices", torch::kCUDA, &writeSplitIndices);
  ops.impl("split_sites", torch::kCUDA, &splitSites);
  ops.impl("prune_sites", torch::kCUDA, &pruneSites);
  ops.impl("build_hilbert_pairs", torch::kCUDA, &buildHilbertPairs);
  ops.impl("write_hilbert_order", torch::kCUDA, &writeHilbertOrder);
  ops.impl("init_gradient_weighted", torch::kCUDA, &initGradientWeighted);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Module intentionally empty: ops are registered via TORCH_LIBRARY.
}
