#pragma once

#include <torch/torch.h>

#ifdef METAL_KERNEL
// MPS implementation
torch::Tensor renderVoronoi(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                             double inv_scale_sq, int64_t width, int64_t height);
torch::Tensor renderVoronoiPadded(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                  double inv_scale_sq, int64_t width, int64_t height,
                                  int64_t cand_width, int64_t cand_height);

// Candidate initialization + update
std::tuple<torch::Tensor, torch::Tensor> initCandidates(int64_t width, int64_t height, int64_t siteCount, int64_t seed);
std::tuple<torch::Tensor, torch::Tensor> vptPass(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                                 double inv_scale_sq, int64_t width, int64_t height, int64_t jump,
                                                 int64_t radius_probes, double radius_scale, int64_t inject_count,
                                                 int64_t seed);

// Backward: dL/dimage -> dL/dsites
torch::Tensor renderVoronoiBackward(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                    torch::Tensor grad_output, double inv_scale_sq,
                                    int64_t width, int64_t height);

// Utility kernels
torch::Tensor clearBuffer(torch::Tensor buffer, int64_t count);
torch::Tensor packCandidateSites(torch::Tensor sites, torch::Tensor packed, int64_t site_count);
torch::Tensor updateCandidatesCompact(torch::Tensor cand0_in, torch::Tensor cand1_in,
                                      torch::Tensor cand0_out, torch::Tensor cand1_out,
                                      torch::Tensor packed_sites, torch::Tensor hilbert_order,
                                      torch::Tensor hilbert_pos, double inv_scale_sq,
                                      int64_t site_count, int64_t step, int64_t step_high,
                                      double radius_scale, int64_t radius_probes, int64_t inject_count,
                                      int64_t hilbert_probes, int64_t hilbert_window,
                                      int64_t cand_downscale, int64_t target_width, int64_t target_height,
                                      int64_t cand_width, int64_t cand_height);
torch::Tensor jfaClear(torch::Tensor cand0, torch::Tensor cand1, int64_t cand_width, int64_t cand_height);
torch::Tensor jfaSeed(torch::Tensor cand0, torch::Tensor sites, int64_t site_count,
                      int64_t cand_downscale, int64_t cand_width, int64_t cand_height);
torch::Tensor jfaFlood(torch::Tensor cand0_in, torch::Tensor cand0_out, torch::Tensor sites,
                       double inv_scale_sq, int64_t site_count, int64_t step_size,
                       int64_t cand_downscale, int64_t cand_width, int64_t cand_height);

torch::Tensor computeGradientsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor rendered, torch::Tensor mask,
                                    torch::Tensor sites,
                                    torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                                    torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                                    torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                                    torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                                    torch::Tensor removal_delta,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t compute_removal, double ssim_weight,
                                    int64_t cand_width, int64_t cand_height);

torch::Tensor tauDiffuse(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                         torch::Tensor grad_raw, torch::Tensor grad_in, torch::Tensor grad_out,
                         int64_t site_count, double lambda, int64_t cand_downscale,
                         int64_t cand_width, int64_t cand_height);

torch::Tensor adamUpdate(torch::Tensor sites, torch::Tensor adam,
                         torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                         torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                         torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                         torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                         double lr_pos, double lr_tau, double lr_radius, double lr_color,
                         double lr_dir, double lr_aniso, double beta1, double beta2, double eps,
                         int64_t t, int64_t width, int64_t height);

torch::Tensor computeSiteStatsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor mask, torch::Tensor sites,
                                    torch::Tensor mass, torch::Tensor energy,
                                    torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                                    torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t cand_width, int64_t cand_height);

torch::Tensor computeDensifyScorePairs(torch::Tensor sites, torch::Tensor mass, torch::Tensor energy,
                                       torch::Tensor pairs, int64_t site_count,
                                       double min_mass, double score_alpha, int64_t pair_count);
torch::Tensor computePruneScorePairs(torch::Tensor sites, torch::Tensor removal_delta, torch::Tensor pairs,
                                     int64_t site_count, double delta_norm, int64_t pair_count);
torch::Tensor radixSortPairs(torch::Tensor pairs, int64_t max_key_exclusive);
torch::Tensor writeSplitIndices(torch::Tensor pairs, torch::Tensor indices, int64_t num_to_split);
torch::Tensor splitSites(torch::Tensor sites, torch::Tensor adam, torch::Tensor split_indices,
                         torch::Tensor mass, torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                         torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                         int64_t current_site_count, int64_t num_to_split, torch::Tensor target);
torch::Tensor pruneSites(torch::Tensor sites, torch::Tensor indices, int64_t count);

torch::Tensor buildHilbertPairs(torch::Tensor sites, torch::Tensor pairs,
                                int64_t site_count, int64_t padded_count,
                                int64_t width, int64_t height, int64_t bits);
torch::Tensor writeHilbertOrder(torch::Tensor pairs, torch::Tensor order, torch::Tensor pos, int64_t site_count);
torch::Tensor initGradientWeighted(torch::Tensor sites, torch::Tensor target, torch::Tensor mask,
                                   int64_t site_count, double init_log_tau, double init_radius,
                                   double gradient_alpha);

// Metal training entry point
#endif

#ifdef CUDA_KERNEL
// CUDA implementation
torch::Tensor renderVoronoi(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                             double inv_scale_sq, int64_t width, int64_t height);
torch::Tensor renderVoronoiPadded(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                  double inv_scale_sq, int64_t width, int64_t height,
                                  int64_t cand_width, int64_t cand_height);
std::tuple<torch::Tensor, torch::Tensor> initCandidates(int64_t width, int64_t height, int64_t siteCount, int64_t seed);
std::tuple<torch::Tensor, torch::Tensor> vptPass(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                                 double inv_scale_sq, int64_t width, int64_t height, int64_t jump,
                                                 int64_t radius_probes, double radius_scale, int64_t inject_count,
                                                 int64_t seed);
torch::Tensor renderVoronoiBackward(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                                    torch::Tensor grad_output, double inv_scale_sq,
                                    int64_t width, int64_t height);
torch::Tensor clearBuffer(torch::Tensor buffer, int64_t count);
torch::Tensor packCandidateSites(torch::Tensor sites, torch::Tensor packed, int64_t site_count);
torch::Tensor updateCandidatesCompact(torch::Tensor cand0_in, torch::Tensor cand1_in,
                                      torch::Tensor cand0_out, torch::Tensor cand1_out,
                                      torch::Tensor packed_sites, torch::Tensor hilbert_order,
                                      torch::Tensor hilbert_pos, double inv_scale_sq,
                                      int64_t site_count, int64_t step, int64_t step_high,
                                      double radius_scale, int64_t radius_probes, int64_t inject_count,
                                      int64_t hilbert_probes, int64_t hilbert_window,
                                      int64_t cand_downscale, int64_t target_width, int64_t target_height,
                                      int64_t cand_width, int64_t cand_height);
torch::Tensor jfaClear(torch::Tensor cand0, torch::Tensor cand1, int64_t cand_width, int64_t cand_height);
torch::Tensor jfaSeed(torch::Tensor cand0, torch::Tensor sites, int64_t site_count,
                      int64_t cand_downscale, int64_t cand_width, int64_t cand_height);
torch::Tensor jfaFlood(torch::Tensor cand0_in, torch::Tensor cand0_out, torch::Tensor sites,
                       double inv_scale_sq, int64_t site_count, int64_t step_size,
                       int64_t cand_downscale, int64_t cand_width, int64_t cand_height);

torch::Tensor computeGradientsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor rendered, torch::Tensor mask,
                                    torch::Tensor sites,
                                    torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                                    torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                                    torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                                    torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                                    torch::Tensor removal_delta,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t compute_removal, double ssim_weight,
                                    int64_t cand_width, int64_t cand_height);

torch::Tensor tauDiffuse(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                         torch::Tensor grad_raw, torch::Tensor grad_in, torch::Tensor grad_out,
                         int64_t site_count, double lambda, int64_t cand_downscale,
                         int64_t cand_width, int64_t cand_height);

torch::Tensor adamUpdate(torch::Tensor sites, torch::Tensor adam,
                         torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                         torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                         torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                         torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                         double lr_pos, double lr_tau, double lr_radius, double lr_color,
                         double lr_dir, double lr_aniso, double beta1, double beta2, double eps,
                         int64_t t, int64_t width, int64_t height);

torch::Tensor computeSiteStatsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor mask, torch::Tensor sites,
                                    torch::Tensor mass, torch::Tensor energy,
                                    torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                                    torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t cand_width, int64_t cand_height);

torch::Tensor computeDensifyScorePairs(torch::Tensor sites, torch::Tensor mass, torch::Tensor energy,
                                       torch::Tensor pairs, int64_t site_count,
                                       double min_mass, double score_alpha, int64_t pair_count);
torch::Tensor computePruneScorePairs(torch::Tensor sites, torch::Tensor removal_delta, torch::Tensor pairs,
                                     int64_t site_count, double delta_norm, int64_t pair_count);
torch::Tensor radixSortPairs(torch::Tensor pairs, int64_t max_key_exclusive);
torch::Tensor writeSplitIndices(torch::Tensor pairs, torch::Tensor indices, int64_t num_to_split);
torch::Tensor splitSites(torch::Tensor sites, torch::Tensor adam, torch::Tensor split_indices,
                         torch::Tensor mass, torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                         torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                         int64_t current_site_count, int64_t num_to_split, torch::Tensor target);
torch::Tensor pruneSites(torch::Tensor sites, torch::Tensor indices, int64_t count);

torch::Tensor buildHilbertPairs(torch::Tensor sites, torch::Tensor pairs,
                                int64_t site_count, int64_t padded_count,
                                int64_t width, int64_t height, int64_t bits);
torch::Tensor writeHilbertOrder(torch::Tensor pairs, torch::Tensor order, torch::Tensor pos, int64_t site_count);
torch::Tensor initGradientWeighted(torch::Tensor sites, torch::Tensor target, torch::Tensor mask,
                                   int64_t site_count, double init_log_tau, double init_radius,
                                   double gradient_alpha);
#endif
