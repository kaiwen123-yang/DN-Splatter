# VR Room historical implementation baseline

Run status **complete**.
Recorded checkpoint step 29999; observed split counts {'train': 376, 'within': 42, 'with': 186}.
The historical loss places the flattening term inside the normal weight. Table 4 gives one VR Room row; the official implementation reports two trajectories.

| Trajectory | PSNR (dB) | Difference vs 26.8 dB | SSIM | Difference vs 0.879 |
|---|---:|---:|---:|---:|
| within | 26.9720 | +0.1720 | 0.88108 | +0.00208 |
| with | 23.7448 | -3.0552 | 0.82667 | -0.05233 |

Visibility-culled F-score at 5 cm was 86.863%, a difference of -0.237 percentage points from Table 4.
The official Faro output preserves its shared batch accumulators. The audited file gives each registered frame equal weight within its own trajectory.
The previews contain fixed held-out views and two raycast views of the culled surface. Their pixel arrays and the mesh topology passed the recorded checks.
