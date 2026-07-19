# Aggregated results

- generated: 2026-07-19T20:26:08+00:00
- gpu_type: `NVIDIA RTX A6000` (single-type table per plan section 0.1)
- metric: `tta_val_acc`; CI = 95% Student-t

| config | experiment | n | seeds | tta_val_acc mean | std | 95% CI ± | wall mean (s) | wall 95% CI ± | total cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| configs/wp22_headtohead_routed.yaml | airbench_smoke | 1 | 1601-1601 | 0.9416 | - | - | 15.362 | - | 0.003 |
| sweep_out/wp22_exploratory_fullrouted.yaml | airbench_smoke | 100 | 0-99 | 0.94036 | 0.0013904 | 0.00027588 | 11.174 | 0.095304 | 0.3 |
| sweep_out/wp22_goscconst_025.yaml | airbench_smoke | 100 | 0-99 | 0.94016 | 0.0013991 | 0.00027761 | 11.27 | 0.10083 | 0.3 |
| sweep_out/wp22_goscconst_050.yaml | airbench_smoke | 100 | 0-99 | 0.94021 | 0.0013271 | 0.00026332 | 11.121 | 0.087781 | 0.3 |
| sweep_out/wp22_goscconst_075.yaml | airbench_smoke | 100 | 0-99 | 0.94015 | 0.0014462 | 0.00028696 | 11.098 | 0.087917 | 0.3 |
| sweep_out/wp22_headtohead_adamuon.yaml | airbench_smoke | 100 | 0-99 | 0.93199 | 0.0014743 | 0.00029253 | 10.32 | 0.084382 | 0.3 |
| sweep_out/wp22_headtohead_dynmuon.yaml | airbench_smoke | 100 | 0-99 | 0.93297 | 0.0016021 | 0.0003179 | 10.506 | 0.10539 | 0.3 |
| sweep_out/wp22_headtohead_muon.yaml | airbench_smoke | 100 | 0-99 | 0.94014 | 0.0013615 | 0.00027016 | 10.286 | 0.09211 | 0.3 |
| sweep_out/wp22_headtohead_normuon.yaml | airbench_smoke | 100 | 0-99 | 0.93729 | 0.0012939 | 0.00025674 | 10.815 | 0.087127 | 0.3 |
| sweep_out/wp22_headtohead_routed.yaml | airbench_smoke | 100 | 0-99 | 0.94025 | 0.0014808 | 0.00029381 | 11.314 | 0.095333 | 0.3 |
| sweep_out/wp22_null_muon_wd.yaml | airbench_smoke | 100 | 0-99 | 0.94013 | 0.0012277 | 0.00024361 | 10.267 | 0.085116 | 0.3 |
| sweep_out/wp22_null_routed_randomgating.yaml | airbench_smoke | 100 | 0-99 | 0.94006 | 0.0013222 | 0.00026235 | 11.133 | 0.085743 | 0.3 |
| sweep_out/wp22_tuneB_muon.yaml | airbench_smoke | 100 | 0-99 | 0.9403 | 0.001306 | 0.00025914 | 10.323 | 0.085311 | 0.3 |
| sweep_out/wp22_tuneB_routed.yaml | airbench_smoke | 100 | 0-99 | 0.9401 | 0.0013832 | 0.00027445 | 11.102 | 0.085574 | 0.3 |
