# Gradient Conflict Error-Bar Probe

- runs: 5 seeds
- error bar: per-layer STD across seeds

## Macro Means

| metric | mean over layers | mean error bar |
| --- | ---: | ---: |
| gc | 0.407936 | 0.066446 |
| gc_after | 0.403415 | 0.065497 |
| gc_bidir | 0.349809 | 0.068721 |

## Runs

| seed | path | GC macro | GC after macro | GC bidir macro |
| --- | --- | ---: | ---: | ---: |
| 3407 | `logs/offline_probe_bestever03nofreeze_step1400_bs72_long` | 0.343471 | 0.339509 | 0.264945 |
| 3408 | `logs/offline_probe_bestever03nofreeze_step1400_bs72_seed3408` | 0.496427 | 0.490284 | 0.432845 |
| 3409 | `logs/offline_probe_bestever03nofreeze_step1400_bs72_seed3409` | 0.382908 | 0.378247 | 0.334240 |
| 3410 | `logs/offline_probe_bestever03nofreeze_step1400_bs72_seed3410` | 0.409770 | 0.405650 | 0.354039 |
| 3411 | `logs/offline_probe_bestever03nofreeze_step1400_bs72_seed3411` | 0.407106 | 0.403385 | 0.362974 |

## Files

- `gc_plot_errorbar.png`
- `gc_plot_errorbar.pdf`
- `gc_plot_iclr_errorbar.png`
- `gc_plot_iclr_errorbar.pdf`
- `gc_layerwise_errorbar.tsv`
- `gc_errorbar_summary.json`
