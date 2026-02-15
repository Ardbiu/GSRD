# GSRD Final Report

## Headline Numbers
- **best_map**: {'detector': 'mock', 'dataset': 'toy_det', 'granularity': 'coarse', 'mAP_mean': 0.8455315160364846}
- **max_worst_case_drop**: 0.0
- **risk_target_success_rate**: 1.0
- **mean_test_coverage**: 1.0

## Generated Figures (PNG + PDF)
- granularity_fig: `artifacts/local_bigger_v4/outputs/report/figures/granularity_sensitivity.png` and `artifacts/local_bigger_v4/outputs/report/figures/granularity_sensitivity.pdf`
- risk_fig: `artifacts/local_bigger_v4/outputs/report/figures/risk_vs_coverage.png` and `artifacts/local_bigger_v4/outputs/report/figures/risk_vs_coverage.pdf`
- domain_fig: `artifacts/local_bigger_v4/outputs/report/figures/id_to_ood_domain_breakdown.png` and `artifacts/local_bigger_v4/outputs/report/figures/id_to_ood_domain_breakdown.pdf`

## Generated Tables
- performance_csv: `artifacts/local_bigger_v4/outputs/report/tables/table_performance_by_granularity.csv`
- performance_tex: `artifacts/local_bigger_v4/outputs/report/tables/table_performance_by_granularity.tex`
- worst_drop_csv: `artifacts/local_bigger_v4/outputs/report/tables/table_worst_case_drop.csv`
- worst_drop_tex: `artifacts/local_bigger_v4/outputs/report/tables/table_worst_case_drop.tex`
- calib_csv: `artifacts/local_bigger_v4/outputs/report/tables/table_calibration_effectiveness.csv`
- calib_tex: `artifacts/local_bigger_v4/outputs/report/tables/table_calibration_effectiveness.tex`
- risk_summary_csv: `artifacts/local_bigger_v4/outputs/report/tables/table_risk_summary.csv`
- risk_summary_tex: `artifacts/local_bigger_v4/outputs/report/tables/table_risk_summary.tex`

## Experimental Notes
- Granularity summaries include vocabulary-list uncertainty (95% CI).
- Shift interaction includes uncertainty over ID-to-OOD drop estimates.
- Risk calibration reports target satisfaction rates and guarantee margins.