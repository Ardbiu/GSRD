# GSRD Final Report

## Headline Numbers
- **best_map**: {'detector': 'mock', 'dataset': 'toy_det', 'granularity': 'fine', 'mAP_mean': 0.7283786642219345}
- **max_worst_case_drop**: 0.2791553000081275
- **risk_target_success_rate**: 0.8106995884773662
- **mean_test_coverage**: 0.8044872844094269

## Generated Figures (PNG + PDF)
- granularity_fig: `artifacts/local_bigger_v8/outputs/report/figures/granularity_sensitivity.png` and `artifacts/local_bigger_v8/outputs/report/figures/granularity_sensitivity.pdf`
- risk_fig: `artifacts/local_bigger_v8/outputs/report/figures/risk_vs_coverage.png` and `artifacts/local_bigger_v8/outputs/report/figures/risk_vs_coverage.pdf`
- domain_fig: `artifacts/local_bigger_v8/outputs/report/figures/id_to_ood_domain_breakdown.png` and `artifacts/local_bigger_v8/outputs/report/figures/id_to_ood_domain_breakdown.pdf`

## Generated Tables
- performance_csv: `artifacts/local_bigger_v8/outputs/report/tables/table_performance_by_granularity.csv`
- performance_tex: `artifacts/local_bigger_v8/outputs/report/tables/table_performance_by_granularity.tex`
- worst_drop_csv: `artifacts/local_bigger_v8/outputs/report/tables/table_worst_case_drop.csv`
- worst_drop_tex: `artifacts/local_bigger_v8/outputs/report/tables/table_worst_case_drop.tex`
- calib_csv: `artifacts/local_bigger_v8/outputs/report/tables/table_calibration_effectiveness.csv`
- calib_tex: `artifacts/local_bigger_v8/outputs/report/tables/table_calibration_effectiveness.tex`
- risk_summary_csv: `artifacts/local_bigger_v8/outputs/report/tables/table_risk_summary.csv`
- risk_summary_tex: `artifacts/local_bigger_v8/outputs/report/tables/table_risk_summary.tex`
- novelty_csv: `artifacts/local_bigger_v8/outputs/report/tables/table_novelty_metrics.csv`
- novelty_tex: `artifacts/local_bigger_v8/outputs/report/tables/table_novelty_metrics.tex`

## Experimental Notes
- Granularity summaries include vocabulary-list uncertainty (95% CI).
- Counterfactual vocabulary stress tests quantify brittleness to lexical perturbations.
- Novelty tables include GSI and vocabulary-instability metrics.
- Shift interaction includes uncertainty over ID-to-OOD drop estimates.
- Risk calibration reports target satisfaction rates and guarantee margins.