# GSRD Final Report

## Headline Numbers
- **best_map**: {'detector': 'mock', 'dataset': 'toy_det', 'granularity': 'fine', 'mAP_mean': 0.0001650165016501}
- **max_worst_case_drop**: 0.0
- **risk_target_success_rate**: 0.0
- **mean_test_coverage**: 0.0

## Generated Figures (PNG + PDF)
- granularity_fig: `artifacts/eccv_grade_small/outputs/report/figures/granularity_sensitivity.png` and `artifacts/eccv_grade_small/outputs/report/figures/granularity_sensitivity.pdf`
- risk_fig: `artifacts/eccv_grade_small/outputs/report/figures/risk_vs_coverage.png` and `artifacts/eccv_grade_small/outputs/report/figures/risk_vs_coverage.pdf`

## Generated Tables
- performance_csv: `artifacts/eccv_grade_small/outputs/report/tables/table_performance_by_granularity.csv`
- performance_tex: `artifacts/eccv_grade_small/outputs/report/tables/table_performance_by_granularity.tex`
- worst_drop_csv: `artifacts/eccv_grade_small/outputs/report/tables/table_worst_case_drop.csv`
- worst_drop_tex: `artifacts/eccv_grade_small/outputs/report/tables/table_worst_case_drop.tex`
- calib_csv: `artifacts/eccv_grade_small/outputs/report/tables/table_calibration_effectiveness.csv`
- calib_tex: `artifacts/eccv_grade_small/outputs/report/tables/table_calibration_effectiveness.tex`
- risk_summary_csv: `artifacts/eccv_grade_small/outputs/report/tables/table_risk_summary.csv`
- risk_summary_tex: `artifacts/eccv_grade_small/outputs/report/tables/table_risk_summary.tex`

## Experimental Notes
- Granularity summaries include vocabulary-list uncertainty (95% CI).
- Shift interaction includes uncertainty over ID-to-OOD drop estimates.
- Risk calibration reports target satisfaction rates and guarantee margins.