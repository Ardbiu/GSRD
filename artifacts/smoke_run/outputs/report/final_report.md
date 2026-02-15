# GSRD Final Report

## Headline Numbers
- **best_map**: {'detector': 'mock', 'dataset': 'toy_det', 'granularity': 'mixed', 'mAP_mean': 0.0001833516685001}
- **mean_worst_case_drop**: 0.0
- **risk_control_success_rate**: 0.0
- **mean_coverage**: 0.0

## Generated Figures
- granularity_fig: `artifacts/smoke_run/outputs/report/figures/granularity_sensitivity.png`
- risk_fig: `artifacts/smoke_run/outputs/report/figures/risk_vs_coverage.png`

## Generated Tables
- performance_csv: `artifacts/smoke_run/outputs/report/tables/table_performance_by_granularity.csv`
- performance_tex: `artifacts/smoke_run/outputs/report/tables/table_performance_by_granularity.tex`
- worst_drop_csv: `artifacts/smoke_run/outputs/report/tables/table_worst_case_drop.csv`
- worst_drop_tex: `artifacts/smoke_run/outputs/report/tables/table_worst_case_drop.tex`
- calib_csv: `artifacts/smoke_run/outputs/report/tables/table_calibration_effectiveness.csv`
- calib_tex: `artifacts/smoke_run/outputs/report/tables/table_calibration_effectiveness.tex`

## Notes
- Results use cached detector outputs and are reproducible from logged configs + seeds.
- Granularity metrics aggregate over multiple vocabulary lists to avoid cherry-picking.