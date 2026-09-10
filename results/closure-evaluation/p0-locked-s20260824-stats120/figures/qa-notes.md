# P0 figure QA, 2026-09-11

- One quantitative panel: the trajectory of mean final positives per augmented GT in each size bin, for one locked-baseline run. All 120 completed epochs are retained; no smoothing, extrapolation or seed-level error bars.
- Python/matplotlib only. Source preflight: 18 pass, 2 warn, 0 fail. The warnings are accepted because this report uses SVG/PDF plus a 600-dpi PNG (TIFF is not required), and the plotted means are within-epoch GT summaries from one run, not independent seed aggregates.
- PDF text audit: 19 text runs, minimum 7 pt, none below 5 pt. Source width is 183 mm. SVG/PDF editable text and source CSV are retained.
- Visual inspection: title, axes, legend and footer are readable without clipping or collisions. Circle/square/triangle markers supplement colour. Medium and large curves overlap because their values are close; neither series was shifted to manufacture separation. Axis starts at zero.
- The last-ten-epoch shift coincides with the configured Mosaic closure. The figure does not claim an isolated causal effect of STAL or a gain over adaptive/fixed controls.
- No microscopy, image cropping, synthetic observations, multiple-comparison testing or independent-seed uncertainty applies. This figure is not a P1 performance certificate.
