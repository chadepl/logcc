# Local-to-Global Correlation Clustering (LoGCC)

Repository for the LoGCC paper. 

```
pip install numpy
pip install scipy
pip install scikit-image
pip install matplotlib
```

## Experiments

This directory contains the scripts for running the experiments (data) and for generating the results for the paper (fig, tab).

To run the files, from the root of the repository run (note that we don't add the .py add the end):
`python -m route.to.the.python.file`

Experiments (run in the following order):
- `data_scaling_grid`: runs the pivot and cn-pivot methods with and without acceleration for a series of scenarios with synthetic data.
- `data_rd_grid`: runs the pivot and cn-pivot methods with and without acceleration for three real datasets: meteo, han_brainstem, han_parotidr.
- `data_rd_grid_add_time_and_cfv`: creates tables with times and cost function values (cfvs) based on results of the `data_rd_grid` script.
- `data_rd_grid_local_rhos`: runs the pivot and cn-pivot methods with and without acceleration for three real datasets: meteo, han_brainstem, han_parotidR. For the accelerated methods, it tries different values for the local threshold parameter (\rho_local).
- `data_rd_grid_local_rhos_add_time_and_cfv`: creates tables with times and cost function values (cfvs) based on results of the `data_rd_grid` script.

Figures (as in the paper):
- Fig. 1 (`fig_teaser` and `fig_teaser_blobs`): generate images for the teaser, which showcases how our framework operates.
- Fig. 3 (`fig_bounds`): visual explanation of the bounds we derive in the paper.
- Fig. 4 (`fig_sd_scaling_benchmark`): figures summarizing the results of scaling experiment on synthetic data across conditions.
- Fig. 5 (`fig_rd_cfv`): compares cost function values of accelerated and non-accelerated variants.
- Fig. 6 (`fig_rd_comp_qual`): produces clustering images of accelerated and unaccelerated variants.
- Fig. 7 (`fig_rd_rho_exp` and `fig_rd_rho_exp_han`): generate images to demonstrate the framework usage: an image with the results of the local step and several images with different global thresholds.
- Fig. 8 (`fig_rd_local_rho_ablation`): generates images showcasing clusterings with different local rhos and their difference in time and cost function value.

Tables (as in the paper):
- Tab. 1 (`tab_rd_times`): displays the times of the `data_rd_grid` experiment.
- Tab. 2 (`tab_rd_rho_exp`): displays the times when evaluating multiple thresholds.