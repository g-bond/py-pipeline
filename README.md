# schollab_py_pipeline

A better write-up for usage instructions is required.

Code for handling, processing, and analysing the results from 2p movies (and some ephys).
Usual processing stages:
  - Collection off a scope and onto network-adjacent-storage (NAS)
  - Registration to reduce bulk motion
  - Optional fine-pass of piecewise-rigid registration for sparse movies.
  - Fine-tuning of a denoising model from DeepInterpolation
  - Application of the fine-tuned model to registered movies, yielding denoised movie.
  - Region-of-interest painting for this scene.
  - Extraction of calcium traces and dF/F signal for each ROI
  - ROI-based analysis of cells

A