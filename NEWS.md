# LBBNN 0.1.1 (development version)
* Initial tests added; cleanup for R CMD check portability.
* Initial CRAN submission.

# LBBNN 0.1.2 (minor bug fix)
* Fix bug in LBBNN_Linear so layers work properly in convolutional architecture.
* Updated experiment on convolutional architecture for R journal submission.
* Added default values to mpm and draws in predict() function.

# LBBNN 0.1.3 (minor cosmetic changes)
* Changed some function names to be consistent with CRAN guidelines.
* Removed export of function that is only used internally.
* Changed assignment operator to be consistent everywhere. 
* Removed datasets not used in RJ article submission.

# LBBNN 0.1.4 (minor bug fix)
* Added relu to input-skip layers. Updated experiments.

# LBBNN 0.1.5
* All torch-dependent examples are now guarded by
 `torch_available()` to ensure safe execution when torch or libtorch
  is not available.

# LBBNN 0.1.6 (resubmission to R journal)
* Added more vignettes, built pkgdown website.
* More robust testing.
* Bug fix in conv2d layer.
* Small changes to make training on gpu more efficient. 
* Added possibility to turn off printing to console during training.
* Utility functions work with standard LBBNN architecture without input-skip.
* predict() and validate_lbbnn() no longer change the model object.
* fixed bug that compute_paths() was not being called for LBBNN models.
* added initialization keywords such as 'dense' or 'balanced' for inclusion probabilities. 
* added possibility for different weight initializations. 

