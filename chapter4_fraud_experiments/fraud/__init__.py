"""Experiments of Chapter 4 of the thesis: Confidex on two real fraud datasets.

Modules, in the order in which the pipeline uses them:

    registry, specs, ordinal_maps, loader    the two datasets, with checks of fingerprint and counts
    seeds                                    one random generator per experimental condition
    data                                     the partitions into training, calibration, validation and test
    preprocessor                             preparation of the variables, fitted on the training part only
    plan, defaults, families, model_registry the 36 members of each construction and how they are built
    resampling, tabpfn_service               oversampling for the perceptron; the TabPFN remote service
    members                                  training, calibration on the log-odds and validation diagnostics
    thresholds, readings                     the two thresholds; the signal and the simpler readings
    evaluate, ablations                      the evaluation of one condition and its comparisons
    performance                              descriptive performance of members and ensembles
    manifest                                 provenance of the results (package versions, git state)
"""
