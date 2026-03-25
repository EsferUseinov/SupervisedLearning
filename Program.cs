using SupervisedLearning.Experiment;

var runner = new ExperimentRunner();
runner.RunSweep(ExperimentConfig.Default(), ParameterGrid.Empty());
