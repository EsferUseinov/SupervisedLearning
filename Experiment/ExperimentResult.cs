namespace SupervisedLearning.Experiment;

using SupervisedLearning.Core;

public class ExperimentResult
{
    public ExperimentConfig Config { get; set; } = new();
    public TrainingResult TrainingResult { get; set; } = new();
    public double ValidationAccuracy { get; set; }
}
