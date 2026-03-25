namespace SupervisedLearning.Experiment;

using SupervisedLearning.Training;

public class ExperimentConfig
{
    public int[] NetworkTopology { get; set; } = Array.Empty<int>();
    public TrainingConfig TrainingConfig { get; set; } = new();

    public static ExperimentConfig Default() => throw new NotImplementedException();
}

public class ParameterGrid
{
    public double[] LearningRates { get; set; } = Array.Empty<double>();
    public int[] BatchSizes { get; set; } = Array.Empty<int>();
    public int[] ThreadCounts { get; set; } = Array.Empty<int>();

    public static ParameterGrid Empty() => new();
}
