namespace SupervisedLearning.Experiment;

using SupervisedLearning.Training;

public class ExperimentConfig
{
    public int[] NetworkTopology { get; set; } = Array.Empty<int>();
    public TrainingConfig TrainingConfig { get; set; } = new();

    public static ExperimentConfig Default() => new ExperimentConfig
    {
        NetworkTopology = new[] { 4, 16, 8, 2 },
        TrainingConfig = new TrainingConfig
        {
            Epochs = 20,
            BatchSize = 32,
            LearningRate = 0.01,
            ThreadCount = 4,
            Seed = 42
        }
    };
}

public class ParameterGrid
{
    public double[] LearningRates { get; set; } = Array.Empty<double>();
    public int[] BatchSizes { get; set; } = Array.Empty<int>();
    public int[] ThreadCounts { get; set; } = Array.Empty<int>();

    public static ParameterGrid Empty() => new();
}
