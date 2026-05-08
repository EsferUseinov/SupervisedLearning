namespace SupervisedLearning.Training;

public class TrainingConfig
{
    public int Epochs { get; set; } = 10;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.01;
    public int Seed { get; set; } = 42;
    public int[] LrDecaySteps { get; set; } = Array.Empty<int>();
    public double LrDecayFactor { get; set; } = 1.0;
}
