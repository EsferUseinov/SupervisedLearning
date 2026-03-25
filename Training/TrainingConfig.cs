namespace SupervisedLearning.Training;

using SupervisedLearning.Core.Interfaces;

public class TrainingConfig
{
    public int Epochs { get; set; } = 10;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.01;
    public int ThreadCount { get; set; } = 4;
    public int Seed { get; set; } = 42;
    public IOptimizer? Optimizer { get; set; }
    public ILossFunction? LossFunction { get; set; }
}
