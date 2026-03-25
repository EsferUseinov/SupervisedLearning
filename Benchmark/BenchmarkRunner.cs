namespace SupervisedLearning.Benchmark;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;
using SupervisedLearning.Training;

public class BenchmarkRunner
{
    public BenchmarkReport Compare(
        ITrainingStrategy[] strategies,
        Network network,
        DataSet dataset,
        TrainingConfig config) => throw new NotImplementedException();

    public BenchmarkReport ScalabilitySweep(
        Network network,
        DataSet dataset,
        TrainingConfig config,
        int[] threadCounts) => throw new NotImplementedException();
}
