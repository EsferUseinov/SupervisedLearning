namespace SupervisedLearning.Training.Strategies;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class SequentialStrategy : ITrainingStrategy
{
    public string Name => "Sequential";

    public EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate) =>
        throw new NotImplementedException();
}
