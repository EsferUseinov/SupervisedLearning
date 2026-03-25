namespace SupervisedLearning.Core.Interfaces;

using SupervisedLearning.Core;
using SupervisedLearning.Data;

public interface ITrainingStrategy
{
    string Name { get; }
    EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate);
}
