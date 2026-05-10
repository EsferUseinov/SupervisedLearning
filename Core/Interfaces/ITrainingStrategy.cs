namespace SupervisedLearning.Core.Interfaces;

using SupervisedLearning.Core;
using SupervisedLearning.Data;


public interface ITrainingStrategy
{
    double RunEpoch(Network network, DataSample[] batch, double learningRate);
}
