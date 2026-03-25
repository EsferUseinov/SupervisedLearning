namespace SupervisedLearning.Training;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class Trainer
{
    private readonly ITrainingStrategy _strategy;
    private readonly TrainingConfig _config;

    public Trainer(ITrainingStrategy strategy, TrainingConfig config)
    {
        _strategy = strategy;
        _config = config;
    }

    public TrainingResult Train(Network network, DataSet dataset) =>
        throw new NotImplementedException();
}
