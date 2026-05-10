namespace SupervisedLearning.Training;

using System.Diagnostics;
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

    public TrainingResult Train(Network network, DataSet dataset)
    {
        var epochResults = new List<EpochResult>();
        var totalTimer = Stopwatch.StartNew();

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            dataset.Shuffle(_config.Seed + epoch);
            var batches = dataset.CreateBatches(_config.BatchSize);

            double epochLoss = 0.0;
            var epochTimer = Stopwatch.StartNew();

            foreach (var batch in batches)
            {
                var batchResult = _strategy.RunEpoch(network, batch, _config.LearningRate);
                epochLoss += batchResult.Loss;
            }

            epochTimer.Stop();

            epochResults.Add(new EpochResult
            {
                Loss = epochLoss / batches.Length,
                EpochDurationMs = epochTimer.ElapsedMilliseconds
            });
        }

        totalTimer.Stop();

        return new TrainingResult
        {
            EpochResults = epochResults.ToArray(),
            TotalDurationMs = totalTimer.ElapsedMilliseconds,
            FinalLoss = epochResults[^1].Loss
        };
    }
}
