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
        dataset.Shuffle(_config.Seed);
        var batches = dataset.CreateBatches(_config.BatchSize);
        var epochResults = new List<EpochResult>();
        var lossCurve = new List<double>();

        var totalTimer = Stopwatch.StartNew();

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            double epochLoss = 0.0;
            long epochForwardMs = 0;
            long epochBackwardMs = 0;
            int samplesProcessed = 0;
            var epochTimer = Stopwatch.StartNew();

            foreach (var batch in batches)
            {
                var batchResult = _strategy.RunEpoch(network, batch, _config.LearningRate);
                epochLoss += batchResult.Loss;
                epochForwardMs += batchResult.ForwardTimeMs;
                epochBackwardMs += batchResult.BackwardTimeMs;
                samplesProcessed += batchResult.SamplesProcessed;
            }

            epochTimer.Stop();

            var epochResult = new EpochResult
            {
                Loss = epochLoss / batches.Length,
                ForwardTimeMs = epochForwardMs,
                BackwardTimeMs = epochBackwardMs,
                SyncTimeMs = 0,
                EpochDurationMs = epochTimer.ElapsedMilliseconds,
                SamplesProcessed = samplesProcessed
            };

            epochResults.Add(epochResult);
            lossCurve.Add(epochResult.Loss);
        }

        totalTimer.Stop();

        return new TrainingResult
        {
            EpochResults = epochResults.ToArray(),
            TotalDurationMs = totalTimer.ElapsedMilliseconds,
            FinalLoss = epochResults[^1].Loss,
            StrategyName = _strategy.Name,
            LossCurve = lossCurve.ToArray()
        };
    }
}
