namespace SupervisedLearning.Benchmark;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Strategies;

public class BenchmarkRunner
{
    private const int WarmupRuns = 1;
    private const int MeasuredRuns = 3;

    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer _optimizer;

    public BenchmarkRunner(ILossFunction lossFunction, IOptimizer optimizer)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
    }

    public BenchmarkReport Compare(
        ITrainingStrategy[] strategies,
        Network network,
        DataSet dataset,
        TrainingConfig config)
    {
        var results = new TrainingResult[strategies.Length];

        for (int s = 0; s < strategies.Length; s++)
        {
            for (int w = 0; w < WarmupRuns; w++)
                new Trainer(strategies[s], config).Train(network.Clone(), dataset);

            long totalMs = 0;
            TrainingResult last = null!;
            for (int r = 0; r < MeasuredRuns; r++)
            {
                last = new Trainer(strategies[s], config).Train(network.Clone(), dataset);
                totalMs += last.TotalDurationMs;
            }

            results[s] = new TrainingResult
            {
                EpochResults = last.EpochResults,
                TotalDurationMs = totalMs / MeasuredRuns,
                FinalLoss = last.FinalLoss,
                StrategyName = strategies[s].Name,
                LossCurve = last.LossCurve
            };
        }

        double baselineMs = results[0].TotalDurationMs;
        double parallelMs = results.Length > 1 ? results[1].TotalDurationMs : baselineMs;
        double speedup = baselineMs / Math.Max(parallelMs, 1);
        double efficiency = config.ThreadCount > 0 ? speedup / config.ThreadCount : 1.0;

        return new BenchmarkReport
        {
            Results = results,
            Speedup = speedup,
            Efficiency = efficiency
        };
    }

    public BenchmarkReport ScalabilitySweep(
        Network network,
        DataSet dataset,
        TrainingConfig config,
        int[] threadCounts)
    {
        var seqStrategy = new SequentialStrategy(_lossFunction, _optimizer);

        for (int w = 0; w < WarmupRuns; w++)
            new Trainer(seqStrategy, config).Train(network.Clone(), dataset);

        long seqTotal = 0;
        for (int r = 0; r < MeasuredRuns; r++)
            seqTotal += new Trainer(seqStrategy, config).Train(network.Clone(), dataset).TotalDurationMs;

        long seqAvg = seqTotal / MeasuredRuns;

        var entries = new ScalabilityEntry[threadCounts.Length];

        for (int i = 0; i < threadCounts.Length; i++)
        {
            int t = threadCounts[i];
            var parStrategy = new DataParallelStrategy(_lossFunction, _optimizer, t);

            for (int w = 0; w < WarmupRuns; w++)
                new Trainer(parStrategy, config).Train(network.Clone(), dataset);

            long parTotal = 0;
            for (int r = 0; r < MeasuredRuns; r++)
                parTotal += new Trainer(parStrategy, config).Train(network.Clone(), dataset).TotalDurationMs;

            long parAvg = parTotal / MeasuredRuns;
            double speedup = seqAvg / (double)Math.Max(parAvg, 1);

            entries[i] = new ScalabilityEntry
            {
                ThreadCount = t,
                Speedup = speedup,
                Efficiency = speedup / t
            };
        }

        return new BenchmarkReport { ScalabilityData = entries };
    }
}
