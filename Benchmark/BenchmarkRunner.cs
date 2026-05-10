namespace SupervisedLearning.Benchmark;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Strategies;

public class BenchmarkRunner
{
    private const int WarmupRuns  = 1;
    private const int MeasuredRuns = 3;

    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer    _optimizer;

    public BenchmarkRunner(ILossFunction lossFunction, IOptimizer optimizer)
    {
        _lossFunction = lossFunction;
        _optimizer    = optimizer;
    }

    public BenchmarkReport ScalabilitySweep(
        Network     network,
        DataSet     dataset,
        TrainingConfig config,
        int[]       threadCounts)
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
            var threadStrategy = new DataParallelStrategy(_lossFunction, _optimizer, t, useThreadPool: false);
            var poolStrategy   = new DataParallelStrategy(_lossFunction, _optimizer, t, useThreadPool: true);

            for (int w = 0; w < WarmupRuns; w++)
            {
                new Trainer(threadStrategy, config).Train(network.Clone(), dataset);
                new Trainer(poolStrategy,   config).Train(network.Clone(), dataset);
            }

            long threadTotal = 0, poolTotal = 0;
            for (int r = 0; r < MeasuredRuns; r++)
            {
                threadTotal += new Trainer(threadStrategy, config).Train(network.Clone(), dataset).TotalDurationMs;
                poolTotal   += new Trainer(poolStrategy,   config).Train(network.Clone(), dataset).TotalDurationMs;
            }

            long threadAvg = threadTotal / MeasuredRuns;
            long poolAvg   = poolTotal   / MeasuredRuns;
            double threadSpeedup = seqAvg / (double)Math.Max(threadAvg, 1);
            double poolSpeedup   = seqAvg / (double)Math.Max(poolAvg,   1);

            entries[i] = new ScalabilityEntry
            {
                ThreadCount      = t,
                SeqTimeMs        = seqAvg,
                ThreadTimeMs     = threadAvg,
                ThreadSpeedup    = threadSpeedup,
                ThreadEfficiency = threadSpeedup / t,
                PoolTimeMs       = poolAvg,
                PoolSpeedup      = poolSpeedup,
                PoolEfficiency   = poolSpeedup / t,
            };
        }

        return new BenchmarkReport { ScalabilityData = entries };
    }

    public BenchmarkReport DataSizeSweep(
        Network        network,
        DataSet        fullDataset,
        TrainingConfig config,
        int[]          sampleCounts,
        int            parallelThreads)
    {
        var seqStrategy    = new SequentialStrategy(_lossFunction, _optimizer);
        var threadStrategy = new DataParallelStrategy(_lossFunction, _optimizer, parallelThreads, useThreadPool: false);
        var poolStrategy   = new DataParallelStrategy(_lossFunction, _optimizer, parallelThreads, useThreadPool: true);
        var entries        = new DataSizeEntry[sampleCounts.Length];

        for (int i = 0; i < sampleCounts.Length; i++)
        {
            int n      = Math.Min(sampleCounts[i], fullDataset.Samples.Length);
            var subset = new DataSet(fullDataset.Samples[..n]);

            for (int w = 0; w < WarmupRuns; w++)
            {
                new Trainer(seqStrategy,    config).Train(network.Clone(), subset);
                new Trainer(threadStrategy, config).Train(network.Clone(), subset);
                new Trainer(poolStrategy,   config).Train(network.Clone(), subset);
            }

            long seqTotal = 0, threadTotal = 0, poolTotal = 0;
            for (int r = 0; r < MeasuredRuns; r++)
            {
                seqTotal    += new Trainer(seqStrategy,    config).Train(network.Clone(), subset).TotalDurationMs;
                threadTotal += new Trainer(threadStrategy, config).Train(network.Clone(), subset).TotalDurationMs;
                poolTotal   += new Trainer(poolStrategy,   config).Train(network.Clone(), subset).TotalDurationMs;
            }

            long seqAvg    = seqTotal    / MeasuredRuns;
            long threadAvg = threadTotal / MeasuredRuns;
            long poolAvg   = poolTotal   / MeasuredRuns;
            double threadSpeedup = seqAvg / (double)Math.Max(threadAvg, 1);
            double poolSpeedup   = seqAvg / (double)Math.Max(poolAvg,   1);

            entries[i] = new DataSizeEntry
            {
                SampleCount      = n,
                SeqTimeMs        = seqAvg,
                ThreadTimeMs     = threadAvg,
                ThreadSpeedup    = threadSpeedup,
                ThreadEfficiency = threadSpeedup / parallelThreads,
                PoolTimeMs       = poolAvg,
                PoolSpeedup      = poolSpeedup,
                PoolEfficiency   = poolSpeedup / parallelThreads,
            };
        }

        return new BenchmarkReport { DataSizeData = entries };
    }
}
