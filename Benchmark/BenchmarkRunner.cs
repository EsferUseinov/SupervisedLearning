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

    public BenchmarkReport DataSizeSweep(
        Network network,
        DataSet fullDataset,
        TrainingConfig config,
        int[] sampleCounts,
        int parallelThreads)
    {
        var seqStrategy = new SequentialStrategy(_lossFunction, _optimizer);
        var parStrategy = new DataParallelStrategy(_lossFunction, _optimizer, parallelThreads);
        var entries = new DataSizeEntry[sampleCounts.Length];

        for (int i = 0; i < sampleCounts.Length; i++)
        {
            int n = Math.Min(sampleCounts[i], fullDataset.Samples.Length);
            var subset = new DataSet(fullDataset.Samples[..n]);

            for (int w = 0; w < WarmupRuns; w++)
            {
                new Trainer(seqStrategy, config).Train(network.Clone(), subset);
                new Trainer(parStrategy, config).Train(network.Clone(), subset);
            }

            long seqTotal = 0;
            long parTotal = 0;
            for (int r = 0; r < MeasuredRuns; r++)
            {
                seqTotal += new Trainer(seqStrategy, config).Train(network.Clone(), subset).TotalDurationMs;
                parTotal += new Trainer(parStrategy, config).Train(network.Clone(), subset).TotalDurationMs;
            }

            long seqAvg = seqTotal / MeasuredRuns;
            long parAvg = parTotal / MeasuredRuns;
            double speedup = seqAvg / (double)Math.Max(parAvg, 1);

            entries[i] = new DataSizeEntry
            {
                SampleCount = n,
                SeqTimeMs = seqAvg,
                ParTimeMs = parAvg,
                Speedup = speedup,
                Efficiency = speedup / parallelThreads
            };
        }

        return new BenchmarkReport { DataSizeData = entries };
    }
}
