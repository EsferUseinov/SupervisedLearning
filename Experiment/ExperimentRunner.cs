namespace SupervisedLearning.Experiment;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Activations;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Data;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Strategies;

public class ExperimentRunner
{
    private readonly DataSet _dataset;
    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer _optimizer;

    public ExperimentRunner(DataSet dataset, ILossFunction lossFunction, IOptimizer optimizer)
    {
        _dataset = dataset;
        _lossFunction = lossFunction;
        _optimizer = optimizer;
    }

    public ExperimentResult[] RunSweep(ExperimentConfig baseConfig, ParameterGrid grid)
    {
        _dataset.Shuffle(baseConfig.TrainingConfig.Seed);
        var splits = _dataset.TrainTestSplit(0.8);
        var trainSet = splits[0];
        var testSet = splits[1];

        double[] lrs = grid.LearningRates.Length > 0
            ? grid.LearningRates
            : new[] { baseConfig.TrainingConfig.LearningRate };

        int[] batches = grid.BatchSizes.Length > 0
            ? grid.BatchSizes
            : new[] { baseConfig.TrainingConfig.BatchSize };

        int[] threads = grid.ThreadCounts.Length > 0
            ? grid.ThreadCounts
            : new[] { baseConfig.TrainingConfig.ThreadCount };

        var results = new List<ExperimentResult>();

        foreach (double lr in lrs)
            foreach (int batchSize in batches)
                foreach (int threadCount in threads)
                {
                    var config = new TrainingConfig
                    {
                        Epochs = baseConfig.TrainingConfig.Epochs,
                        BatchSize = batchSize,
                        LearningRate = lr,
                        ThreadCount = threadCount,
                        Seed = baseConfig.TrainingConfig.Seed
                    };

                    var expConfig = new ExperimentConfig
                    {
                        NetworkTopology = baseConfig.NetworkTopology,
                        TrainingConfig = config
                    };

                    var network = BuildNetwork(baseConfig.NetworkTopology, baseConfig.TrainingConfig.Seed);

                    ITrainingStrategy strategy = threadCount > 1
                        ? new DataParallelStrategy(_lossFunction, _optimizer, threadCount)
                        : (ITrainingStrategy)new SequentialStrategy(_lossFunction, _optimizer);

                    var trainingResult = new Trainer(strategy, config).Train(network, trainSet);
                    double accuracy = ComputeAccuracy(network, testSet.Samples);

                    results.Add(new ExperimentResult
                    {
                        Config = expConfig,
                        TrainingResult = trainingResult,
                        ValidationAccuracy = accuracy
                    });
                }

        return results.ToArray();
    }

    private Network BuildNetwork(int[] topology, int seed)
    {
        var network = new Network();
        for (int i = 0; i < topology.Length - 1; i++)
        {
            bool isLast = i == topology.Length - 2;
            IActivation activation = isLast ? new Sigmoid() : new ReLU();
            network.AddLayer(new DenseLayer(topology[i], topology[i + 1], activation, seed + i));
        }
        return network;
    }

    private static double ComputeAccuracy(Network network, DataSample[] samples)
    {
        int correct = 0;
        foreach (var sample in samples)
        {
            double[] predicted = network.Forward(sample.Input);
            if (ArgMax(predicted) == ArgMax(sample.Label))
                correct++;
        }
        return (double)correct / samples.Length;
    }

    private static int ArgMax(double[] values)
    {
        int maxIdx = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[maxIdx]) maxIdx = i;
        return maxIdx;
    }
}
