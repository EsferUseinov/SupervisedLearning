namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Data;

public class CorrectnessVerifier
{
    private readonly ITrainingStrategy _sequential;
    private readonly ITrainingStrategy _parallel;
    private readonly int _batchSize;
    private readonly double _learningRate;
    private readonly double _epsilon;

    public CorrectnessVerifier(
        ITrainingStrategy sequential,
        ITrainingStrategy parallel,
        int batchSize,
        double learningRate,
        double epsilon = 1e-8)
    {
        _sequential = sequential;
        _parallel = parallel;
        _batchSize = batchSize;
        _learningRate = learningRate;
        _epsilon = epsilon;
    }

    public bool DeterministicCompare(
        Network networkA,
        Network networkB,
        DataSet dataset,
        int seed)
    {
        return MaxWeightDiff(networkA, networkB, dataset, seed) <= _epsilon;
    }

    public double MaxWeightDiff(
        Network networkA,
        Network networkB,
        DataSet dataset,
        int seed)
    {
        dataset.Shuffle(seed);
        var batches = dataset.CreateBatches(_batchSize);

        double globalMaxDiff = 0.0;

        foreach (var batch in batches)
        {
            _sequential.RunEpoch(networkA, batch, _learningRate);
            _parallel.RunEpoch(networkB, batch, _learningRate);

            double maxDiff = ComputeMaxWeightDiff(networkA, networkB);
            if (maxDiff > globalMaxDiff) globalMaxDiff = maxDiff;

            if (maxDiff > _epsilon)
                return maxDiff;
        }

        return globalMaxDiff;
    }

    private static double ComputeMaxWeightDiff(Network networkA, Network networkB)
    {
        double maxDiff = 0.0;

        for (int l = 0; l < networkA.Layers.Count; l++)
        {
            var layerA = (DenseLayer)networkA.Layers[l];
            var layerB = (DenseLayer)networkB.Layers[l];

            for (int i = 0; i < layerA.OutputSize; i++)
            {
                for (int j = 0; j < layerA.InputSize; j++)
                {
                    double diff = Math.Abs(layerA.GetWeight(i, j) - layerB.GetWeight(i, j));
                    if (diff > maxDiff) maxDiff = diff;
                }

                double biasDiff = Math.Abs(layerA.GetBias(i) - layerB.GetBias(i));
                if (biasDiff > maxDiff) maxDiff = biasDiff;
            }
        }

        return maxDiff;
    }
}
