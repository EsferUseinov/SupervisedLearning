namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
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
            double diff = networkA.Layers[l].MaxAbsDiff(networkB.Layers[l]);
            if (diff > maxDiff) maxDiff = diff;
        }
        return maxDiff;
    }
}
