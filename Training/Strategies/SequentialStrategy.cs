namespace SupervisedLearning.Training.Strategies;

using System.Diagnostics;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class SequentialStrategy : ITrainingStrategy
{
    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer _optimizer;

    public string Name => "Sequential";

    public SequentialStrategy(ILossFunction lossFunction, IOptimizer optimizer)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
    }

    public EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate)
    {
        var epochTimer = Stopwatch.StartNew();

        int layerCount = network.Layers.Count;
        double totalLoss = 0.0;

        for (int i = 0; i < layerCount; i++)
            network.Layers[i].GetGradients().Reset();

        foreach (var sample in batch)
        {
            double[] predicted = network.Forward(sample.Input);
            totalLoss += _lossFunction.Compute(predicted, sample.Label);
            network.Backward(_lossFunction.Gradient(predicted, sample.Label));
        }

        for (int i = 0; i < layerCount; i++)
        {
            network.Layers[i].GetGradients().Scale(1.0 / batch.Length);
            _optimizer.UpdateWeights(network.Layers[i], network.Layers[i].GetGradients(), learningRate);
        }

        epochTimer.Stop();

        return new EpochResult
        {
            Loss = totalLoss / batch.Length,
            EpochDurationMs = epochTimer.ElapsedMilliseconds
        };
    }
}
