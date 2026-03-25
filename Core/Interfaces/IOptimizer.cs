namespace SupervisedLearning.Core.Interfaces;

using SupervisedLearning.Core;

public interface IOptimizer
{
    void UpdateWeights(ILayer layer, GradientPacket gradients, double learningRate);
}
