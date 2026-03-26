namespace SupervisedLearning.Training.Optimizers;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;

public class SGDOptimizer : IOptimizer
{
    public void UpdateWeights(ILayer layer, GradientPacket gradients, double learningRate) =>
        layer.ApplyGradients(gradients, learningRate);
}
