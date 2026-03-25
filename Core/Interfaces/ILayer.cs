namespace SupervisedLearning.Core.Interfaces;

using SupervisedLearning.Core;

public interface ILayer
{
    int InputSize { get; }
    int OutputSize { get; }
    double[] Forward(double[] input);
    double[] Backward(double[] gradientFromNext);
    GradientPacket GetGradients();
    void ApplyGradients(GradientPacket gradients, double learningRate);
    ILayer Clone();
}
