namespace SupervisedLearning.Core.Layers;

using SupervisedLearning.Core.Interfaces;

public class DenseLayer : ILayer
{
    public int InputSize { get; }
    public int OutputSize { get; }

    public DenseLayer(int inputSize, int outputSize, IActivation activation)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        throw new NotImplementedException();
    }

    public double[] Forward(double[] input) => throw new NotImplementedException();
    public double[] Backward(double[] gradientFromNext) => throw new NotImplementedException();
    public GradientPacket GetGradients() => throw new NotImplementedException();
    public void ApplyGradients(GradientPacket gradients, double learningRate) => throw new NotImplementedException();
    public ILayer Clone() => throw new NotImplementedException();
}
