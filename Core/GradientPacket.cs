namespace SupervisedLearning.Core;

public class GradientPacket
{
    public double[,] WeightGradients { get; }
    public double[] BiasGradients { get; }

    public GradientPacket(int inputSize, int outputSize)
    {
        WeightGradients = new double[outputSize, inputSize];
        BiasGradients = new double[outputSize];
    }

    public void Accumulate(GradientPacket other) => throw new NotImplementedException();
    public void Scale(double factor) => throw new NotImplementedException();
    public void Reset() => throw new NotImplementedException();
}
