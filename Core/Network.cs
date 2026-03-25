namespace SupervisedLearning.Core;

using SupervisedLearning.Core.Interfaces;

public class Network
{
    private readonly List<ILayer> _layers = new();

    public IReadOnlyList<ILayer> Layers => _layers;

    public void AddLayer(ILayer layer) => throw new NotImplementedException();
    public double[] Forward(double[] input) => throw new NotImplementedException();
    public double[] Backward(double[] lossGradient) => throw new NotImplementedException();
    public Network Clone() => throw new NotImplementedException();
}
