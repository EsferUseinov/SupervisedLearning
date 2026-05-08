namespace SupervisedLearning.Core;

using SupervisedLearning.Core.Interfaces;

public class Network
{
    private readonly List<ILayer> _layers = new();

    public IReadOnlyList<ILayer> Layers => _layers;

    public void AddLayer(ILayer layer)
    {
        if (_layers.Count > 0 && _layers[^1].OutputSize != layer.InputSize)
            throw new ArgumentException(
                $"Layer input size {layer.InputSize} does not match previous layer output size {_layers[^1].OutputSize}.");
        _layers.Add(layer);
    }

    public double[] Forward(double[] input)
    {
        double[] current = input;
        foreach (var layer in _layers)
            current = layer.Forward(current);
        return current;
    }

    public double[] Backward(double[] lossGradient)
    {
        double[] grad = lossGradient;
        for (int i = _layers.Count - 1; i >= 0; i--)
            grad = _layers[i].Backward(grad);
        return grad;
    }

    public void SetTrainingMode(bool training)
    {
        foreach (var layer in _layers)
            if (layer is Layers.DenseLayer dense)
                dense.IsTraining = training;
    }

    public Network Clone()
    {
        var clone = new Network();
        foreach (var layer in _layers)
            clone._layers.Add(layer.Clone());
        return clone;
    }
}
