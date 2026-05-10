namespace SupervisedLearning.Core.Layers;

using SupervisedLearning.Core.Interfaces;

public class DenseLayer : ILayer
{
    private readonly double[,] _weights;
    private readonly double[] _bias;
    private readonly IActivation _activation;
    private double[] _lastInput = Array.Empty<double>();
    private double[] _lastZ = Array.Empty<double>();
    private readonly GradientPacket _gradients;

    public int InputSize { get; }
    public int OutputSize { get; }

    public DenseLayer(int inputSize, int outputSize, IActivation activation, int seed = 0)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        _activation = activation;
        _weights = new double[outputSize, inputSize];
        _bias = new double[outputSize];
        _gradients = new GradientPacket(outputSize * inputSize, outputSize);

        var rng = new Random(seed);
        double scale = Math.Sqrt(2.0 / (inputSize + outputSize));

        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < inputSize; j++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                _weights[i, j] = gaussian * scale;
            }
    }

    public double[] Forward(double[] input)
    {
        _lastInput = input;
        _lastZ = new double[OutputSize];

        for (int i = 0; i < OutputSize; i++)
        {
            double z = _bias[i];
            for (int j = 0; j < InputSize; j++)
                z += _weights[i, j] * input[j];
            _lastZ[i] = z;
        }

        return _activation.ComputeVector(_lastZ);
    }

    public double[] Backward(double[] gradientFromNext)
    {
        var dz = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
            dz[i] = gradientFromNext[i] * _activation.Derivative(_lastZ[i]);

        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
                _gradients.WeightGradients[i * InputSize + j] += dz[i] * _lastInput[j];
            _gradients.BiasGradients[i] += dz[i];
        }

        var dx = new double[InputSize];
        for (int i = 0; i < OutputSize; i++)
            for (int j = 0; j < InputSize; j++)
                dx[j] += _weights[i, j] * dz[i];

        return dx;
    }

    public double GetWeight(int i, int j) => _weights[i, j];
    public void SetWeight(int i, int j, double value) => _weights[i, j] = value;
    public double GetBias(int i) => _bias[i];
    public void SetBias(int i, double value) => _bias[i] = value;

    public GradientPacket GetGradients() => _gradients;
    public GradientPacket CreateEmptyGradients() => new GradientPacket(OutputSize * InputSize, OutputSize);

    public void ApplyGradients(GradientPacket gradients, double learningRate)
    {
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
                _weights[i, j] -= learningRate * gradients.WeightGradients[i * InputSize + j];
            _bias[i] -= learningRate * gradients.BiasGradients[i];
        }
    }

    public void SyncFrom(ILayer source)
    {
        var src = (DenseLayer)source;
        Array.Copy(src._weights, _weights, _weights.Length);
        Array.Copy(src._bias, _bias, _bias.Length);
    }

    public double MaxAbsDiff(ILayer other)
    {
        var o = (DenseLayer)other;
        double max = 0.0;
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                double d = Math.Abs(_weights[i, j] - o._weights[i, j]);
                if (d > max) max = d;
            }
            double bd = Math.Abs(_bias[i] - o._bias[i]);
            if (bd > max) max = bd;
        }
        return max;
    }

    public ILayer Clone()
    {
        var clone = new DenseLayer(InputSize, OutputSize, _activation);
        Array.Copy(_weights, clone._weights, _weights.Length);
        Array.Copy(_bias, clone._bias, _bias.Length);
        return clone;
    }
}
