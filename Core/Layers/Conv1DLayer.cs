namespace SupervisedLearning.Core.Layers;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;

public class Conv1DLayer : ILayer
{
    private readonly double[] _filters;
    private readonly double[] _bias;
    private readonly IActivation _activation;
    private readonly int _inChannels;
    private readonly int _seqLen;
    private readonly int _filterSize;
    private readonly int _numFilters;
    private readonly int _outputLen;
    private readonly GradientPacket _gradients;
    private double[] _lastInput = Array.Empty<double>();
    private double[] _lastZ = Array.Empty<double>();

    public int InputSize => _seqLen * _inChannels;
    public int OutputSize => _outputLen * _numFilters;
    public int NumFilters => _numFilters;
    public int FilterSize => _filterSize;
    public int InChannels => _inChannels;
    public int SeqLen => _seqLen;
    public int FilterWeightCount => _numFilters * _filterSize * _inChannels;
    public double GetFilter(int flatIdx) => _filters[flatIdx];
    public void SetFilter(int flatIdx, double value) => _filters[flatIdx] = value;
    public double GetConvBias(int f) => _bias[f];
    public void SetConvBias(int f, double value) => _bias[f] = value;

    public Conv1DLayer(int seqLen, int inChannels, int filterSize, int numFilters, IActivation activation, int seed = 0)
    {
        _seqLen = seqLen;
        _inChannels = inChannels;
        _filterSize = filterSize;
        _numFilters = numFilters;
        _activation = activation;
        _outputLen = seqLen - filterSize + 1;

        int filterWeightCount = numFilters * filterSize * inChannels;
        _filters = new double[filterWeightCount];
        _bias = new double[numFilters];
        _gradients = new GradientPacket(filterWeightCount, numFilters);

        var rng = new Random(seed);
        double scale = Math.Sqrt(2.0 / (filterSize * inChannels + numFilters));
        for (int i = 0; i < filterWeightCount; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            _filters[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
        }
    }

    public double[] Forward(double[] input)
    {
        _lastInput = input;
        _lastZ = new double[_outputLen * _numFilters];

        for (int pos = 0; pos < _outputLen; pos++)
        {
            for (int f = 0; f < _numFilters; f++)
            {
                double val = _bias[f];
                int filterBase = f * _filterSize * _inChannels;
                for (int t = 0; t < _filterSize; t++)
                {
                    int inputBase = (pos + t) * _inChannels;
                    int filterOffset = t * _inChannels;
                    for (int c = 0; c < _inChannels; c++)
                        val += _filters[filterBase + filterOffset + c] * input[inputBase + c];
                }
                _lastZ[pos * _numFilters + f] = val;
            }
        }

        return _activation.ComputeVector(_lastZ);
    }

    public double[] Backward(double[] gradientFromNext)
    {
        var dz = new double[_outputLen * _numFilters];
        for (int i = 0; i < dz.Length; i++)
            dz[i] = gradientFromNext[i] * _activation.Derivative(_lastZ[i]);

        for (int f = 0; f < _numFilters; f++)
        {
            for (int pos = 0; pos < _outputLen; pos++)
                _gradients.BiasGradients[f] += dz[pos * _numFilters + f];
        }

        for (int f = 0; f < _numFilters; f++)
        {
            int filterBase = f * _filterSize * _inChannels;
            for (int t = 0; t < _filterSize; t++)
            {
                int filterOffset = t * _inChannels;
                for (int c = 0; c < _inChannels; c++)
                {
                    double acc = 0.0;
                    for (int pos = 0; pos < _outputLen; pos++)
                        acc += dz[pos * _numFilters + f] * _lastInput[(pos + t) * _inChannels + c];
                    _gradients.WeightGradients[filterBase + filterOffset + c] += acc;
                }
            }
        }

        var dx = new double[_seqLen * _inChannels];
        for (int pos2 = 0; pos2 < _seqLen; pos2++)
        {
            for (int c2 = 0; c2 < _inChannels; c2++)
            {
                double sum = 0.0;
                for (int t = 0; t < _filterSize; t++)
                {
                    int outPos = pos2 - t;
                    if (outPos < 0 || outPos >= _outputLen) continue;
                    for (int f = 0; f < _numFilters; f++)
                        sum += dz[outPos * _numFilters + f] *
                               _filters[f * _filterSize * _inChannels + t * _inChannels + c2];
                }
                dx[pos2 * _inChannels + c2] = sum;
            }
        }

        return dx;
    }

    public GradientPacket GetGradients() => _gradients;
    public GradientPacket CreateEmptyGradients() =>
        new GradientPacket(_numFilters * _filterSize * _inChannels, _numFilters);

    public void ApplyGradients(GradientPacket gradients, double learningRate)
    {
        for (int i = 0; i < _filters.Length; i++)
            _filters[i] -= learningRate * gradients.WeightGradients[i];
        for (int f = 0; f < _numFilters; f++)
            _bias[f] -= learningRate * gradients.BiasGradients[f];
    }

    public void SyncFrom(ILayer source)
    {
        var src = (Conv1DLayer)source;
        Array.Copy(src._filters, _filters, _filters.Length);
        Array.Copy(src._bias, _bias, _bias.Length);
    }

    public double MaxAbsDiff(ILayer other)
    {
        var o = (Conv1DLayer)other;
        double max = 0.0;
        for (int i = 0; i < _filters.Length; i++)
        {
            double d = Math.Abs(_filters[i] - o._filters[i]);
            if (d > max) max = d;
        }
        for (int f = 0; f < _numFilters; f++)
        {
            double d = Math.Abs(_bias[f] - o._bias[f]);
            if (d > max) max = d;
        }
        return max;
    }

    public ILayer Clone()
    {
        var clone = new Conv1DLayer(_seqLen, _inChannels, _filterSize, _numFilters, _activation);
        Array.Copy(_filters, clone._filters, _filters.Length);
        Array.Copy(_bias, clone._bias, _bias.Length);
        return clone;
    }
}
