namespace SupervisedLearning.Core.Layers;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;

public class MaxPool1DLayer : ILayer
{
    private readonly int _inputLen;
    private readonly int _numChannels;
    private readonly GradientPacket _gradients;
    private int[] _argmax = Array.Empty<int>();

    public int InputSize => _inputLen * _numChannels;
    public int OutputSize => _numChannels;

    public MaxPool1DLayer(int inputLen, int numChannels)
    {
        _inputLen = inputLen;
        _numChannels = numChannels;
        _gradients = new GradientPacket(0, 0);
    }

    public double[] Forward(double[] input)
    {
        _argmax = new int[_numChannels];
        var output = new double[_numChannels];
        for (int c = 0; c < _numChannels; c++)
        {
            double maxVal = double.NegativeInfinity;
            int maxPos = 0;
            for (int pos = 0; pos < _inputLen; pos++)
            {
                double val = input[pos * _numChannels + c];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxPos = pos;
                }
            }
            output[c] = maxVal;
            _argmax[c] = maxPos;
        }
        return output;
    }

    public double[] Backward(double[] gradientFromNext)
    {
        var dx = new double[_inputLen * _numChannels];
        for (int c = 0; c < _numChannels; c++)
            dx[_argmax[c] * _numChannels + c] = gradientFromNext[c];
        return dx;
    }

    public GradientPacket GetGradients() => _gradients;
    public GradientPacket CreateEmptyGradients() => new GradientPacket(0, 0);

    public void ApplyGradients(GradientPacket gradients, double learningRate) { }

    public void SyncFrom(ILayer source) { }

    public double MaxAbsDiff(ILayer other) => 0.0;

    public ILayer Clone() => new MaxPool1DLayer(_inputLen, _numChannels);
}
