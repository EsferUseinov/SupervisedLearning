namespace SupervisedLearning.Core.Layers;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;

public class EmbeddingLayer : ILayer
{
    private readonly double[] _table;
    private readonly int _vocabSize;
    private readonly int _embeddingDim;
    private readonly int _seqLen;
    private readonly GradientPacket _gradients;
    private int[] _lastTokens = Array.Empty<int>();

    public int InputSize => _seqLen;
    public int OutputSize => _seqLen * _embeddingDim;
    public int VocabSize => _vocabSize;
    public int EmbeddingDim => _embeddingDim;
    public int SeqLen => _seqLen;
    public double GetTableEntry(int flatIdx) => _table[flatIdx];
    public void SetTableEntry(int flatIdx, double value) => _table[flatIdx] = value;

    public EmbeddingLayer(int vocabSize, int embeddingDim, int seqLen, int seed = 0)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        _seqLen = seqLen;
        _table = new double[vocabSize * embeddingDim];
        _gradients = new GradientPacket(vocabSize * embeddingDim, 0);

        var rng = new Random(seed);
        double scale = 1.0 / Math.Sqrt(embeddingDim);
        for (int i = embeddingDim; i < _table.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            _table[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
        }
    }

    public double[] Forward(double[] input)
    {
        _lastTokens = new int[_seqLen];
        var output = new double[_seqLen * _embeddingDim];
        for (int pos = 0; pos < _seqLen; pos++)
        {
            int idx = (int)input[pos];
            _lastTokens[pos] = idx;
            int tableBase = idx * _embeddingDim;
            int outBase = pos * _embeddingDim;
            for (int d = 0; d < _embeddingDim; d++)
                output[outBase + d] = _table[tableBase + d];
        }
        return output;
    }

    public double[] Backward(double[] gradientFromNext)
    {
        for (int pos = 0; pos < _seqLen; pos++)
        {
            int idx = _lastTokens[pos];
            if (idx == 0) continue;
            int tableBase = idx * _embeddingDim;
            int gradBase = pos * _embeddingDim;
            for (int d = 0; d < _embeddingDim; d++)
                _gradients.WeightGradients[tableBase + d] += gradientFromNext[gradBase + d];
        }
        return new double[_seqLen];
    }

    public GradientPacket GetGradients() => _gradients;
    public GradientPacket CreateEmptyGradients() => new GradientPacket(_vocabSize * _embeddingDim, 0);

    public void ApplyGradients(GradientPacket gradients, double learningRate)
    {
        for (int i = 0; i < _table.Length; i++)
            _table[i] -= learningRate * gradients.WeightGradients[i];
    }

    public void SyncFrom(ILayer source)
    {
        var src = (EmbeddingLayer)source;
        Array.Copy(src._table, _table, _table.Length);
    }

    public double MaxAbsDiff(ILayer other)
    {
        var o = (EmbeddingLayer)other;
        double max = 0.0;
        for (int i = 0; i < _table.Length; i++)
        {
            double d = Math.Abs(_table[i] - o._table[i]);
            if (d > max) max = d;
        }
        return max;
    }

    public ILayer Clone()
    {
        var clone = new EmbeddingLayer(_vocabSize, _embeddingDim, _seqLen);
        Array.Copy(_table, clone._table, _table.Length);
        return clone;
    }
}
