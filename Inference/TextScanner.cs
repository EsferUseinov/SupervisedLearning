namespace SupervisedLearning.Inference;

using SupervisedLearning.Core;
using SupervisedLearning.Data;
using SupervisedLearning.Data.Preprocessing;

public class TextScanner
{
    private readonly Network _network;
    private readonly Vocabulary _vocab;
    private readonly int _seqLen;

    public record DetectedSpan(int StartChar, int EndChar, string Technique, double Confidence);

    public TextScanner(Network network, Vocabulary vocab, int seqLen)
    {
        _network = network;
        _vocab = vocab;
        _seqLen = seqLen;
    }

    public IReadOnlyList<DetectedSpan> Scan(string text, int stride = 16, double minConfidence = 0.5)
    {
        var tokens = TokenizeWithPositions(text);
        if (tokens.Length == 0) return Array.Empty<DetectedSpan>();

        var results = new List<DetectedSpan>();
        int notPropIdx = SemEvalLoader.Techniques.Length - 1;

        for (int windowStart = 0; windowStart < tokens.Length; windowStart += stride)
        {
            int windowEnd = Math.Min(windowStart + _seqLen, tokens.Length);
            double[] probs = _network.Forward(BuildInput(tokens, windowStart, windowEnd));
            int predicted = MathHelper.ArgMax(probs);

            if (predicted != notPropIdx && probs[predicted] >= minConfidence)
            {
                results.Add(new DetectedSpan(
                    tokens[windowStart].CharStart,
                    tokens[windowEnd - 1].CharEnd,
                    SemEvalLoader.Techniques[predicted],
                    probs[predicted]));
            }
        }

        return results;
    }

    private double[] BuildInput((string Token, int CharStart, int CharEnd)[] tokens, int from, int to)
    {
        var input = new double[_seqLen];
        for (int i = 0; i < _seqLen; i++)
        {
            int pos = from + i;
            input[i] = pos < to ? _vocab.GetIndex(tokens[pos].Token) : _vocab.PadIndex;
        }
        return input;
    }

    private static (string Token, int CharStart, int CharEnd)[] TokenizeWithPositions(string text)
    {
        string lower = text.ToLowerInvariant();
        var result = new List<(string, int, int)>();
        int i = 0;

        while (i < lower.Length)
        {
            if (Tokenizer.SeparatorSet.Contains(lower[i])) { i++; continue; }
            int start = i;
            while (i < lower.Length && !Tokenizer.SeparatorSet.Contains(lower[i])) i++;
            result.Add((lower[start..i], start, i));
        }

        return result.ToArray();
    }
}
