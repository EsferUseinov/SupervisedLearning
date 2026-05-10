namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Data;

public class GradientChecker
{
    private readonly double _epsilon;
    private readonly double _tolerance;

    public GradientChecker(double epsilon = 1e-5, double tolerance = 1e-4)
    {
        _epsilon = epsilon;
        _tolerance = tolerance;
    }

    public (double MaxError, int Passed, int Total) Evaluate(Network network, ILossFunction loss, DataSample sample)
    {
        foreach (var layer in network.Layers)
            layer.GetGradients().Reset();

        double[] predicted = network.Forward(sample.Input);
        network.Backward(loss.Gradient(predicted, sample.Label));

        var analytical = new GradientPacket[network.Layers.Count];
        for (int l = 0; l < network.Layers.Count; l++)
        {
            var copy = network.Layers[l].CreateEmptyGradients();
            copy.Accumulate(network.Layers[l].GetGradients());
            analytical[l] = copy;
        }

        double maxError = 0.0;
        int passed = 0, total = 0;

        for (int l = 0; l < network.Layers.Count; l++)
        {
            var (lErr, lPassed, lTotal) = network.Layers[l] switch
            {
                DenseLayer dense   => EvaluateDense(network, loss, sample, dense, analytical[l]),
                EmbeddingLayer emb => EvaluateEmbedding(network, loss, sample, emb, analytical[l]),
                Conv1DLayer conv   => EvaluateConv1D(network, loss, sample, conv, analytical[l]),
                _                  => (0.0, 0, 0)
            };
            if (lErr > maxError) maxError = lErr;
            passed += lPassed;
            total  += lTotal;
        }

        return (maxError, passed, total);
    }

    private (double maxError, int passed, int total) EvaluateDense(
        Network network, ILossFunction loss, DataSample sample,
        DenseLayer dense, GradientPacket analytical)
    {
        double maxError = 0.0;
        int passed = 0, total = 0;

        for (int i = 0; i < dense.OutputSize; i++)
        {
            for (int j = 0; j < dense.InputSize; j++)
            {
                double orig = dense.GetWeight(i, j);

                dense.SetWeight(i, j, orig + _epsilon);
                double lp = loss.Compute(network.Forward(sample.Input), sample.Label);

                dense.SetWeight(i, j, orig - _epsilon);
                double lm = loss.Compute(network.Forward(sample.Input), sample.Label);

                dense.SetWeight(i, j, orig);

                double err = RelErr(analytical.WeightGradients[i * dense.InputSize + j],
                    (lp - lm) / (2.0 * _epsilon));
                total++;
                if (err <= _tolerance) passed++;
                if (err > maxError) maxError = err;
            }

            double origBias = dense.GetBias(i);

            dense.SetBias(i, origBias + _epsilon);
            double lpb = loss.Compute(network.Forward(sample.Input), sample.Label);

            dense.SetBias(i, origBias - _epsilon);
            double lmb = loss.Compute(network.Forward(sample.Input), sample.Label);

            dense.SetBias(i, origBias);

            double errBias = RelErr(analytical.BiasGradients[i], (lpb - lmb) / (2.0 * _epsilon));
            total++;
            if (errBias <= _tolerance) passed++;
            if (errBias > maxError) maxError = errBias;
        }

        return (maxError, passed, total);
    }

    private (double maxError, int passed, int total) EvaluateEmbedding(
        Network network, ILossFunction loss, DataSample sample,
        EmbeddingLayer emb, GradientPacket analytical)
    {
        var seenTokens = new HashSet<int>();
        for (int i = 0; i < sample.Input.Length; i++)
            seenTokens.Add((int)sample.Input[i]);

        double maxError = 0.0;
        int passed = 0, total = 0;

        foreach (int tokenIdx in seenTokens)
        {
            if (tokenIdx == 0) continue;
            for (int d = 0; d < emb.EmbeddingDim; d++)
            {
                int flatIdx = tokenIdx * emb.EmbeddingDim + d;
                double orig = emb.GetTableEntry(flatIdx);

                emb.SetTableEntry(flatIdx, orig + _epsilon);
                double lp = loss.Compute(network.Forward(sample.Input), sample.Label);

                emb.SetTableEntry(flatIdx, orig - _epsilon);
                double lm = loss.Compute(network.Forward(sample.Input), sample.Label);

                emb.SetTableEntry(flatIdx, orig);

                double err = RelErr(analytical.WeightGradients[flatIdx], (lp - lm) / (2.0 * _epsilon));
                total++;
                if (err <= _tolerance) passed++;
                if (err > maxError) maxError = err;
            }
        }

        return (maxError, passed, total);
    }

    private (double maxError, int passed, int total) EvaluateConv1D(
        Network network, ILossFunction loss, DataSample sample,
        Conv1DLayer conv, GradientPacket analytical)
    {
        double maxError = 0.0;
        int passed = 0, total = 0;

        for (int i = 0; i < conv.FilterWeightCount; i++)
        {
            double orig = conv.GetFilter(i);

            conv.SetFilter(i, orig + _epsilon);
            double lp = loss.Compute(network.Forward(sample.Input), sample.Label);

            conv.SetFilter(i, orig - _epsilon);
            double lm = loss.Compute(network.Forward(sample.Input), sample.Label);

            conv.SetFilter(i, orig);

            double err = RelErr(analytical.WeightGradients[i], (lp - lm) / (2.0 * _epsilon));
            total++;
            if (err <= _tolerance) passed++;
            if (err > maxError) maxError = err;
        }

        for (int f = 0; f < conv.NumFilters; f++)
        {
            double orig = conv.GetConvBias(f);

            conv.SetConvBias(f, orig + _epsilon);
            double lp = loss.Compute(network.Forward(sample.Input), sample.Label);

            conv.SetConvBias(f, orig - _epsilon);
            double lm = loss.Compute(network.Forward(sample.Input), sample.Label);

            conv.SetConvBias(f, orig);

            double err = RelErr(analytical.BiasGradients[f], (lp - lm) / (2.0 * _epsilon));
            total++;
            if (err <= _tolerance) passed++;
            if (err > maxError) maxError = err;
        }

        return (maxError, passed, total);
    }

    private static double RelErr(double analytical, double numerical) =>
        Math.Abs(analytical - numerical) / (Math.Abs(analytical) + Math.Abs(numerical) + 1e-8);
}
