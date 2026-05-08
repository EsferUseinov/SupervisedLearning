namespace SupervisedLearning.Core.Loss;

using SupervisedLearning.Core.Interfaces;

public class CrossEntropyLoss : ILossFunction
{
    private const double Eps = 1e-12;
    private readonly double[] _classWeights;

    public CrossEntropyLoss(double[] classWeights)
    {
        _classWeights = classWeights;
    }

    public double Compute(double[] predicted, double[] actual)
    {
        int trueClass = ArgMax(actual);
        double p = Math.Clamp(predicted[trueClass], Eps, 1.0);
        return -_classWeights[trueClass] * Math.Log(p);
    }

    public double[] Gradient(double[] predicted, double[] actual)
    {
        int trueClass = ArgMax(actual);
        var grad = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
            grad[i] = _classWeights[trueClass] * (predicted[i] - actual[i]);
        return grad;
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[idx]) idx = i;
        return idx;
    }
}
