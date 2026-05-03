namespace SupervisedLearning.Core.Loss;

using SupervisedLearning.Core.Interfaces;

public class CrossEntropyLoss : ILossFunction
{
    private const double Eps = 1e-12;

    public double Compute(double[] predicted, double[] actual)
    {
        double loss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = Math.Clamp(predicted[i], Eps, 1.0 - Eps);
            loss -= actual[i] * Math.Log(p) + (1.0 - actual[i]) * Math.Log(1.0 - p);
        }
        return loss;
    }

    public double[] Gradient(double[] predicted, double[] actual)
    {
        var grad = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = Math.Clamp(predicted[i], Eps, 1.0 - Eps);
            grad[i] = -actual[i] / p + (1.0 - actual[i]) / (1.0 - p);
        }
        return grad;
    }
}
