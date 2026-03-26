namespace SupervisedLearning.Core.Loss;

using SupervisedLearning.Core.Interfaces;

public class MSELoss : ILossFunction
{
    public double Compute(double[] predicted, double[] actual)
    {
        double sum = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / 2.0;
    }

    public double[] Gradient(double[] predicted, double[] actual)
    {
        var grad = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
            grad[i] = predicted[i] - actual[i];
        return grad;
    }
}
