namespace SupervisedLearning.Core.Activations;

using SupervisedLearning.Core.Interfaces;

public class Softmax : IActivation
{
    public double Compute(double x) =>
        throw new NotSupportedException("Softmax is defined only for vectors.");

    public double Derivative(double x) => 1.0;


    public double[] ComputeVector(double[] x)
    {
        double max = x[0];
        for (int i = 1; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        double sum = 0.0;
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = Math.Exp(x[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < x.Length; i++)
            result[i] /= sum;
        return result;
    }

    public double[] DerivativeVector(double[] x)
    {
        var ones = new double[x.Length];
        Array.Fill(ones, 1.0);
        return ones;
    }
}
