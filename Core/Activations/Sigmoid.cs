namespace SupervisedLearning.Core.Activations;

using SupervisedLearning.Core.Interfaces;

public class Sigmoid : IActivation
{
    public double Compute(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double Derivative(double x)
    {
        double s = Compute(x);
        return s * (1.0 - s);
    }

    public double[] ComputeVector(double[] x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            result[i] = Compute(x[i]);
        return result;
    }

    public double[] DerivativeVector(double[] x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            result[i] = Derivative(x[i]);
        return result;
    }
}
