namespace SupervisedLearning.Core.Activations;

using SupervisedLearning.Core.Interfaces;

public class ReLU : IActivation
{
    public double Derivative(double x) => x > 0.0 ? 1.0 : 0.0;

    public double[] ComputeVector(double[] x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            result[i] = x[i] > 0.0 ? x[i] : 0.0;
        return result;
    }
}
