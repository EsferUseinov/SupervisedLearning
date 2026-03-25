namespace SupervisedLearning.Core.Interfaces;

public interface IActivation
{
    double Compute(double x);
    double Derivative(double x);
    double[] ComputeVector(double[] x);
    double[] DerivativeVector(double[] x);
}
