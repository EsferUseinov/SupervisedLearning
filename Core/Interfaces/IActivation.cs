namespace SupervisedLearning.Core.Interfaces;

public interface IActivation
{
    double Derivative(double x);
    double[] ComputeVector(double[] x);
}
