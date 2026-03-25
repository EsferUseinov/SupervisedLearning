namespace SupervisedLearning.Core.Interfaces;

public interface ILossFunction
{
    double Compute(double[] predicted, double[] actual);
    double[] Gradient(double[] predicted, double[] actual);
}
