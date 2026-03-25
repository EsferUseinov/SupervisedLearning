namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
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

    public bool Check(Network network, ILossFunction loss, DataSample sample) =>
        throw new NotImplementedException();
}
