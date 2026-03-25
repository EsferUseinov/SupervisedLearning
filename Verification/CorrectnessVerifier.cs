namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Data;

public class CorrectnessVerifier
{
    private readonly double _epsilon;

    public CorrectnessVerifier(double epsilon = 1e-8)
    {
        _epsilon = epsilon;
    }

    public bool DeterministicCompare(
        Network networkA,
        Network networkB,
        DataSet dataset,
        int seed) => throw new NotImplementedException();
}
