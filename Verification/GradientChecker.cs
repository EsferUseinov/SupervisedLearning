namespace SupervisedLearning.Verification;

using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Core.Layers;
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
        ComputeMaxRelativeError(network, loss, sample) <= _tolerance;

    public double ComputeMaxRelativeError(Network network, ILossFunction loss, DataSample sample)
    {
        foreach (var layer in network.Layers)
            layer.GetGradients().Reset();

        double[] predicted = network.Forward(sample.Input);
        double[] lossGrad = loss.Gradient(predicted, sample.Label);
        network.Backward(lossGrad);

        var analytical = new GradientPacket[network.Layers.Count];
        for (int l = 0; l < network.Layers.Count; l++)
        {
            var copy = new GradientPacket(network.Layers[l].InputSize, network.Layers[l].OutputSize);
            copy.Accumulate(network.Layers[l].GetGradients());
            analytical[l] = copy;
        }

        double maxError = 0.0;

        for (int l = 0; l < network.Layers.Count; l++)
        {
            var dense = (DenseLayer)network.Layers[l];

            for (int i = 0; i < dense.OutputSize; i++)
            {
                for (int j = 0; j < dense.InputSize; j++)
                {
                    double orig = dense.GetWeight(i, j);

                    dense.SetWeight(i, j, orig + _epsilon);
                    double lossPlus = loss.Compute(network.Forward(sample.Input), sample.Label);

                    dense.SetWeight(i, j, orig - _epsilon);
                    double lossMinus = loss.Compute(network.Forward(sample.Input), sample.Label);

                    dense.SetWeight(i, j, orig);

                    double numerical = (lossPlus - lossMinus) / (2.0 * _epsilon);
                    double analyticalVal = analytical[l].WeightGradients[i, j];
                    double relErr = Math.Abs(analyticalVal - numerical) /
                        (Math.Abs(analyticalVal) + Math.Abs(numerical) + 1e-8);

                    if (relErr > maxError) maxError = relErr;
                }

                double origBias = dense.GetBias(i);

                dense.SetBias(i, origBias + _epsilon);
                double lossPlusBias = loss.Compute(network.Forward(sample.Input), sample.Label);

                dense.SetBias(i, origBias - _epsilon);
                double lossMinusBias = loss.Compute(network.Forward(sample.Input), sample.Label);

                dense.SetBias(i, origBias);

                double numericalBias = (lossPlusBias - lossMinusBias) / (2.0 * _epsilon);
                double analyticalBias = analytical[l].BiasGradients[i];
                double relErrBias = Math.Abs(analyticalBias - numericalBias) /
                    (Math.Abs(analyticalBias) + Math.Abs(numericalBias) + 1e-8);

                if (relErrBias > maxError) maxError = relErrBias;
            }
        }

        return maxError;
    }
}
