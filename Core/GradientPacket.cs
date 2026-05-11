namespace SupervisedLearning.Core;

public class GradientPacket
{
    public double[] WeightGradients { get; }
    public double[] BiasGradients { get; }

    public GradientPacket(int weightCount, int biasCount)
    {
        WeightGradients = new double[weightCount];
        BiasGradients = new double[biasCount];
    }

    public void Accumulate(GradientPacket other)
    {
        for (int i = 0; i < WeightGradients.Length; i++)
            WeightGradients[i] += other.WeightGradients[i];
        for (int i = 0; i < BiasGradients.Length; i++)
            BiasGradients[i] += other.BiasGradients[i];
    }

    public void Scale(double factor)
    {
        for (int i = 0; i < WeightGradients.Length; i++)
            WeightGradients[i] *= factor;
        for (int i = 0; i < BiasGradients.Length; i++)
            BiasGradients[i] *= factor;
    }

    public void Reset()
    {
        Array.Clear(WeightGradients);
        Array.Clear(BiasGradients);
    }

    public void AccumulateRange(GradientPacket other, int fromWeight, int toWeight)
    {
        for (int i = fromWeight; i < toWeight; i++)
            WeightGradients[i] += other.WeightGradients[i];
    }

    public void ScaleRange(double factor, int fromWeight, int toWeight)
    {
        for (int i = fromWeight; i < toWeight; i++)
            WeightGradients[i] *= factor;
    }
}
