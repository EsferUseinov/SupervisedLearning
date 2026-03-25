namespace SupervisedLearning.Core;

public class GradientPacket
{
    public double[,] WeightGradients { get; }
    public double[] BiasGradients { get; }

    public GradientPacket(int inputSize, int outputSize)
    {
        WeightGradients = new double[outputSize, inputSize];
        BiasGradients = new double[outputSize];
    }

    public void Accumulate(GradientPacket other)
    {
        int rows = WeightGradients.GetLength(0);
        int cols = WeightGradients.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                WeightGradients[i, j] += other.WeightGradients[i, j];
            BiasGradients[i] += other.BiasGradients[i];
        }
    }

    public void Scale(double factor)
    {
        int rows = WeightGradients.GetLength(0);
        int cols = WeightGradients.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                WeightGradients[i, j] *= factor;
            BiasGradients[i] *= factor;
        }
    }

    public void Reset()
    {
        int rows = WeightGradients.GetLength(0);
        int cols = WeightGradients.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                WeightGradients[i, j] = 0.0;
            BiasGradients[i] = 0.0;
        }
    }
}
