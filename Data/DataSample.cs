namespace SupervisedLearning.Data;

public class DataSample
{
    public double[] Input { get; }
    public double[] Label { get; }

    public DataSample(double[] input, double[] label)
    {
        Input = input;
        Label = label;
    }
}
