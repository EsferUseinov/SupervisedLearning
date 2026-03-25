namespace SupervisedLearning.Data;

public static class DataLoader
{
    public static DataSet LoadFromCsv(string path) => throw new NotImplementedException();

    public static DataSet GenerateSynthetic(int samples, int inputSize, int outputSize, int seed)
    {
        var rng = new Random(seed);
        var result = new DataSample[samples];
        for (int i = 0; i < samples; i++)
        {
            var input = new double[inputSize];
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++)
            {
                input[j] = rng.NextDouble() * 2.0 - 1.0;
                sum += input[j];
            }
            var label = new double[outputSize];
            int classIndex = sum > 0.0 ? 0 : 1;
            if (classIndex < outputSize)
                label[classIndex] = 1.0;
            result[i] = new DataSample(input, label);
        }
        return new DataSet(result);
    }
}
