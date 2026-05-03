namespace SupervisedLearning.Data;

public static class DataLoader
{
    public static DataSet LoadFromCsv(string path, int outputClasses = 10)
    {
        var samples = new List<DataSample>();
        using var reader = new StreamReader(path);
        string? line = reader.ReadLine();
        while (line != null)
        {
            if (line.Length == 0 || !char.IsDigit(line[0]))
            {
                line = reader.ReadLine();
                continue;
            }
            var parts = line.Split(',');
            int label = int.Parse(parts[0]);
            var input = new double[parts.Length - 1];
            for (int j = 1; j < parts.Length; j++)
                input[j - 1] = int.Parse(parts[j]) / 255.0;
            var labelVec = new double[outputClasses];
            labelVec[label] = 1.0;
            samples.Add(new DataSample(input, labelVec));
            line = reader.ReadLine();
        }
        return new DataSet(samples.ToArray());
    }

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
