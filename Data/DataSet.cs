namespace SupervisedLearning.Data;

public class DataSet
{
    public DataSample[] Samples { get; }

    public DataSet(DataSample[] samples) => Samples = samples;

    public void Shuffle(int seed)
    {
        var rng = new Random(seed);
        for (int i = Samples.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (Samples[i], Samples[j]) = (Samples[j], Samples[i]);
        }
    }

    public DataSample[][] CreateBatches(int batchSize)
    {
        int batchCount = (int)Math.Ceiling((double)Samples.Length / batchSize);
        var batches = new DataSample[batchCount][];
        for (int i = 0; i < batchCount; i++)
        {
            int start = i * batchSize;
            int length = Math.Min(batchSize, Samples.Length - start);
            batches[i] = new DataSample[length];
            Array.Copy(Samples, start, batches[i], 0, length);
        }
        return batches;
    }

    public DataSet[] TrainTestSplit(double trainRatio)
    {
        int trainCount = (int)(Samples.Length * trainRatio);
        var train = new DataSample[trainCount];
        var test = new DataSample[Samples.Length - trainCount];
        Array.Copy(Samples, 0, train, 0, trainCount);
        Array.Copy(Samples, trainCount, test, 0, test.Length);
        return new[] { new DataSet(train), new DataSet(test) };
    }
}
