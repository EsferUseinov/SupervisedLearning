namespace SupervisedLearning.Data;

public class DataSet
{
    public DataSample[] Samples { get; }

    public DataSet(DataSample[] samples) => Samples = samples;

    public DataSet[] TrainTestSplit(double trainRatio) => throw new NotImplementedException();
    public DataSample[][] CreateBatches(int batchSize) => throw new NotImplementedException();
    public void Shuffle(int seed) => throw new NotImplementedException();
}
