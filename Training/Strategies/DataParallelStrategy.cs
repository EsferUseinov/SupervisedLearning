namespace SupervisedLearning.Training.Strategies;

using System.Threading;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class DataParallelStrategy : ITrainingStrategy
{
    private readonly int _threadCount;
    private readonly bool _useThreadPool;

    public DataParallelStrategy(int threadCount, bool useThreadPool = false)
    {
        _threadCount = threadCount;
        _useThreadPool = useThreadPool;
    }

    public string Name => _useThreadPool
        ? $"DataParallel-ThreadPool-{_threadCount}"
        : $"DataParallel-Thread-{_threadCount}";

    public EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate) =>
        throw new NotImplementedException();

    private void RunWithThreads(Network network, DataSample[][] chunks, GradientPacket[] localGradients) =>
        throw new NotImplementedException();

    private void RunWithThreadPool(Network network, DataSample[][] chunks, GradientPacket[] localGradients) =>
        throw new NotImplementedException();

    private static DataSample[][] SplitIntoChunks(DataSample[] batch, int count) =>
        throw new NotImplementedException();

    private static GradientPacket AccumulateGradients(GradientPacket[] localGradients) =>
        throw new NotImplementedException();
}
