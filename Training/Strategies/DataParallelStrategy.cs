namespace SupervisedLearning.Training.Strategies;

using System.Diagnostics;
using System.Threading;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class DataParallelStrategy : ITrainingStrategy
{
    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer _optimizer;
    private readonly int _threadCount;
    private readonly bool _useThreadPool;

    public DataParallelStrategy(ILossFunction lossFunction, IOptimizer optimizer,
        int threadCount, bool useThreadPool = false)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
        _threadCount = threadCount;
        _useThreadPool = useThreadPool;
    }

    public string Name => _useThreadPool
        ? $"DataParallel-ThreadPool-{_threadCount}"
        : $"DataParallel-Thread-{_threadCount}";

    public EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate)
    {
        var epochTimer = Stopwatch.StartNew();

        int layerCount = network.Layers.Count;
        DataSample[][] chunks = SplitIntoChunks(batch, _threadCount);
        int actualThreads = chunks.Length;

        var networkClones = new Network[actualThreads];
        var threadGradients = new GradientPacket[actualThreads][];
        var threadLosses = new double[actualThreads];

        for (int t = 0; t < actualThreads; t++)
        {
            networkClones[t] = network.Clone();
            threadGradients[t] = new GradientPacket[layerCount];
            for (int l = 0; l < layerCount; l++)
                threadGradients[t][l] = new GradientPacket(
                    network.Layers[l].InputSize, network.Layers[l].OutputSize);
        }

        if (_useThreadPool)
            RunWithThreadPool(networkClones, chunks, threadGradients, threadLosses);
        else
            RunWithThreads(networkClones, chunks, threadGradients, threadLosses);

        var syncTimer = Stopwatch.StartNew();
        GradientPacket[] accumulated = AccumulateGradients(threadGradients, layerCount);
        syncTimer.Stop();

        for (int l = 0; l < layerCount; l++)
        {
            accumulated[l].Scale(1.0 / batch.Length);
            _optimizer.UpdateWeights(network.Layers[l], accumulated[l], learningRate);
        }

        epochTimer.Stop();

        double totalLoss = 0.0;
        foreach (double loss in threadLosses) totalLoss += loss;

        return new EpochResult
        {
            Loss = totalLoss / batch.Length,
            ForwardTimeMs = 0,
            BackwardTimeMs = 0,
            SyncTimeMs = syncTimer.ElapsedMilliseconds,
            EpochDurationMs = epochTimer.ElapsedMilliseconds,
            SamplesProcessed = batch.Length
        };
    }

    private void RunWithThreads(
        Network[] networkClones,
        DataSample[][] chunks,
        GradientPacket[][] threadGradients,
        double[] threadLosses)
    {
        var threads = new Thread[networkClones.Length];
        for (int t = 0; t < networkClones.Length; t++)
        {
            int idx = t;
            threads[t] = new Thread(() =>
                ProcessChunk(networkClones[idx], chunks[idx], threadGradients[idx], threadLosses, idx));
            threads[t].Start();
        }
        foreach (var thread in threads) thread.Join();
    }

    private void RunWithThreadPool(
        Network[] networkClones,
        DataSample[][] chunks,
        GradientPacket[][] threadGradients,
        double[] threadLosses)
    {
        using var countdown = new CountdownEvent(networkClones.Length);
        for (int t = 0; t < networkClones.Length; t++)
        {
            int idx = t;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                ProcessChunk(networkClones[idx], chunks[idx], threadGradients[idx], threadLosses, idx);
                countdown.Signal();
            });
        }
        countdown.Wait();
    }

    private void ProcessChunk(
        Network networkClone,
        DataSample[] chunk,
        GradientPacket[] layerGradients,
        double[] threadLosses,
        int threadIdx)
    {
        int layerCount = networkClone.Layers.Count;
        foreach (var g in layerGradients) g.Reset();

        double totalLoss = 0.0;

        foreach (var sample in chunk)
        {
            double[] predicted = networkClone.Forward(sample.Input);
            totalLoss += _lossFunction.Compute(predicted, sample.Label);
            double[] lossGrad = _lossFunction.Gradient(predicted, sample.Label);
            networkClone.Backward(lossGrad);

            for (int l = 0; l < layerCount; l++)
            {
                layerGradients[l].Accumulate(networkClone.Layers[l].GetGradients());
                networkClone.Layers[l].GetGradients().Reset();
            }
        }

        threadLosses[threadIdx] = totalLoss;
    }

    private static DataSample[][] SplitIntoChunks(DataSample[] batch, int count)
    {
        int actualCount = Math.Min(count, batch.Length);
        var chunks = new DataSample[actualCount][];
        int baseSize = batch.Length / actualCount;
        int remainder = batch.Length % actualCount;

        int offset = 0;
        for (int i = 0; i < actualCount; i++)
        {
            int size = baseSize + (i < remainder ? 1 : 0);
            chunks[i] = new DataSample[size];
            Array.Copy(batch, offset, chunks[i], 0, size);
            offset += size;
        }
        return chunks;
    }

    private static GradientPacket[] AccumulateGradients(GradientPacket[][] threadGradients, int layerCount)
    {
        var accumulated = new GradientPacket[layerCount];
        for (int l = 0; l < layerCount; l++)
        {
            int outputSize = threadGradients[0][l].WeightGradients.GetLength(0);
            int inputSize = threadGradients[0][l].WeightGradients.GetLength(1);
            accumulated[l] = new GradientPacket(inputSize, outputSize);

            for (int t = 0; t < threadGradients.Length; t++)
                accumulated[l].Accumulate(threadGradients[t][l]);
        }
        return accumulated;
    }
}
