namespace SupervisedLearning.Training.Strategies;

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

    private Network[]? _networkClones;
    private GradientPacket[]? _accumulated;
    private double[]? _threadLosses;
    private ArraySegment<DataSample>[]? _chunks;

    private const int ParallelReduceThreshold = 10_000;

    public DataParallelStrategy(ILossFunction lossFunction, IOptimizer optimizer,
        int threadCount, bool useThreadPool = false)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
        _threadCount = threadCount;
        _useThreadPool = useThreadPool;
    }

    private void EnsureInitialized(Network network)
    {
        if (_networkClones != null) return;

        int layerCount = network.Layers.Count;

        _networkClones = new Network[_threadCount];
        _accumulated = new GradientPacket[layerCount];
        _threadLosses = new double[_threadCount];
        _chunks = new ArraySegment<DataSample>[_threadCount];

        for (int t = 0; t < _threadCount; t++)
            _networkClones[t] = network.Clone();

        for (int l = 0; l < layerCount; l++)
            _accumulated[l] = network.Layers[l].CreateEmptyGradients();
    }

    private static void SyncWeights(Network source, Network target)
    {
        for (int i = 0; i < source.Layers.Count; i++)
            target.Layers[i].SyncFrom(source.Layers[i]);
    }

    public double RunEpoch(Network network, DataSample[] batch, double learningRate)
    {
        EnsureInitialized(network);

        int layerCount    = network.Layers.Count;
        int actualThreads = FillChunks(batch);

        if (_useThreadPool)
        {
            SyncWeightsWithThreadPool(network, actualThreads);
            RunWithThreadPool(_networkClones!, _chunks!, _threadLosses!, actualThreads);
        }
        else
        {
            SyncWeightsWithThreads(network, actualThreads);
            RunWithThreads(_networkClones!, _chunks!, _threadLosses!, actualThreads);
        }

        ReduceAndUpdate(network, actualThreads, layerCount, batch.Length, learningRate);

        double totalLoss = 0.0;
        for (int t = 0; t < actualThreads; t++) totalLoss += _threadLosses![t];

        return totalLoss / batch.Length;
    }

    private int FillChunks(DataSample[] batch)
    {
        int actualCount = Math.Min(_threadCount, batch.Length);
        int baseSize    = batch.Length / actualCount;
        int remainder   = batch.Length % actualCount;
        int offset      = 0;

        for (int i = 0; i < actualCount; i++)
        {
            int size    = baseSize + (i < remainder ? 1 : 0);
            _chunks![i] = new ArraySegment<DataSample>(batch, offset, size);
            offset     += size;
        }

        return actualCount;
    }

    private void SyncWeightsWithThreads(Network network, int actualThreads)
    {
        var threads = new Thread[actualThreads];
        for (int t = 0; t < actualThreads; t++)
        {
            int idx = t;
            threads[t] = new Thread(() => SyncWeights(network, _networkClones![idx]));
            threads[t].Start();
        }
        foreach (var th in threads) th.Join();
    }

    private void SyncWeightsWithThreadPool(Network network, int actualThreads)
    {
        using var done = new CountdownEvent(actualThreads);
        for (int t = 0; t < actualThreads; t++)
        {
            int idx = t;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                SyncWeights(network, _networkClones![idx]);
                done.Signal();
            });
        }
        done.Wait();
    }

    private void RunWithThreads(
        Network[] networkClones,
        ArraySegment<DataSample>[] chunks,
        double[] threadLosses,
        int actualThreads)
    {
        var threads = new Thread[actualThreads];
        for (int t = 0; t < actualThreads; t++)
        {
            int idx = t;
            threads[t] = new Thread(() =>
                ProcessChunk(networkClones[idx], chunks[idx], threadLosses, idx));
            threads[t].Start();
        }
        foreach (var thread in threads) thread.Join();
    }

    private void RunWithThreadPool(
        Network[] networkClones,
        ArraySegment<DataSample>[] chunks,
        double[] threadLosses,
        int actualThreads)
    {
        using var countdown = new CountdownEvent(actualThreads);
        for (int t = 0; t < actualThreads; t++)
        {
            int idx = t;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                ProcessChunk(networkClones[idx], chunks[idx], threadLosses, idx);
                countdown.Signal();
            });
        }
        countdown.Wait();
    }

    private void ReduceAndUpdate(Network network, int actualThreads, int layerCount, int batchSize, double learningRate)
    {
        double invBatch = 1.0 / batchSize;

        for (int l = 0; l < layerCount; l++)
        {
            var acc    = _accumulated![l];
            int wCount = acc.WeightGradients.Length;

            if (wCount >= ParallelReduceThreshold)
            {
                int rangeSize = (wCount + _threadCount - 1) / _threadCount;

                if (_useThreadPool)
                    ReduceWeightsWithThreadPool(acc, actualThreads, l, wCount, rangeSize, invBatch);
                else
                    ReduceWeightsWithThreads(acc, actualThreads, l, wCount, rangeSize, invBatch);

                int bCount = acc.BiasGradients.Length;
                if (bCount > 0)
                {
                    Array.Clear(acc.BiasGradients);
                    for (int t = 0; t < actualThreads; t++)
                    {
                        var srcBias = _networkClones![t].Layers[l].GetGradients().BiasGradients;
                        for (int i = 0; i < bCount; i++)
                            acc.BiasGradients[i] += srcBias[i];
                    }
                    for (int i = 0; i < bCount; i++)
                        acc.BiasGradients[i] *= invBatch;
                }
            }
            else
            {
                acc.Reset();
                for (int t = 0; t < actualThreads; t++)
                    acc.Accumulate(_networkClones![t].Layers[l].GetGradients());
                acc.Scale(invBatch);
            }

            _optimizer.UpdateWeights(network.Layers[l], acc, learningRate);
        }
    }

    private void ReduceWeightsWithThreads(
        GradientPacket acc, int actualThreads, int layerIdx,
        int wCount, int rangeSize, double invBatch)
    {
        int rangeCount = (wCount + rangeSize - 1) / rangeSize;
        var threads    = new Thread[rangeCount];

        for (int r = 0; r < rangeCount; r++)
        {
            int rf = r * rangeSize;
            int rt = Math.Min(rf + rangeSize, wCount);

            threads[r] = new Thread(() =>
            {
                Array.Clear(acc.WeightGradients, rf, rt - rf);
                for (int ti = 0; ti < actualThreads; ti++)
                    acc.AccumulateRange(_networkClones![ti].Layers[layerIdx].GetGradients(), rf, rt);
                acc.ScaleRange(invBatch, rf, rt);
            });
            threads[r].Start();
        }

        foreach (var th in threads) th.Join();
    }

    private void ReduceWeightsWithThreadPool(
        GradientPacket acc, int actualThreads, int layerIdx,
        int wCount, int rangeSize, double invBatch)
    {
        using var done = new CountdownEvent(_threadCount);

        for (int r = 0; r < _threadCount; r++)
        {
            int rf = r * rangeSize;
            if (rf >= wCount) { done.Signal(); continue; }
            int rt = Math.Min(rf + rangeSize, wCount);

            ThreadPool.QueueUserWorkItem(_ =>
            {
                Array.Clear(acc.WeightGradients, rf, rt - rf);
                for (int ti = 0; ti < actualThreads; ti++)
                    acc.AccumulateRange(_networkClones![ti].Layers[layerIdx].GetGradients(), rf, rt);
                acc.ScaleRange(invBatch, rf, rt);
                done.Signal();
            });
        }

        done.Wait();
    }

    private void ProcessChunk(
        Network networkClone,
        ArraySegment<DataSample> chunk,
        double[] threadLosses,
        int threadIdx)
    {
        int layerCount = networkClone.Layers.Count;
        double totalLoss = 0.0;

        for (int l = 0; l < layerCount; l++)
            networkClone.Layers[l].GetGradients().Reset();

        foreach (var sample in chunk)
        {
            double[] predicted = networkClone.Forward(sample.Input);
            totalLoss += _lossFunction.Compute(predicted, sample.Label);
            networkClone.Backward(_lossFunction.Gradient(predicted, sample.Label));
        }

        threadLosses[threadIdx] = totalLoss;
    }
}
