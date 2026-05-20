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

    private Thread[]? _persistentThreads;
    private Action[]? _threadWork;
    private Barrier? _startBarrier;
    private Barrier? _endBarrier;
    private int _nextChunk;
    private CountdownEvent? _poolCountdown;

    private const int ParallelReduceThreshold = 10_000;
    private const int GrainFactor = 4;
    private const int CacheLineDoubles = 8;

    private int ChunkCount => _useThreadPool ? _threadCount * GrainFactor : _threadCount;

    public DataParallelStrategy(ILossFunction lossFunction, IOptimizer optimizer,
        int threadCount, bool useThreadPool = false)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
        _threadCount = threadCount;
        _useThreadPool = useThreadPool;

        if (_useThreadPool)
        {
            ThreadPool.GetMaxThreads(out int _, out int maxIo);
            ThreadPool.SetMaxThreads(_threadCount, maxIo);
            ThreadPool.GetMinThreads(out int _, out int minIo);
            ThreadPool.SetMinThreads(_threadCount, minIo);
            _poolCountdown = new CountdownEvent(_threadCount);
        }
    }

    private void EnsureInitialized(Network network)
    {
        if (_networkClones != null) return;

        int layerCount = network.Layers.Count;

        _networkClones = new Network[_threadCount];
        _accumulated = new GradientPacket[layerCount];
        _threadLosses = new double[_threadCount * CacheLineDoubles];
        _chunks = new ArraySegment<DataSample>[ChunkCount];

        for (int t = 0; t < _threadCount; t++)
            _networkClones[t] = network.Clone();

        for (int l = 0; l < layerCount; l++)
            _accumulated[l] = network.Layers[l].CreateEmptyGradients();

        if (!_useThreadPool)
        {
            InitPersistentThreads();
        }
    }

    private void InitPersistentThreads()
    {
        _threadWork = new Action[_threadCount];
        _startBarrier = new Barrier(_threadCount + 1);
        _endBarrier = new Barrier(_threadCount + 1);
        _persistentThreads = new Thread[_threadCount];

        for (int t = 0; t < _threadCount; t++)
        {
            int idx = t;
            _persistentThreads[t] = new Thread(() =>
            {
                while (true)
                {
                    _startBarrier!.SignalAndWait();
                    try
                    {
                        _threadWork![idx]();
                    }
                    finally
                    {
                        _endBarrier!.SignalAndWait();
                    }
                }
            }) { IsBackground = true };
            _persistentThreads[t].Start();
        }
    }

    private void Dispatch()
    {
        _startBarrier!.SignalAndWait();
        _endBarrier!.SignalAndWait();
    }

    private static void SyncWeights(Network source, Network target)
    {
        for (int i = 0; i < source.Layers.Count; i++)
            target.Layers[i].SyncFrom(source.Layers[i]);
    }

    public double RunEpoch(Network network, DataSample[] batch, double learningRate)
    {
        EnsureInitialized(network);

        int layerCount = network.Layers.Count;
        int actualChunks = FillChunks(batch);
        int workerCount;

        if (_useThreadPool)
        {
            SyncWeightsWithThreadPool(network);
            RunWithThreadPool(_networkClones!, _chunks!, _threadLosses!, actualChunks);
            workerCount = _threadCount;
        }
        else
        {
            SyncWeightsWithThreads(network, actualChunks);
            RunWithThreads(_networkClones!, _chunks!, _threadLosses!, actualChunks);
            workerCount = actualChunks;
        }

        ReduceAndUpdate(network, workerCount, layerCount, batch.Length, learningRate);

        double totalLoss = 0.0;
        for (int t = 0; t < workerCount; t++) totalLoss += _threadLosses![t * CacheLineDoubles];

        return totalLoss / batch.Length;
    }

    private int FillChunks(DataSample[] batch)
    {
        int actualCount = Math.Min(ChunkCount, batch.Length);
        int baseSize = batch.Length / actualCount;
        int remainder = batch.Length % actualCount;
        int offset = 0;

        for (int i = 0; i < actualCount; i++)
        {
            int size = baseSize + (i < remainder ? 1 : 0);
            _chunks![i] = new ArraySegment<DataSample>(batch, offset, size);
            offset += size;
        }

        return actualCount;
    }

    private void SyncWeightsWithThreads(Network network, int actualThreads)
    {
        for (int t = 0; t < _threadCount; t++)
        {
            int idx = t;
            _threadWork![t] = t < actualThreads
                ? () => SyncWeights(network, _networkClones![idx])
                : static () => { };
        }
        Dispatch();
    }

    private void SyncWeightsWithThreadPool(Network network)
    {
        _poolCountdown!.Reset(_threadCount);
        for (int t = 0; t < _threadCount; t++)
        {
            int idx = t;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                SyncWeights(network, _networkClones![idx]);
                _poolCountdown!.Signal();
            });
        }
        _poolCountdown!.Wait();
    }

    private void RunWithThreads(
        Network[] networkClones,
        ArraySegment<DataSample>[] chunks,
        double[] threadLosses,
        int actualThreads)
    {
        for (int t = 0; t < _threadCount; t++)
        {
            int idx = t;
            _threadWork![t] = t < actualThreads
                ? () => ProcessChunk(networkClones[idx], chunks[idx], threadLosses, idx)
                : static () => { };
        }
        Dispatch();
    }

    private void RunWithThreadPool(
        Network[] networkClones,
        ArraySegment<DataSample>[] chunks,
        double[] threadLosses,
        int actualChunks)
    {
        _poolCountdown!.Reset(_threadCount);
        _nextChunk = 0;

        for (int w = 0; w < _threadCount; w++)
        {
            int workerIdx = w;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                Network clone = networkClones[workerIdx];
                int layerCount = clone.Layers.Count;
                double totalLoss = 0.0;

                for (int l = 0; l < layerCount; l++)
                    clone.Layers[l].GetGradients().Reset();

                while (true)
                {
                    int chunkIdx = Interlocked.Increment(ref _nextChunk) - 1;
                    if (chunkIdx >= actualChunks) break;

                    foreach (var sample in chunks[chunkIdx])
                    {
                        double[] predicted = clone.Forward(sample.Input);
                        totalLoss += _lossFunction.Compute(predicted, sample.Label);
                        clone.Backward(_lossFunction.Gradient(predicted, sample.Label));
                    }
                }

                threadLosses[workerIdx * CacheLineDoubles] = totalLoss;
                _poolCountdown!.Signal();
            });
        }

        _poolCountdown!.Wait();
    }

    private void ReduceAndUpdate(Network network, int actualThreads, int layerCount, int batchSize, double learningRate)
    {
        double invBatch = 1.0 / batchSize;

        for (int l = 0; l < layerCount; l++)
        {
            var acc = _accumulated![l];
            int wCount = acc.WeightGradients.Length;

            if (wCount >= ParallelReduceThreshold)
            {
                int rawRange = (wCount + _threadCount - 1) / _threadCount;
                int rangeSize = (rawRange + CacheLineDoubles - 1) / CacheLineDoubles * CacheLineDoubles;

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
        for (int r = 0; r < _threadCount; r++)
        {
            int rf = r * rangeSize;
            int rt = Math.Min(rf + rangeSize, wCount);
            _threadWork![r] = rf < wCount
                ? () =>
                {
                    Array.Clear(acc.WeightGradients, rf, rt - rf);
                    for (int ti = 0; ti < actualThreads; ti++)
                        acc.AccumulateRange(_networkClones![ti].Layers[layerIdx].GetGradients(), rf, rt);
                    acc.ScaleRange(invBatch, rf, rt);
                }
                : static () => { };
        }
        Dispatch();
    }

    private void ReduceWeightsWithThreadPool(
        GradientPacket acc, int actualThreads, int layerIdx,
        int wCount, int rangeSize, double invBatch)
    {
        _poolCountdown!.Reset(_threadCount);

        for (int r = 0; r < _threadCount; r++)
        {
            int rf = r * rangeSize;
            if (rf >= wCount) { _poolCountdown!.Signal(); continue; }
            int rt = Math.Min(rf + rangeSize, wCount);

            ThreadPool.QueueUserWorkItem(_ =>
            {
                Array.Clear(acc.WeightGradients, rf, rt - rf);
                for (int ti = 0; ti < actualThreads; ti++)
                    acc.AccumulateRange(_networkClones![ti].Layers[layerIdx].GetGradients(), rf, rt);
                acc.ScaleRange(invBatch, rf, rt);
                _poolCountdown!.Signal();
            });
        }

        _poolCountdown!.Wait();
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

        threadLosses[threadIdx * CacheLineDoubles] = totalLoss;
    }
}
