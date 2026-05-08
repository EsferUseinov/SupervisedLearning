namespace SupervisedLearning.Training.Strategies;

using System.Diagnostics;
using System.Threading;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Data;

public class DataParallelStrategy : ITrainingStrategy, IDisposable
{
    private readonly ILossFunction _lossFunction;
    private readonly IOptimizer _optimizer;
    private readonly int _threadCount;
    private readonly bool _useThreadPool;
    private readonly bool _usePersistentThreads;

    private Network[]? _networkClones;
    private GradientPacket[][]? _threadGradients;
    private GradientPacket[]? _accumulated;
    private double[]? _threadLosses;

    private SemaphoreSlim[]? _startSignals;
    private SemaphoreSlim? _doneSignal;
    private volatile bool _stopping;
    private DataSample[][]? _currentChunks;

    public DataParallelStrategy(ILossFunction lossFunction, IOptimizer optimizer,
        int threadCount, bool useThreadPool = false, bool usePersistentThreads = false)
    {
        _lossFunction = lossFunction;
        _optimizer = optimizer;
        _threadCount = threadCount;
        _useThreadPool = useThreadPool;
        _usePersistentThreads = usePersistentThreads && !useThreadPool;
    }

    public string Name => _usePersistentThreads
        ? $"DataParallel-Thread-{_threadCount}-Persistent"
        : _useThreadPool
            ? $"DataParallel-ThreadPool-{_threadCount}"
            : $"DataParallel-Thread-{_threadCount}";

    private void EnsureInitialized(Network network)
    {
        if (_networkClones != null) return;

        int layerCount = network.Layers.Count;

        _networkClones = new Network[_threadCount];
        _threadGradients = new GradientPacket[_threadCount][];
        _accumulated = new GradientPacket[layerCount];
        _threadLosses = new double[_threadCount];

        for (int t = 0; t < _threadCount; t++)
        {
            _networkClones[t] = network.Clone();
            _networkClones[t].SetTrainingMode(true);
            _threadGradients[t] = new GradientPacket[layerCount];
            for (int l = 0; l < layerCount; l++)
                _threadGradients[t][l] = network.Layers[l].CreateEmptyGradients();
        }

        for (int l = 0; l < layerCount; l++)
            _accumulated[l] = network.Layers[l].CreateEmptyGradients();

        if (_usePersistentThreads)
            StartWorkerThreads();
    }

    private void StartWorkerThreads()
    {
        _startSignals = new SemaphoreSlim[_threadCount];
        _doneSignal = new SemaphoreSlim(0);

        for (int t = 0; t < _threadCount; t++)
        {
            _startSignals[t] = new SemaphoreSlim(0, 1);
            int idx = t;
            var thread = new Thread(() => WorkerLoop(idx));
            thread.IsBackground = true;
            thread.Start();
        }
    }

    private void WorkerLoop(int idx)
    {
        while (true)
        {
            _startSignals![idx].Wait();
            if (_stopping) return;
            ProcessChunk(_networkClones![idx], _currentChunks![idx], _threadGradients![idx], _threadLosses!, idx);
            _doneSignal!.Release();
        }
    }

    private static void SyncWeights(Network source, Network target)
    {
        for (int i = 0; i < source.Layers.Count; i++)
            target.Layers[i].SyncFrom(source.Layers[i]);
    }

    public EpochResult RunEpoch(Network network, DataSample[] batch, double learningRate)
    {
        var epochTimer = Stopwatch.StartNew();

        EnsureInitialized(network);

        int layerCount = network.Layers.Count;
        DataSample[][] chunks = SplitIntoChunks(batch, _threadCount);
        int actualThreads = chunks.Length;

        for (int t = 0; t < actualThreads; t++)
        {
            SyncWeights(network, _networkClones![t]);
            for (int l = 0; l < layerCount; l++)
                _threadGradients![t][l].Reset();
        }

        if (_usePersistentThreads)
            RunWithPersistentThreads(chunks, actualThreads);
        else if (_useThreadPool)
            RunWithThreadPool(_networkClones!, chunks, _threadGradients!, _threadLosses!, actualThreads);
        else
            RunWithThreads(_networkClones!, chunks, _threadGradients!, _threadLosses!, actualThreads);

        var syncTimer = Stopwatch.StartNew();

        for (int l = 0; l < layerCount; l++)
        {
            _accumulated![l].Reset();
            for (int t = 0; t < actualThreads; t++)
                _accumulated[l].Accumulate(_threadGradients![t][l]);
        }

        syncTimer.Stop();

        for (int l = 0; l < layerCount; l++)
        {
            _accumulated![l].Scale(1.0 / batch.Length);
            _optimizer.UpdateWeights(network.Layers[l], _accumulated[l], learningRate);
        }

        epochTimer.Stop();

        double totalLoss = 0.0;
        for (int t = 0; t < actualThreads; t++) totalLoss += _threadLosses![t];

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

    private void RunWithPersistentThreads(DataSample[][] chunks, int actualThreads)
    {
        _currentChunks = chunks;
        for (int t = 0; t < actualThreads; t++)
            _startSignals![t].Release();
        for (int t = 0; t < actualThreads; t++)
            _doneSignal!.Wait();
    }

    private void RunWithThreads(
        Network[] networkClones,
        DataSample[][] chunks,
        GradientPacket[][] threadGradients,
        double[] threadLosses,
        int actualThreads)
    {
        var threads = new Thread[actualThreads];
        for (int t = 0; t < actualThreads; t++)
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
        double[] threadLosses,
        int actualThreads)
    {
        using var countdown = new CountdownEvent(actualThreads);
        for (int t = 0; t < actualThreads; t++)
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
        double totalLoss = 0.0;

        for (int l = 0; l < layerCount; l++)
            networkClone.Layers[l].GetGradients().Reset();

        foreach (var sample in chunk)
        {
            double[] predicted = networkClone.Forward(sample.Input);
            totalLoss += _lossFunction.Compute(predicted, sample.Label);
            networkClone.Backward(_lossFunction.Gradient(predicted, sample.Label));
        }

        for (int l = 0; l < layerCount; l++)
            layerGradients[l].Accumulate(networkClone.Layers[l].GetGradients());

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

    public void Dispose()
    {
        if (_startSignals == null || _stopping) return;
        _stopping = true;
        foreach (var sig in _startSignals) sig.Release();
    }
}
