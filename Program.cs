using System.Diagnostics;
using System.Globalization;
using SupervisedLearning.Benchmark;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Activations;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Core.Loss;
using SupervisedLearning.Data;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Optimizers;
using SupervisedLearning.Training.Strategies;
using SupervisedLearning.Verification;

// ── Paths ─────────────────────────────────────────────────────────────────
const string TrainPath    = "mnist_train.csv";
const string TestPath     = "mnist_test.csv";
const string ModelPath    = "mnist_model.bin";
const string SeqCachePath = "seq_cache.txt";

// ── Hyperparameters ───────────────────────────────────────────────────────
const int    TrainSamples    = 60_000;
const int    BenchSamples    = 3_000;
const int    TrainEpochs     = 10;
const int    BenchEpochs     = 2;
const int    BatchSize       = 64;
const double LearningRate    = 0.01;
const int    ParallelThreads = 4;

// ── Section switches ──────────────────────────────────────────────────────
//
//  LoadSeqFromCache = true:  пропустити послідовне навчання, завантажити
//                            результат із seq_cache.txt та модель із mnist_model.bin
//  SaveSeqToCache = true:    зберегти результат після послідовного навчання
//
//  Типові сценарії:
//    Тільки паралельне:   RunSequential = false, LoadSeqFromCache = true
//    Тільки scalability:  RunSequential = false, RunParallel* = false,
//                         RunScalability = true, решта = false
//
bool RunSequential     = false;
bool SaveSeqToCache    = false;
bool LoadSeqFromCache  = true;
bool RunParallelThread = true;
bool RunParallelPool   = true;
bool RunAccuracy       = true;
bool RunVerification   = true;
bool RunScalability    = true;
bool RunSizeVsSpeedup  = true;

// ── Load Data ─────────────────────────────────────────────────────────────
Console.WriteLine("=== Loading MNIST ===");
var loadTimer = Stopwatch.StartNew();
var fullTrain = DataLoader.LoadFromCsv(TrainPath);
var testSet = DataLoader.LoadFromCsv(TestPath);
loadTimer.Stop();
Console.WriteLine($"Train: {fullTrain.Samples.Length} samples  " +
                  $"Test: {testSet.Samples.Length} samples  ({loadTimer.ElapsedMilliseconds} ms)");

var trainSet = new DataSet(fullTrain.Samples[..TrainSamples]);

// ── Helpers ───────────────────────────────────────────────────────────────
Network BuildNet(int seed = 0)
{
    var n = new Network();
    n.AddLayer(new DenseLayer(784, 256, new ReLU(), seed));
    n.AddLayer(new DenseLayer(256, 128, new ReLU(), seed + 1));
    n.AddLayer(new DenseLayer(128, 10, new Sigmoid(), seed + 2));
    return n;
}

int ArgMax(double[] v)
{
    int idx = 0;
    for (int i = 1; i < v.Length; i++)
        if (v[i] > v[idx]) idx = i;
    return idx;
}

double ComputeAccuracy(Network net, DataSample[] samples)
{
    int correct = 0;
    foreach (var s in samples)
        if (ArgMax(net.Forward(s.Input)) == ArgMax(s.Label))
            correct++;
    return (double)correct / samples.Length;
}

void PrintEpochTable(TrainingResult r)
{
    Console.WriteLine($"  {"Ep",3} | {"Loss",10} | {"Time (ms)",10}");
    Console.WriteLine($"  {new string('-', 30)}");
    for (int i = 0; i < r.EpochResults.Length; i++)
        Console.WriteLine($"  {i + 1,3} | {r.EpochResults[i].Loss,10:F6} | {r.EpochResults[i].EpochDurationMs,10}");
    Console.WriteLine($"  Total: {r.TotalDurationMs} ms  Final loss: {r.FinalLoss:F6}");
}

TrainingResult? LoadSeqCache()
{
    if (!File.Exists(SeqCachePath))
    {
        Console.WriteLine($"  [cache] File not found: {SeqCachePath}");
        return null;
    }
    var lines = File.ReadAllLines(SeqCachePath);
    long totalMs = long.Parse(lines[0]);
    double finalLoss = double.Parse(lines[1], CultureInfo.InvariantCulture);
    int count = int.Parse(lines[2]);
    var epochs = new EpochResult[count];
    for (int i = 0; i < count; i++)
    {
        var p = lines[3 + i].Split(',');
        epochs[i] = new EpochResult
        {
            Loss = double.Parse(p[0], CultureInfo.InvariantCulture),
            EpochDurationMs = long.Parse(p[1])
        };
    }
    return new TrainingResult
    {
        TotalDurationMs = totalMs,
        FinalLoss = finalLoss,
        EpochResults = epochs,
        StrategyName = "Sequential (cached)",
        LossCurve = Array.ConvertAll(epochs, e => e.Loss)
    };
}

void SaveSeqCache(TrainingResult r)
{
    var lines = new List<string>
    {
        r.TotalDurationMs.ToString(),
        r.FinalLoss.ToString("R", CultureInfo.InvariantCulture),
        r.EpochResults.Length.ToString()
    };
    foreach (var e in r.EpochResults)
        lines.Add($"{e.Loss.ToString("R", CultureInfo.InvariantCulture)},{e.EpochDurationMs}");
    File.WriteAllLines(SeqCachePath, lines);
    Console.WriteLine($"  Seq result cached → {SeqCachePath}");
}

var loss = new CrossEntropyLoss();
var sgd = new SGDOptimizer();

var trainCfg = new TrainingConfig
{
    Epochs = TrainEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42
};

// ── 1. Sequential ─────────────────────────────────────────────────────────
TrainingResult? seqResult = null;
Network? seqNet = null;

Console.WriteLine($"\n=== Sequential vs Parallel ({TrainSamples} samples, {TrainEpochs} epochs) ===");
Console.WriteLine($"    Network: 784 -> 256 -> 128 -> 10  |  Loss: CrossEntropy  |  Batch: {BatchSize}");

if (LoadSeqFromCache)
{
    Console.WriteLine("\n  [Sequential — loaded from cache]");
    seqResult = LoadSeqCache();
    if (seqResult != null && File.Exists(ModelPath))
    {
        seqNet = BuildNet(seed: 42);
        NetworkSerializer.Load(seqNet, ModelPath);
        PrintEpochTable(seqResult);
    }
    else if (seqResult != null)
    {
        Console.WriteLine($"  [cache] Model file not found: {ModelPath}  (accuracy skipped for Sequential)");
    }
}
else if (RunSequential)
{
    seqNet = BuildNet(seed: 42);
    Console.WriteLine("\n  [Sequential]");
    seqResult = new Trainer(new SequentialStrategy(loss, sgd), trainCfg).Train(seqNet, trainSet);
    PrintEpochTable(seqResult);

    if (SaveSeqToCache) SaveSeqCache(seqResult);

    NetworkSerializer.Save(seqNet, ModelPath);
    Console.WriteLine($"  Model saved → {ModelPath}");
}

// ── 2. Parallel — Thread ──────────────────────────────────────────────────
TrainingResult? parResult = null;
Network? parNet = null;

if (RunParallelThread)
{
    parNet = BuildNet(seed: 42);
    Console.WriteLine($"\n  [Parallel — {ParallelThreads} threads, Thread-based]");
    parResult = new Trainer(
        new DataParallelStrategy(loss, sgd, threadCount: ParallelThreads), trainCfg)
        .Train(parNet, trainSet);
    PrintEpochTable(parResult);
}

// ── 3. Parallel — ThreadPool ──────────────────────────────────────────────
TrainingResult? parResult2 = null;
Network? parNet2 = null;

if (RunParallelPool)
{
    parNet2 = BuildNet(seed: 42);
    Console.WriteLine($"\n  [Parallel — {ParallelThreads} threads, ThreadPool-based]");
    parResult2 = new Trainer(
        new DataParallelStrategy(loss, sgd, threadCount: ParallelThreads, useThreadPool: true), trainCfg)
        .Train(parNet2, trainSet);
    PrintEpochTable(parResult2);
}

// ── Comparison table ──────────────────────────────────────────────────────
if (seqResult != null && (parResult != null || parResult2 != null))
{
    Console.WriteLine("\n  --- Comparison ---");
    Console.WriteLine($"  {"Strategy",-30} | {"Time (ms)",10} | {"Speedup",8} | {"Loss diff",12}");
    Console.WriteLine($"  {new string('-', 70)}");
    Console.WriteLine($"  {"Sequential",-30} | {seqResult.TotalDurationMs,10} |   {"1.00x",6} | {"—",12}");

    if (parResult != null)
    {
        double sp = (double)seqResult.TotalDurationMs / parResult.TotalDurationMs;
        double ld = Math.Abs(seqResult.FinalLoss - parResult.FinalLoss);
        Console.WriteLine($"  {"DataParallel-Thread-" + ParallelThreads,-30} | {parResult.TotalDurationMs,10} | {sp,7:F2}x | {ld,12:E4}");
    }
    if (parResult2 != null)
    {
        double sp = (double)seqResult.TotalDurationMs / parResult2.TotalDurationMs;
        double ld = Math.Abs(seqResult.FinalLoss - parResult2.FinalLoss);
        Console.WriteLine($"  {"DataParallel-ThreadPool-" + ParallelThreads,-30} | {parResult2.TotalDurationMs,10} | {sp,7:F2}x | {ld,12:E4}");
    }
}

// ── 4. Accuracy ───────────────────────────────────────────────────────────
if (RunAccuracy && (seqNet != null || parNet != null || parNet2 != null))
{
    Console.WriteLine($"\n=== Test Set Accuracy ({testSet.Samples.Length} samples) ===");
    if (seqNet != null) Console.WriteLine($"  Sequential:            {ComputeAccuracy(seqNet, testSet.Samples):P2}");
    if (parNet != null) Console.WriteLine($"  Parallel (Thread):     {ComputeAccuracy(parNet, testSet.Samples):P2}");
    if (parNet2 != null) Console.WriteLine($"  Parallel (ThreadPool): {ComputeAccuracy(parNet2, testSet.Samples):P2}");
}

// ── 5. Correctness verification ───────────────────────────────────────────
if (RunVerification)
{
    Console.WriteLine("\n=== Correctness Verification (1000 samples) ===");
    var verBase = BuildNet(seed: 7);
    var verifier = new CorrectnessVerifier(
        new SequentialStrategy(loss, sgd),
        new DataParallelStrategy(loss, sgd, ParallelThreads),
        batchSize: BatchSize, learningRate: LearningRate, epsilon: 1e-8);
    double maxDiff = verifier.MaxWeightDiff(
        verBase.Clone(), verBase.Clone(),
        new DataSet(trainSet.Samples[..1000]), seed: 42);
    Console.WriteLine($"  Max weight diff (Sequential vs Parallel): {maxDiff:E4}  [{(maxDiff <= 1e-8 ? "PASSED" : "FAILED")}]");
}

// ── 6. Scalability sweep ──────────────────────────────────────────────────
if (RunScalability)
{
    Console.WriteLine($"\n=== Scalability: Thread Count ({BenchSamples} samples, {BenchEpochs} epochs) ===");
    var benchSet = new DataSet(fullTrain.Samples[..BenchSamples]);
    var benchCfg = new TrainingConfig
    {
        Epochs = BenchEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42
    };
    new BenchmarkRunner(loss, sgd)
        .ScalabilitySweep(BuildNet(seed: 0), benchSet, benchCfg, new[] { 2, 4, 6, 8 })
        .Print();
}

// ── 7. Dataset size vs speedup ────────────────────────────────────────────
if (RunSizeVsSpeedup)
{
    Console.WriteLine($"\n=== Dataset Size vs Speedup ({ParallelThreads} threads, 2 epochs) ===");
    var sizeCfg = new TrainingConfig
    {
        Epochs = 2, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42
    };
    int[] sizes = { 1_000, 3_000, 5_000, 10_000 };

    Console.WriteLine($"  {"Samples",10} | {"Sequential (ms)",15} | {"Parallel (ms)",14} | {"Speedup",8}");
    Console.WriteLine($"  {new string('-', 58)}");
    foreach (int n in sizes)
    {
        var sub = new DataSet(fullTrain.Samples[..n]);
        long sMs = new Trainer(new SequentialStrategy(loss, sgd), sizeCfg)
                       .Train(BuildNet(seed: 0), sub).TotalDurationMs;
        long pMs = new Trainer(new DataParallelStrategy(loss, sgd, ParallelThreads), sizeCfg)
                       .Train(BuildNet(seed: 0), sub).TotalDurationMs;
        Console.WriteLine($"  {n,10} | {sMs,15} | {pMs,14} | {(double)sMs / pMs,8:F2}x");
    }
}
