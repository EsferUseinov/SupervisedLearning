using System.Diagnostics;
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

const string TrainPath    = "mnist_train.csv";
const string TestPath     = "mnist_test.csv";
const int    TrainSamples = 60_000;
const int    BenchSamples = 3_000;
const int    TrainEpochs  = 10;
const int    BenchEpochs  = 2;

// ── Load Data ─────────────────────────────────────────────────────────────

Console.WriteLine("=== Loading MNIST ===");
var loadTimer = Stopwatch.StartNew();
var fullTrain = DataLoader.LoadFromCsv(TrainPath);
var testSet   = DataLoader.LoadFromCsv(TestPath);
loadTimer.Stop();
Console.WriteLine($"Train: {fullTrain.Samples.Length} samples  Test: {testSet.Samples.Length} samples  ({loadTimer.ElapsedMilliseconds} ms)");

var trainSet = new DataSet(fullTrain.Samples[..TrainSamples]);

// ── Helpers ───────────────────────────────────────────────────────────────

Network BuildNet(int seed = 0)
{
    var n = new Network();
    n.AddLayer(new DenseLayer(784, 256, new ReLU(),    seed));
    n.AddLayer(new DenseLayer(256, 128, new ReLU(),    seed + 1));
    n.AddLayer(new DenseLayer(128,  10, new Sigmoid(), seed + 2));
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

var loss = new CrossEntropyLoss();
var sgd  = new SGDOptimizer();

var trainCfg = new TrainingConfig
{
    Epochs = TrainEpochs, BatchSize = 64, LearningRate = 0.01, Seed = 42
};

// ── 1. Sequential vs Parallel Training ───────────────────────────────────

Console.WriteLine($"\n=== Sequential vs Parallel ({TrainSamples} samples, {TrainEpochs} epochs) ===");
Console.WriteLine($"    Network: 784 -> 256 -> 128 -> 10  |  Loss: CrossEntropy  |  Batch: 64");

var baseNet = BuildNet(seed: 42);
var seqNet  = baseNet.Clone();
var parNet  = baseNet.Clone();

Console.WriteLine("\n  [Sequential]");
var seqResult = new Trainer(new SequentialStrategy(loss, sgd), trainCfg).Train(seqNet, trainSet);
PrintEpochTable(seqResult);

Console.WriteLine("\n  [Parallel — 4 threads, Thread-based]");
var parResult = new Trainer(new DataParallelStrategy(loss, sgd, threadCount: 4), trainCfg).Train(parNet, trainSet);
PrintEpochTable(parResult);

Console.WriteLine("\n  [Parallel — 4 threads, ThreadPool-based]");
var parNet2    = baseNet.Clone();
var parResult2 = new Trainer(new DataParallelStrategy(loss, sgd, threadCount: 4, useThreadPool: true), trainCfg).Train(parNet2, trainSet);
PrintEpochTable(parResult2);

Console.WriteLine("\n  --- Comparison ---");
double speedupThread     = (double)seqResult.TotalDurationMs / parResult.TotalDurationMs;
double speedupThreadPool = (double)seqResult.TotalDurationMs / parResult2.TotalDurationMs;
Console.WriteLine($"  {"Strategy",-30} | {"Time (ms)",10} | {"Speedup",8} | {"Loss diff",12}");
Console.WriteLine($"  {new string('-', 70)}");
Console.WriteLine($"  {"Sequential",-30} | {seqResult.TotalDurationMs,10} |   {"1.00x",6} | {"—",12}");
Console.WriteLine($"  {"DataParallel-Thread-4",-30} | {parResult.TotalDurationMs,10} | {speedupThread,7:F2}x | {Math.Abs(seqResult.FinalLoss - parResult.FinalLoss),12:E4}");
Console.WriteLine($"  {"DataParallel-ThreadPool-4",-30} | {parResult2.TotalDurationMs,10} | {speedupThreadPool,7:F2}x | {Math.Abs(seqResult.FinalLoss - parResult2.FinalLoss),12:E4}");

// ── Save Model ───────────────────────────────────────────────────────────

const string ModelPath = "mnist_model.bin";
NetworkSerializer.Save(seqNet, ModelPath);
Console.WriteLine($"\n  Model saved → {ModelPath}");

// ── 2. Accuracy on Test Set ───────────────────────────────────────────────

Console.WriteLine($"\n=== Test Set Accuracy ({testSet.Samples.Length} samples) ===");
Console.WriteLine($"  Sequential:            {ComputeAccuracy(seqNet,  testSet.Samples):P2}");
Console.WriteLine($"  Parallel (Thread):     {ComputeAccuracy(parNet,  testSet.Samples):P2}");
Console.WriteLine($"  Parallel (ThreadPool): {ComputeAccuracy(parNet2, testSet.Samples):P2}");

// ── 3. Correctness Verification ───────────────────────────────────────────

Console.WriteLine("\n=== Correctness Verification (1000 samples) ===");
var verBase  = BuildNet(seed: 7);
var verifier = new CorrectnessVerifier(
    new SequentialStrategy(loss, sgd),
    new DataParallelStrategy(loss, sgd, 4),
    batchSize: 64, learningRate: 0.01, epsilon: 1e-8);
double maxDiff = verifier.MaxWeightDiff(
    verBase.Clone(), verBase.Clone(),
    new DataSet(trainSet.Samples[..1000]), seed: 42);
Console.WriteLine($"  Max weight diff (Sequential vs Parallel): {maxDiff:E4}  [{(maxDiff <= 1e-8 ? "PASSED" : "FAILED")}]");

// ── 4. Scalability: Thread Count ─────────────────────────────────────────

Console.WriteLine($"\n=== Scalability: Thread Count ({BenchSamples} samples, {BenchEpochs} epochs) ===");
var benchSet = new DataSet(fullTrain.Samples[..BenchSamples]);
var benchCfg = new TrainingConfig { Epochs = BenchEpochs, BatchSize = 64, LearningRate = 0.01, Seed = 42 };

new BenchmarkRunner(loss, sgd)
    .ScalabilitySweep(BuildNet(seed: 0), benchSet, benchCfg, new[] { 2, 4, 6, 8 })
    .Print();

// ── 5. Dataset Size vs Speedup ────────────────────────────────────────────

Console.WriteLine("\n=== Dataset Size vs Speedup (4 threads, 2 epochs) ===");
var sizeCfg = new TrainingConfig { Epochs = 2, BatchSize = 64, LearningRate = 0.01, Seed = 42 };
int[] sizes = { 1_000, 3_000, 5_000, 10_000 };

Console.WriteLine($"  {"Samples",10} | {"Sequential (ms)",15} | {"Parallel (ms)",14} | {"Speedup",8}");
Console.WriteLine($"  {new string('-', 58)}");
foreach (int n in sizes)
{
    var sub  = new DataSet(fullTrain.Samples[..n]);
    long sMs = new Trainer(new SequentialStrategy(loss, sgd), sizeCfg)
                   .Train(BuildNet(seed: 0), sub).TotalDurationMs;
    long pMs = new Trainer(new DataParallelStrategy(loss, sgd, 4), sizeCfg)
                   .Train(BuildNet(seed: 0), sub).TotalDurationMs;
    Console.WriteLine($"  {n,10} | {sMs,15} | {pMs,14} | {(double)sMs / pMs,8:F2}x");
}
