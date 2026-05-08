using System.Diagnostics;
using System.Globalization;
using SupervisedLearning.Benchmark;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Activations;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Core.Loss;
using SupervisedLearning.Data;
using SupervisedLearning.Data.Preprocessing;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Optimizers;
using SupervisedLearning.Training.Strategies;
using SupervisedLearning.Evaluation;
using SupervisedLearning.Inference;
using SupervisedLearning.Verification;

// ── Paths ─────────────────────────────────────────────────────────────────
const string ArticlesDir  = "datasets/train-articles";
const string LabelsPath          = "datasets/train-task2-TC.labels";
const string SyntheticLabelsPath = "datasets/synthetic-task2-TC.labels";
const string SeqCachePath        = "seq_cache.txt";
const string ModelPath           = "model.bin";

// ── Architecture ──────────────────────────────────────────────────────────
const int    VocabSize       = 5000;
const int    SeqLen          = 32;
const int    EmbDim          = 32;
const int    FilterSize      = 3;
const int    NumFilters      = 64;
const int    DenseHidden     = 32;
const int    NumClasses      = 9;
const int    OutputLen       = SeqLen - FilterSize + 1;

// ── Hyperparameters ───────────────────────────────────────────────────────
const int    TrainEpochs     = 200;
const int    BenchEpochs     = 2;
const int    BatchSize       = 256;
const double LearningRate    = 0.04;
const int    ParallelThreads = 6;
int[]        LrDecaySteps    = Array.Empty<int>();
const double LrDecayFactor   = 0.5;
const int    WarmupSamples   = 200;
const int    WarmupEpochs    = 2;
const double DropoutRate     = 0.0;

// ── Section switches ──────────────────────────────────────────────────────
bool UseSyntheticData      = true;
bool RunSequential         = true;
bool LoadSeqFromCache      = false;
bool SaveSeqToCache        = true;
bool LoadModelWeights      = false;
bool SaveModelWeights      = true;
bool RunParallelThread     = true;
bool RunParallelPool       = true;
bool RunParallelPersistent = true;
bool RunAccuracy           = true;
bool RunVerification       = true;
bool RunReport             = true;
bool RunGradientCheck      = false;
bool RunScalability        = true;
bool RunDataSizeSweep      = true;
bool RunTextScan           = false;
bool RunWarmupBeforeTrain  = true;
bool RunWarmupBeforeBench  = true;
bool WarmupForwardOnly     = false;

// ── Load data ─────────────────────────────────────────────────────────────
Console.WriteLine("=== Loading SemEval-2020 Task 11 ===");
var loadTimer = Stopwatch.StartNew();

Console.Write("  Building vocabulary... ");
var vocab = SemEvalLoader.BuildVocabulary(ArticlesDir, VocabSize);
Console.WriteLine($"{vocab.Size} tokens");

Console.Write("  Loading samples... ");
string? syntheticPath = UseSyntheticData && File.Exists(SyntheticLabelsPath) ? SyntheticLabelsPath : null;
var (trainSet, valSet) = SemEvalLoader.LoadWithSplit(
    ArticlesDir, LabelsPath, vocab, SeqLen,
    totalNotPropSamples: 2000, trainRatio: 0.8, seed: 67,
    extraLabelsPath: syntheticPath);
loadTimer.Stop();

double[] classWeights = SemEvalLoader.ComputeClassWeights(trainSet);
Console.WriteLine($"done ({loadTimer.ElapsedMilliseconds} ms)");
Console.WriteLine($"  Train: {trainSet.Samples.Length}  Val: {valSet.Samples.Length}");

// ── Helpers ───────────────────────────────────────────────────────────────
Network BuildNet(int seed = 0, double dropoutRate = DropoutRate)
{
    var n = new Network();
    n.AddLayer(new EmbeddingLayer(vocab.Size, EmbDim, SeqLen, seed));
    n.AddLayer(new Conv1DLayer(SeqLen, EmbDim, FilterSize, NumFilters, new ReLU(), seed + 1));
    n.AddLayer(new MaxPool1DLayer(OutputLen, NumFilters));
    n.AddLayer(new DenseLayer(NumFilters, DenseHidden, new ReLU(), seed + 2, dropoutRate));
    n.AddLayer(new DenseLayer(DenseHidden, NumClasses, new Softmax(), seed + 3));
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
        if (ArgMax(net.Forward(s.Input)) == ArgMax(s.Label)) correct++;
    return (double)correct / samples.Length;
}

void PrintEpochTable(TrainingResult r)
{
    Console.WriteLine($"  {"Ep",3} | {"Loss",10} | {"Time (ms)",10}");
    Console.WriteLine($"  {new string('-', 30)}");
    double prevLr = r.EpochResults.Length > 0 ? r.EpochResults[0].LearningRate : 0;
    for (int i = 0; i < r.EpochResults.Length; i++)
    {
        double lr = r.EpochResults[i].LearningRate;
        if (i > 0 && Math.Abs(lr - prevLr) > 1e-12)
            Console.WriteLine($"  ** LR decay: {prevLr:G4} → {lr:G4} **");
        Console.WriteLine($"  {i + 1,3} | {r.EpochResults[i].Loss,10:F6} | {r.EpochResults[i].EpochDurationMs,10}");
        prevLr = lr;
    }
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

var loss = new CrossEntropyLoss(classWeights);
var sgd  = new SGDOptimizer();

void RunWarmup(string label, bool forwardOnly)
{
    Console.Write($"  [warmup: {label}]... ");
    var sw = Stopwatch.StartNew();
    var warmupNet = BuildNet(seed: 0);
    int n = Math.Min(WarmupSamples, trainSet.Samples.Length);
    var warmupData = new DataSet(trainSet.Samples[..n]);
    if (forwardOnly)
    {
        for (int w = 0; w < WarmupEpochs; w++)
            foreach (var s in warmupData.Samples)
                warmupNet.Forward(s.Input);
    }
    else
    {
        var warmupCfg = new TrainingConfig
        {
            Epochs = WarmupEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 0
        };
        new Trainer(new SequentialStrategy(loss, sgd), warmupCfg).Train(warmupNet, warmupData);
    }
    sw.Stop();
    Console.WriteLine($"done ({sw.ElapsedMilliseconds} ms)");
}

var trainCfg = new TrainingConfig
{
    Epochs = TrainEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42,
    LrDecaySteps = LrDecaySteps, LrDecayFactor = LrDecayFactor
};

Console.WriteLine($"\n=== Sequential vs Parallel ({trainSet.Samples.Length} samples, {TrainEpochs} epochs) ===");
Console.WriteLine($"    1D-CNN: Emb({vocab.Size},{EmbDim}) → Conv({FilterSize}×{NumFilters}) → MaxPool → Dense({DenseHidden}) → {NumClasses}");

if (RunWarmupBeforeTrain)
    RunWarmup("JIT pre-train", WarmupForwardOnly);

// ── 1. Sequential ─────────────────────────────────────────────────────────
TrainingResult? seqResult = null;
Network? seqNet = null;

if (LoadModelWeights && File.Exists(ModelPath))
{
    Console.WriteLine($"\n  [Sequential — weights loaded from {ModelPath}]");
    seqNet = BuildNet(seed: 42);
    NetworkSerializer.Load(seqNet, ModelPath);
}

if (seqNet == null && LoadSeqFromCache)
{
    Console.WriteLine("\n  [Sequential — loaded from cache]");
    seqResult = LoadSeqCache();
    if (seqResult != null) PrintEpochTable(seqResult);
}

if (seqNet == null && seqResult == null && RunSequential)
{
    seqNet = BuildNet(seed: 42);
    Console.WriteLine("\n  [Sequential]");
    seqResult = new Trainer(new SequentialStrategy(loss, sgd), trainCfg).Train(seqNet, trainSet);
    PrintEpochTable(seqResult);
    if (SaveSeqToCache) SaveSeqCache(seqResult);
    if (SaveModelWeights)
    {
        NetworkSerializer.Save(seqNet, ModelPath);
        Console.WriteLine($"  Model weights saved → {ModelPath}");
    }
}

// ── 2. Parallel — Thread ──────────────────────────────────────────────────
TrainingResult? parResult  = null;
Network?        parNet     = null;

if (RunParallelThread)
{
    parNet = BuildNet(seed: 42);
    Console.WriteLine($"\n  [Parallel — {ParallelThreads} threads, Thread-based]");
    parResult = new Trainer(new DataParallelStrategy(loss, sgd, ParallelThreads), trainCfg)
        .Train(parNet, trainSet);
    PrintEpochTable(parResult);
}

// ── 3. Parallel — ThreadPool ──────────────────────────────────────────────
TrainingResult? parResult2 = null;
Network?        parNet2    = null;

if (RunParallelPool)
{
    parNet2 = BuildNet(seed: 42);
    Console.WriteLine($"\n  [Parallel — {ParallelThreads} threads, ThreadPool-based]");
    parResult2 = new Trainer(
        new DataParallelStrategy(loss, sgd, ParallelThreads, useThreadPool: true), trainCfg)
        .Train(parNet2, trainSet);
    PrintEpochTable(parResult2);
}

// ── 3b. Parallel — Persistent Threads ────────────────────────────────────
TrainingResult? parResult3 = null;
Network?        parNet3    = null;

if (RunParallelPersistent)
{
    parNet3 = BuildNet(seed: 42);
    Console.WriteLine($"\n  [Parallel — {ParallelThreads} threads, Persistent Thread-based]");
    using var persistentStrategy = new DataParallelStrategy(
        loss, sgd, ParallelThreads, useThreadPool: false, usePersistentThreads: true);
    parResult3 = new Trainer(persistentStrategy, trainCfg).Train(parNet3, trainSet);
    PrintEpochTable(parResult3);
}

// ── Comparison table ──────────────────────────────────────────────────────
if (seqResult != null && (parResult != null || parResult2 != null || parResult3 != null))
{
    Console.WriteLine("\n  --- Comparison ---");
    Console.WriteLine($"  {"Strategy",-40} | {"Time (ms)",10} | {"Speedup",8} | {"Loss diff",12}");
    Console.WriteLine($"  {new string('-', 80)}");
    Console.WriteLine($"  {"Sequential",-40} | {seqResult.TotalDurationMs,10} |   {"1.00x",6} | {"—",12}");

    if (parResult != null)
    {
        double sp = (double)seqResult.TotalDurationMs / parResult.TotalDurationMs;
        double ld = Math.Abs(seqResult.FinalLoss - parResult.FinalLoss);
        Console.WriteLine($"  {"DataParallel-Thread-" + ParallelThreads,-40} | {parResult.TotalDurationMs,10} | {sp,7:F2}x | {ld,12:E4}");
    }
    if (parResult2 != null)
    {
        double sp = (double)seqResult.TotalDurationMs / parResult2.TotalDurationMs;
        double ld = Math.Abs(seqResult.FinalLoss - parResult2.FinalLoss);
        Console.WriteLine($"  {"DataParallel-ThreadPool-" + ParallelThreads,-40} | {parResult2.TotalDurationMs,10} | {sp,7:F2}x | {ld,12:E4}");
    }
    if (parResult3 != null)
    {
        double sp = (double)seqResult.TotalDurationMs / parResult3.TotalDurationMs;
        double ld = Math.Abs(seqResult.FinalLoss - parResult3.FinalLoss);
        Console.WriteLine($"  {"DataParallel-Thread-" + ParallelThreads + "-Persistent",-40} | {parResult3.TotalDurationMs,10} | {sp,7:F2}x | {ld,12:E4}");
    }
}

// ── 4. Accuracy ───────────────────────────────────────────────────────────
if (RunAccuracy && (seqNet != null || parNet != null || parNet2 != null || parNet3 != null))
{
    Console.WriteLine($"\n=== Validation Accuracy ({valSet.Samples.Length} samples) ===");
    if (seqNet  != null) Console.WriteLine($"  Sequential:                      {ComputeAccuracy(seqNet,  valSet.Samples):P2}");
    if (parNet  != null) Console.WriteLine($"  Parallel (Thread):               {ComputeAccuracy(parNet,  valSet.Samples):P2}");
    if (parNet2 != null) Console.WriteLine($"  Parallel (ThreadPool):           {ComputeAccuracy(parNet2, valSet.Samples):P2}");
    if (parNet3 != null) Console.WriteLine($"  Parallel (Persistent Threads):   {ComputeAccuracy(parNet3, valSet.Samples):P2}");
}

// ── 4b. Classification report ─────────────────────────────────────────────
if (RunReport && seqNet != null)
    new ClassificationReport(seqNet, valSet.Samples, SemEvalLoader.Techniques).Print();

// ── 5. Correctness verification ───────────────────────────────────────────
if (RunVerification)
{
    int verifySamples = Math.Min(500, trainSet.Samples.Length);
    Console.WriteLine($"\n=== Correctness Verification ({verifySamples} samples) ===");
    var verBase = BuildNet(seed: 7, dropoutRate: 0.0);
    var verifier = new CorrectnessVerifier(
        new SequentialStrategy(loss, sgd),
        new DataParallelStrategy(loss, sgd, ParallelThreads),
        batchSize: BatchSize, learningRate: LearningRate, epsilon: 1e-7);
    double maxDiff = verifier.MaxWeightDiff(
        verBase.Clone(), verBase.Clone(),
        new DataSet(trainSet.Samples[..verifySamples]), seed: 42);
    Console.WriteLine($"  Max weight diff (Sequential vs Parallel): {maxDiff:E4}  [{(maxDiff <= 1e-7 ? "PASSED" : "FAILED")}]");
}

// ── 5b. Gradient check ────────────────────────────────────────────────────
if (RunGradientCheck)
{
    Console.WriteLine($"\n=== Gradient Check (1 sample, all layer types) ===");
    var gcNet = BuildNet(seed: 99, dropoutRate: 0.0);
    var gcSample = trainSet.Samples[0];
    var checker = new GradientChecker(epsilon: 1e-5, tolerance: 1e-4);
    double maxErr = checker.ComputeMaxRelativeError(gcNet, loss, gcSample);
    Console.WriteLine($"  Max relative error (all layers): {maxErr:E4}  [{(maxErr <= 1e-4 ? "PASSED" : "FAILED")}]");
}

if (RunWarmupBeforeBench && (RunScalability || RunDataSizeSweep))
    RunWarmup("JIT pre-bench", WarmupForwardOnly);

// ── 6. Scalability sweep ──────────────────────────────────────────────────
if (RunScalability)
{
    Console.WriteLine($"\n=== Scalability: Thread Count ({trainSet.Samples.Length} samples, {BenchEpochs} epochs) ===");
    var benchCfg = new TrainingConfig
    {
        Epochs = BenchEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42
    };
    new BenchmarkRunner(loss, sgd)
        .ScalabilitySweep(BuildNet(seed: 0), trainSet, benchCfg, new[] { 2, 4, 6, 8 })
        .Print();
}

// ── 7. Dataset size sweep ─────────────────────────────────────────────────
if (RunDataSizeSweep)
{
    int[] dataSizes = { 500, 1000, 2000, Math.Min(4000, trainSet.Samples.Length), trainSet.Samples.Length };
    dataSizes = dataSizes.Distinct().OrderBy(x => x).ToArray();
    Console.WriteLine($"\n=== Scalability: Dataset Size ({ParallelThreads} threads, {BenchEpochs} epochs) ===");
    var benchCfg2 = new TrainingConfig
    {
        Epochs = BenchEpochs, BatchSize = BatchSize, LearningRate = LearningRate, Seed = 42
    };
    new BenchmarkRunner(loss, sgd)
        .DataSizeSweep(BuildNet(seed: 0), trainSet, benchCfg2, dataSizes, ParallelThreads)
        .Print();
}

// ── 8. Text scanning ──────────────────────────────────────────────────────
if (RunTextScan && (seqNet != null || parNet != null || parNet2 != null || parNet3 != null))
{
    Console.WriteLine($"\n=== Text Scanning ===");
    var net = seqNet ?? parNet ?? parNet2 ?? parNet3!;
    var scanner = new TextScanner(net, vocab, SeqLen);

    string[] articleFiles = Directory.GetFiles(ArticlesDir, "article*.txt");
    if (articleFiles.Length > 0)
    {
        Array.Sort(articleFiles);
        string articleText = File.ReadAllText(articleFiles[0]);
        string articleName = Path.GetFileNameWithoutExtension(articleFiles[0]);
        var spans = scanner.Scan(articleText, stride: 16, minConfidence: 0.5);
        Console.WriteLine($"  {articleName}: {spans.Count} spans detected");
        int show = Math.Min(spans.Count, 10);
        for (int i = 0; i < show; i++)
        {
            var s = spans[i];
            Console.WriteLine($"    [{s.StartChar,6}–{s.EndChar,6}] {s.Technique,-35} ({s.Confidence:P1})");
        }
        if (spans.Count > 10)
            Console.WriteLine($"    ... and {spans.Count - 10} more");
    }
}
