using System.Diagnostics;
using SupervisedLearning.Core;
using SupervisedLearning.Core.Activations;
using SupervisedLearning.Core.Interfaces;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Core.Loss;
using SupervisedLearning.Data;
using SupervisedLearning.Training;
using SupervisedLearning.Training.Optimizers;
using SupervisedLearning.Training.Strategies;
using SupervisedLearning.Verification;

Console.WriteLine("=== Activations ===");

var relu = new ReLU();
Console.WriteLine($"ReLU(-2)      = {relu.Compute(-2.0)}");
Console.WriteLine($"ReLU(0)       = {relu.Compute(0.0)}");
Console.WriteLine($"ReLU(3)       = {relu.Compute(3.0)}");
Console.WriteLine($"ReLU'(-1)     = {relu.Derivative(-1.0)}");
Console.WriteLine($"ReLU'(1)      = {relu.Derivative(1.0)}");

Console.WriteLine();
var sigmoid = new Sigmoid();
Console.WriteLine($"Sigmoid(-inf) ~ {sigmoid.Compute(-10.0):F4}");
Console.WriteLine($"Sigmoid(0)    = {sigmoid.Compute(0.0):F4}");
Console.WriteLine($"Sigmoid(+inf) ~ {sigmoid.Compute(10.0):F4}");
Console.WriteLine($"Sigmoid'(0)   = {sigmoid.Derivative(0.0):F4}");

Console.WriteLine();
var softmax = new Softmax();
double[] logits = { 2.0, 1.0, 0.5 };
double[] probs = softmax.ComputeVector(logits);
Console.WriteLine($"Softmax([2, 1, 0.5]) = [{string.Join(", ", probs.Select(p => p.ToString("F4")))}]");
Console.WriteLine($"Sum of probs         = {probs.Sum():F6}");

Console.WriteLine();
Console.WriteLine("=== GradientPacket ===");

var packet = new GradientPacket(inputSize: 3, outputSize: 2);
packet.WeightGradients[0, 0] = 1.0;
packet.WeightGradients[0, 1] = 2.0;
packet.BiasGradients[0] = 0.5;

var other = new GradientPacket(inputSize: 3, outputSize: 2);
other.WeightGradients[0, 0] = 3.0;
other.WeightGradients[0, 1] = 4.0;
other.BiasGradients[0] = 1.5;

packet.Accumulate(other);
Console.WriteLine($"After Accumulate: W[0,0]={packet.WeightGradients[0, 0]}, W[0,1]={packet.WeightGradients[0, 1]}, b[0]={packet.BiasGradients[0]}");

packet.Scale(0.5);
Console.WriteLine($"After Scale(0.5): W[0,0]={packet.WeightGradients[0, 0]}, W[0,1]={packet.WeightGradients[0, 1]}, b[0]={packet.BiasGradients[0]}");

packet.Reset();
Console.WriteLine($"After Reset:      W[0,0]={packet.WeightGradients[0, 0]}, b[0]={packet.BiasGradients[0]}");

Console.WriteLine();
Console.WriteLine("=== Data ===");

var dataset = DataLoader.GenerateSynthetic(samples: 100, inputSize: 4, outputSize: 2, seed: 42);
Console.WriteLine($"Generated {dataset.Samples.Length} samples, inputSize=4, outputSize=2");
Console.WriteLine($"Sample[0]: input=[{string.Join(", ", dataset.Samples[0].Input.Select(v => v.ToString("F3")))}]");
Console.WriteLine($"          label=[{string.Join(", ", dataset.Samples[0].Label)}]");

dataset.Shuffle(seed: 42);
Console.WriteLine($"After Shuffle: Sample[0] label=[{string.Join(", ", dataset.Samples[0].Label)}]");

var batches = dataset.CreateBatches(batchSize: 32);
Console.WriteLine($"CreateBatches(32): {batches.Length} batches, sizes = [{string.Join(", ", batches.Select(b => b.Length))}]");

var splits = dataset.TrainTestSplit(trainRatio: 0.8);
Console.WriteLine($"TrainTestSplit(0.8): train={splits[0].Samples.Length}, test={splits[1].Samples.Length}");

Console.WriteLine();
Console.WriteLine("=== Network Forward/Backward ===");

var network = new Network();
network.AddLayer(new DenseLayer(inputSize: 2, outputSize: 4, activation: new ReLU(), seed: 1));
network.AddLayer(new DenseLayer(inputSize: 4, outputSize: 2, activation: new Sigmoid(), seed: 2));

double[] input = { 1.0, -0.5 };
double[] target = { 1.0, 0.0 };

double[] predicted = network.Forward(input);
Console.WriteLine($"Forward output:  [{string.Join(", ", predicted.Select(v => v.ToString("F6")))}]");

double mseLoss = predicted.Zip(target, (p, t) => (p - t) * (p - t)).Sum() / 2.0;
Console.WriteLine($"MSE loss:        {mseLoss:F6}");

double[] lossGrad = predicted.Zip(target, (p, t) => p - t).ToArray();
Console.WriteLine($"Loss gradient:   [{string.Join(", ", lossGrad.Select(v => v.ToString("F6")))}]");

network.Backward(lossGrad);

var sgd = new SGDOptimizer();
double learningRate = 0.1;

foreach (var layer in network.Layers)
{
    sgd.UpdateWeights(layer, layer.GetGradients(), learningRate);
    layer.GetGradients().Reset();
}

double[] predictedAfter = network.Forward(input);
double mseLossAfter = predictedAfter.Zip(target, (p, t) => (p - t) * (p - t)).Sum() / 2.0;
Console.WriteLine($"Loss after SGD:  {mseLossAfter:F6}  (should be less than {mseLoss:F6})");

Console.WriteLine();
Console.WriteLine("=== Network Clone ===");

var cloned = network.Clone();
double[] outputOriginal = network.Forward(input);
double[] outputCloned = cloned.Forward(input);
Console.WriteLine($"Original output: [{string.Join(", ", outputOriginal.Select(v => v.ToString("F6")))}]");
Console.WriteLine($"Cloned output:   [{string.Join(", ", outputCloned.Select(v => v.ToString("F6")))}]");

double maxDiff = outputOriginal.Zip(outputCloned, (a, b) => Math.Abs(a - b)).Max();
Console.WriteLine($"Max diff:        {maxDiff:E2}  (should be ~0)");

Console.WriteLine();
Console.WriteLine("=== Training (Sequential, 30 epochs) ===");

var trainDataset = DataLoader.GenerateSynthetic(samples: 200, inputSize: 4, outputSize: 2, seed: 7);

var trainNetwork = new Network();
trainNetwork.AddLayer(new DenseLayer(inputSize: 4, outputSize: 8, activation: new ReLU(), seed: 10));
trainNetwork.AddLayer(new DenseLayer(inputSize: 8, outputSize: 4, activation: new ReLU(), seed: 11));
trainNetwork.AddLayer(new DenseLayer(inputSize: 4, outputSize: 2, activation: new Sigmoid(), seed: 12));

var mse = new MSELoss();
var sgdOptimizer = new SGDOptimizer();
var strategy = new SequentialStrategy(mse, sgdOptimizer);
var config = new TrainingConfig
{
    Epochs = 30,
    BatchSize = 32,
    LearningRate = 0.05,
    Seed = 42
};

var trainer = new Trainer(strategy, config);
var result = trainer.Train(trainNetwork, trainDataset);

Console.WriteLine($"Strategy:        {result.StrategyName}");
Console.WriteLine($"Total time:      {result.TotalDurationMs} ms");
Console.WriteLine($"Final loss:      {result.FinalLoss:F6}");
Console.WriteLine();
Console.WriteLine("Loss curve (every 5 epochs):");
for (int i = 0; i < result.EpochResults.Length; i++)
{
    if ((i + 1) % 5 == 0 || i == 0)
        Console.WriteLine($"  Epoch {i + 1,2}: loss={result.EpochResults[i].Loss:F6}");
}

bool lossDecreased = result.LossCurve[^1] < result.LossCurve[0];
Console.WriteLine($"\nLoss decreased:  {lossDecreased}  ({result.LossCurve[0]:F6} -> {result.LossCurve[^1]:F6})");

Console.WriteLine();
Console.WriteLine("=== Gradient Check ===");

var checkNet = new Network();
checkNet.AddLayer(new DenseLayer(inputSize: 3, outputSize: 4, activation: new ReLU(), seed: 20));
checkNet.AddLayer(new DenseLayer(inputSize: 4, outputSize: 2, activation: new Sigmoid(), seed: 21));

var checkSample = new DataSample(
    input: new double[] { 0.5, -0.3, 0.8 },
    label: new double[] { 1.0, 0.0 }
);

var checker = new GradientChecker(epsilon: 1e-5, tolerance: 1e-4);
double maxRelErr = checker.ComputeMaxRelativeError(checkNet, new MSELoss(), checkSample);
bool passed = checker.Check(checkNet, new MSELoss(), checkSample);

Console.WriteLine($"Network:         3 -> 4 -> 2  (ReLU + Sigmoid)");
Console.WriteLine($"Max relative error: {maxRelErr:E4}  (tolerance=1e-4)");
Console.WriteLine($"Gradient check:  {(passed ? "PASSED" : "FAILED")}");

Console.WriteLine();
Console.WriteLine("=== DataParallel vs Sequential ===");

var benchDataset = DataLoader.GenerateSynthetic(samples: 800, inputSize: 8, outputSize: 2, seed: 99);

var benchConfig = new TrainingConfig { Epochs = 10, BatchSize = 64, LearningRate = 0.05, Seed = 7 };

Network BuildNet(int seed0) {
    var n = new Network();
    n.AddLayer(new DenseLayer(8, 16, new ReLU(), seed0));
    n.AddLayer(new DenseLayer(16, 8, new ReLU(), seed0 + 1));
    n.AddLayer(new DenseLayer(8, 2, new Sigmoid(), seed0 + 2));
    return n;
}

var mse2 = new MSELoss();
var sgd2 = new SGDOptimizer();
var baseNet = BuildNet(50);

var entries = new (ITrainingStrategy s, Network net)[]
{
    (new SequentialStrategy(mse2, sgd2),                                        baseNet.Clone()),
    (new DataParallelStrategy(mse2, sgd2, threadCount: 4),                      baseNet.Clone()),
    (new DataParallelStrategy(mse2, sgd2, threadCount: 4, useThreadPool: true), baseNet.Clone()),
};

Console.WriteLine($"{"Strategy",-38} | {"Time (ms)",9} | {"Final loss",10} | {"Speedup",7}");
Console.WriteLine(new string('-', 72));

long seqTime = 0;
foreach (var (s, net) in entries)
{
    var cfg = new TrainingConfig
    {
        Epochs = benchConfig.Epochs, BatchSize = benchConfig.BatchSize,
        LearningRate = benchConfig.LearningRate, Seed = benchConfig.Seed
    };
    var sw = Stopwatch.StartNew();
    var res = new Trainer(s, cfg).Train(net, benchDataset);
    sw.Stop();

    long elapsed = sw.ElapsedMilliseconds;
    if (seqTime == 0) seqTime = elapsed;
    double speedup = seqTime / (double)elapsed;

    Console.WriteLine($"{s.Name,-38} | {elapsed,9} | {res.FinalLoss,10:F6} | {speedup,7:F2}x");
}
