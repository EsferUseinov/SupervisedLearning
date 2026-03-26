using SupervisedLearning.Core;
using SupervisedLearning.Core.Activations;
using SupervisedLearning.Core.Layers;
using SupervisedLearning.Data;
using SupervisedLearning.Training.Optimizers;

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
