namespace SupervisedLearning.Benchmark;

using SupervisedLearning.Core;

public class BenchmarkReport
{
    public TrainingResult[] Results { get; set; } = Array.Empty<TrainingResult>();
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
    public ScalabilityEntry[] ScalabilityData { get; set; } = Array.Empty<ScalabilityEntry>();

    public void Print() => throw new NotImplementedException();
}

public class ScalabilityEntry
{
    public int ThreadCount { get; set; }
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
}
