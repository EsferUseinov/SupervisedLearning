namespace SupervisedLearning.Benchmark;

using SupervisedLearning.Core;

public class BenchmarkReport
{
    public TrainingResult[] Results { get; set; } = Array.Empty<TrainingResult>();
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
    public ScalabilityEntry[] ScalabilityData { get; set; } = Array.Empty<ScalabilityEntry>();

    public void Print()
    {
        if (Results.Length > 0)
        {
            Console.WriteLine($"{"Strategy",-40} | {"Avg Time (ms)",13} | {"Final Loss",10} | {"Speedup",8}");
            Console.WriteLine(new string('-', 82));

            long baselineMs = Results[0].TotalDurationMs;
            foreach (var r in Results)
            {
                double sp = baselineMs / (double)Math.Max(r.TotalDurationMs, 1);
                Console.WriteLine($"{r.StrategyName,-40} | {r.TotalDurationMs,13} | {r.FinalLoss,10:F6} | {sp,7:F2}x");
            }

            Console.WriteLine($"\nSpeedup: {Speedup:F2}x    Efficiency: {Efficiency:F4}");
        }

        if (ScalabilityData.Length > 0)
        {
            Console.WriteLine($"{"Threads",8} | {"Speedup",9} | {"Efficiency",10}");
            Console.WriteLine(new string('-', 36));
            foreach (var e in ScalabilityData)
                Console.WriteLine($"{e.ThreadCount,8} | {e.Speedup,8:F2}x | {e.Efficiency,10:F4}");
        }
    }
}

public class ScalabilityEntry
{
    public int ThreadCount { get; set; }
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
}
