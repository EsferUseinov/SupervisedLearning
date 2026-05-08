namespace SupervisedLearning.Benchmark;

public class BenchmarkReport
{
    public ScalabilityEntry[] ScalabilityData { get; set; } = Array.Empty<ScalabilityEntry>();
    public DataSizeEntry[] DataSizeData { get; set; } = Array.Empty<DataSizeEntry>();

    public void Print()
    {
        if (ScalabilityData.Length > 0)
        {
            Console.WriteLine($"{"Threads",8} | {"Speedup",9} | {"Efficiency",10}");
            Console.WriteLine(new string('-', 36));
            foreach (var e in ScalabilityData)
                Console.WriteLine($"{e.ThreadCount,8} | {e.Speedup,8:F2}x | {e.Efficiency,10:F4}");
        }

        if (DataSizeData.Length > 0)
        {
            Console.WriteLine($"{"Samples",8} | {"Seq (ms)",10} | {"Par (ms)",10} | {"Speedup",9} | {"Efficiency",10}");
            Console.WriteLine(new string('-', 58));
            foreach (var e in DataSizeData)
                Console.WriteLine($"{e.SampleCount,8} | {e.SeqTimeMs,10} | {e.ParTimeMs,10} | {e.Speedup,8:F2}x | {e.Efficiency,10:F4}");
        }
    }
}

public class ScalabilityEntry
{
    public int ThreadCount { get; set; }
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
}

public class DataSizeEntry
{
    public int SampleCount { get; set; }
    public long SeqTimeMs { get; set; }
    public long ParTimeMs { get; set; }
    public double Speedup { get; set; }
    public double Efficiency { get; set; }
}
