namespace SupervisedLearning.Benchmark;

public class BenchmarkReport
{
    public ScalabilityEntry[] ScalabilityData { get; set; } = Array.Empty<ScalabilityEntry>();
    public DataSizeEntry[]    DataSizeData    { get; set; } = Array.Empty<DataSizeEntry>();

    public void Print()
    {
        if (ScalabilityData.Length > 0)
        {
            Console.WriteLine($"  {"Threads",7} | {"Seq (ms)",8} | {"Thread (ms)",11} | {"T-Speedup",9} | {"T-Eff",6} | {"Pool (ms)",9} | {"P-Speedup",9} | {"P-Eff",6}");
            Console.WriteLine($"  {new string('-', 87)}");
            foreach (var e in ScalabilityData)
                Console.WriteLine(
                    $"  {e.ThreadCount,7} | {e.SeqTimeMs,8} | {e.ThreadTimeMs,11} | {e.ThreadSpeedup,8:F2}x | {e.ThreadEfficiency,6:F4} | {e.PoolTimeMs,9} | {e.PoolSpeedup,8:F2}x | {e.PoolEfficiency,6:F4}");
        }

        if (DataSizeData.Length > 0)
        {
            Console.WriteLine($"  {"Samples",7} | {"Seq (ms)",8} | {"Thread (ms)",11} | {"T-Speedup",9} | {"T-Eff",6} | {"Pool (ms)",9} | {"P-Speedup",9} | {"P-Eff",6}");
            Console.WriteLine($"  {new string('-', 87)}");
            foreach (var e in DataSizeData)
                Console.WriteLine(
                    $"  {e.SampleCount,7} | {e.SeqTimeMs,8} | {e.ThreadTimeMs,11} | {e.ThreadSpeedup,8:F2}x | {e.ThreadEfficiency,6:F4} | {e.PoolTimeMs,9} | {e.PoolSpeedup,8:F2}x | {e.PoolEfficiency,6:F4}");
        }
    }
}

public class ScalabilityEntry
{
    public int    ThreadCount      { get; set; }
    public long   SeqTimeMs        { get; set; }
    public long   ThreadTimeMs     { get; set; }
    public double ThreadSpeedup    { get; set; }
    public double ThreadEfficiency { get; set; }
    public long   PoolTimeMs       { get; set; }
    public double PoolSpeedup      { get; set; }
    public double PoolEfficiency   { get; set; }
}

public class DataSizeEntry
{
    public int    SampleCount      { get; set; }
    public long   SeqTimeMs        { get; set; }
    public long   ThreadTimeMs     { get; set; }
    public double ThreadSpeedup    { get; set; }
    public double ThreadEfficiency { get; set; }
    public long   PoolTimeMs       { get; set; }
    public double PoolSpeedup      { get; set; }
    public double PoolEfficiency   { get; set; }
}
