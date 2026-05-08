namespace SupervisedLearning.Core;

public class EpochResult
{
    public long EpochDurationMs { get; set; }
    public long ForwardTimeMs { get; set; }
    public long BackwardTimeMs { get; set; }
    public long SyncTimeMs { get; set; }
    public int SamplesProcessed { get; set; }
    public double Loss { get; set; }
    public double LearningRate { get; set; }
}
