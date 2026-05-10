namespace SupervisedLearning.Core;

public class TrainingResult
{
    public EpochResult[] EpochResults { get; set; } = Array.Empty<EpochResult>();
    public long TotalDurationMs { get; set; }
    public double FinalLoss { get; set; }
}
