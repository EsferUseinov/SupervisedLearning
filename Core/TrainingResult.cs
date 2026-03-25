namespace SupervisedLearning.Core;

public class TrainingResult
{
    public EpochResult[] EpochResults { get; set; } = Array.Empty<EpochResult>();
    public long TotalDurationMs { get; set; }
    public double FinalLoss { get; set; }
    public string StrategyName { get; set; } = string.Empty;
    public double[] LossCurve { get; set; } = Array.Empty<double>();
}
