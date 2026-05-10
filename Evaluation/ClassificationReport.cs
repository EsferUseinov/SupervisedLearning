namespace SupervisedLearning.Evaluation;

using SupervisedLearning.Core;
using SupervisedLearning.Data;

public sealed class ClassificationReport
{
    private readonly string[] _classNames;
    private readonly int[] _tp;
    private readonly int[] _fp;
    private readonly int[] _fn;
    private readonly int[] _support;
    private readonly int _bTP;
    private readonly int _bFP;
    private readonly int _bFN;
    private readonly int _bTN;

    public ClassificationReport(Network net, DataSample[] samples, string[] classNames)
    {
        _classNames = classNames;
        int n = classNames.Length;
        int notPropIdx = n - 1;
        _tp = new int[n];
        _fp = new int[n];
        _fn = new int[n];
        _support = new int[n];

        int bTP = 0, bFP = 0, bFN = 0, bTN = 0;

        foreach (var s in samples)
        {
            int predicted = ArgMax(net.Forward(s.Input));
            int actual    = ArgMax(s.Label);

            _support[actual]++;
            if (predicted == actual)
                _tp[actual]++;
            else
            {
                _fp[predicted]++;
                _fn[actual]++;
            }

            bool predProp = predicted != notPropIdx;
            bool actProp  = actual    != notPropIdx;

            if      ( predProp &&  actProp) bTP++;
            else if ( predProp && !actProp) bFP++;
            else if (!predProp &&  actProp) bFN++;
            else                            bTN++;
        }

        _bTP = bTP; _bFP = bFP; _bFN = bFN; _bTN = bTN;
    }

    public void Print(string label = "")
    {
        int n     = _classNames.Length;
        int total = _support.Sum();

        string heading = label.Length > 0
            ? $"=== Classification Report — {label} ({total} samples) ==="
            : $"=== Classification Report ({total} samples) ===";
        Console.WriteLine($"\n{heading}");
        Console.WriteLine($"  {"Class",-35} {"Prec",6}  {"Rec",6}  {"F1",6}  {"Support",8}");
        Console.WriteLine($"  {new string('─', 67)}");

        double macroP = 0, macroR = 0, macroF1 = 0;
        double wP = 0, wR = 0, wF1 = 0;

        for (int c = 0; c < n; c++)
        {
            double p  = _tp[c] + _fp[c] > 0 ? (double)_tp[c] / (_tp[c] + _fp[c]) : 0;
            double r  = _tp[c] + _fn[c] > 0 ? (double)_tp[c] / (_tp[c] + _fn[c]) : 0;
            double f1 = p + r > 0 ? 2 * p * r / (p + r) : 0;

            Console.WriteLine($"  {_classNames[c],-35} {p,6:F2}   {r,6:F2}   {f1,6:F2}  {_support[c],8}");

            macroP += p;          macroR += r;          macroF1 += f1;
            wP     += p * _support[c];
            wR     += r * _support[c];
            wF1    += f1 * _support[c];
        }

        Console.WriteLine($"  {new string('─', 67)}");
        Console.WriteLine($"  {"Macro avg",-35} {macroP / n,6:F2}   {macroR / n,6:F2}   {macroF1 / n,6:F2}  {total,8}");
        Console.WriteLine($"  {"Weighted avg",-35} {wP / total,6:F2}   {wR / total,6:F2}   {wF1 / total,6:F2}  {total,8}");

        double bP  = _bTP + _bFP > 0 ? (double)_bTP / (_bTP + _bFP) : 0;
        double bR  = _bTP + _bFN > 0 ? (double)_bTP / (_bTP + _bFN) : 0;
        double bF1 = bP + bR > 0 ? 2 * bP * bR / (bP + bR) : 0;

        string binHeading = label.Length > 0
            ? $"=== Binary: Propaganda Detection — {label} ==="
            : "=== Binary: Propaganda Detection ===";
        Console.WriteLine($"\n{binHeading}");
        Console.WriteLine($"               | Predicted Prop | Predicted Not |");
        Console.WriteLine($"  {new string('─', 47)}");
        Console.WriteLine($"  Actual Prop  |   {_bTP,6} (TP)   |  {_bFN,6} (FN)  |");
        Console.WriteLine($"  Actual Not   |   {_bFP,6} (FP)   |  {_bTN,6} (TN)  |");
        Console.WriteLine($"  {new string('─', 47)}");
        Console.WriteLine($"  Precision: {bP:F2}   Recall: {bR:F2}   F1: {bF1:F2}");
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[idx]) idx = i;
        return idx;
    }
}
