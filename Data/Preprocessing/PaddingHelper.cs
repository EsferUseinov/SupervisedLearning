namespace SupervisedLearning.Data.Preprocessing;

public static class PaddingHelper
{
    public static int[] PadOrTruncate(int[] tokens, int maxLength, int padIdx)
    {
        var result = new int[maxLength];
        Array.Fill(result, padIdx);
        int copyLen = Math.Min(tokens.Length, maxLength);
        Array.Copy(tokens, result, copyLen);
        return result;
    }
}
