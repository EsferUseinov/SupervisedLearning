namespace SupervisedLearning.Core;

public static class MathHelper
{
    public static int ArgMax(double[] v)
    {
        int idx = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[idx]) idx = i;
        return idx;
    }
}
