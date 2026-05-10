namespace SupervisedLearning.Data;

using SupervisedLearning.Core;
using SupervisedLearning.Data.Preprocessing;

public static class SemEvalLoader
{
    public static readonly string[] Techniques = new[]
    {
        "Loaded_Language",
        "Name_Calling,Labeling",
        "Repetition",
        "Doubt",
        "Exaggeration,Minimisation",
        "Appeal_to_fear-prejudice",
        "Flag-Waving",
        "Causal_Oversimplification",
        "Other"
    };

    private static readonly HashSet<string> ValidTechniqueSet =
        new HashSet<string>(Techniques[..8], StringComparer.Ordinal);

    public static int NumClasses => Techniques.Length;

    public static Vocabulary BuildVocabulary(string articlesDir, int maxVocabSize = 5000)
    {
        var texts = Directory
            .EnumerateFiles(articlesDir, "article*.txt")
            .Select(File.ReadAllText);
        return Vocabulary.Build(texts, maxVocabSize);
    }

    public static (DataSet train, DataSet val) LoadWithSplit(
        string articlesDir,
        string labelsPath,
        Vocabulary vocab,
        int seqLen = 32,
        int totalNotPropSamples = 2000,
        double trainRatio = 0.8,
        int seed = 67,
        string? extraLabelsPath = null)
    {
        var spans = ParseLabels(labelsPath);
        var syntheticSpans = extraLabelsPath != null
            ? ParseLabels(extraLabelsPath)
            : new Dictionary<string, List<(string, int, int)>>(StringComparer.Ordinal);

        string[] articleIds = spans.Keys.ToArray();

        var rng = new Random(seed);
        Shuffle(articleIds, rng);

        int trainCount = (int)(articleIds.Length * trainRatio);
        var trainIds = new HashSet<string>(articleIds[..trainCount], StringComparer.Ordinal);

        var trainSamples = new List<DataSample>();
        var valSamples = new List<DataSample>();
        var notPropTrain = new List<string>();
        var notPropVal = new List<string>();

        foreach (string id in articleIds)
        {
            string path = Path.Combine(articlesDir, $"article{id}.txt");
            if (!File.Exists(path)) continue;

            string text = File.ReadAllText(path);
            var articleSpans = spans.TryGetValue(id, out var sp) ? sp : new List<(string, int, int)>();
            bool isTrain = trainIds.Contains(id);

            var target = isTrain ? trainSamples : valSamples;
            var notPropPool = isTrain ? notPropTrain : notPropVal;

            foreach (var (technique, start, end) in articleSpans)
            {
                if (!ValidTechniqueSet.Contains(technique)) continue;
                int s = Math.Clamp(start, 0, text.Length);
                int e = Math.Clamp(end, s, text.Length);
                if (e <= s) continue;
                target.Add(MakeSample(text.Substring(s, e - s), TechniqueIndex(technique), vocab, seqLen));
            }

            CollectNotPropCandidates(text, articleSpans, notPropPool, rng);
        }

        foreach (var (id, articleSpans) in syntheticSpans)
        {
            string path = Path.Combine(articlesDir, $"article{id}.txt");
            if (!File.Exists(path)) continue;

            string text = File.ReadAllText(path);

            foreach (var (technique, start, end) in articleSpans)
            {
                if (!ValidTechniqueSet.Contains(technique)) continue;
                int s = Math.Clamp(start, 0, text.Length);
                int e = Math.Clamp(end, s, text.Length);
                if (e <= s) continue;
                trainSamples.Add(MakeSample(text.Substring(s, e - s), TechniqueIndex(technique), vocab, seqLen));
            }

            CollectNotPropCandidates(text, articleSpans, notPropTrain, rng);
        }

        int trainNotProp = (int)(totalNotPropSamples * trainRatio);
        int valNotProp = totalNotPropSamples - trainNotProp;

        AddNotPropaganda(trainSamples, notPropTrain, trainNotProp, vocab, seqLen, rng);
        AddNotPropaganda(valSamples, notPropVal, valNotProp, vocab, seqLen, new Random(seed + 1));

        return (new DataSet(trainSamples.ToArray()), new DataSet(valSamples.ToArray()));
    }

    public static double[] ComputeClassWeights(DataSet trainSet)
    {
        var counts = new int[NumClasses];
        foreach (var sample in trainSet.Samples)
            counts[MathHelper.ArgMax(sample.Label)]++;
        double total = trainSet.Samples.Length;
        var weights = new double[NumClasses];
        for (int c = 0; c < NumClasses; c++)
            weights[c] = counts[c] > 0 ? (total / NumClasses) / counts[c] : 1.0;
        return weights;
    }

    private static Dictionary<string, List<(string technique, int start, int end)>> ParseLabels(string path)
    {
        var result = new Dictionary<string, List<(string, int, int)>>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split('\t');
            if (parts.Length < 4) continue;
            string articleId = parts[0].Trim();
            string technique = parts[1].Trim();
            if (!int.TryParse(parts[2].Trim(), out int start)) continue;
            if (!int.TryParse(parts[3].Trim(), out int end)) continue;
            if (!result.TryGetValue(articleId, out var list))
            {
                list = new List<(string, int, int)>();
                result[articleId] = list;
            }
            list.Add((technique, start, end));
        }
        return result;
    }

    private static void CollectNotPropCandidates(
        string text,
        List<(string technique, int start, int end)> spans,
        List<string> pool,
        Random rng)
    {
        var covered = new bool[text.Length];
        foreach (var (_, start, end) in spans)
        {
            int s = Math.Clamp(start, 0, text.Length);
            int e = Math.Clamp(end, s, text.Length);
            for (int i = s; i < e; i++)
                covered[i] = true;
        }

        int pos = 0;
        while (pos < text.Length)
        {
            if (covered[pos]) { pos++; continue; }
            int gapStart = pos;
            while (pos < text.Length && !covered[pos]) pos++;
            int gapLen = pos - gapStart;
            if (gapLen < 20) continue;

            int offset = gapStart;
            while (offset < pos - 20)
            {
                int len = Math.Min(rng.Next(30, 80), pos - offset);
                string candidate = text.Substring(offset, len).Trim();
                if (candidate.Length >= 10)
                    pool.Add(candidate);
                offset += len;
            }
        }
    }

    private static void AddNotPropaganda(
        List<DataSample> target,
        List<string> candidates,
        int count,
        Vocabulary vocab,
        int seqLen,
        Random rng)
    {
        Shuffle(candidates, rng);
        int take = Math.Min(count, candidates.Count);
        for (int i = 0; i < take; i++)
            target.Add(MakeSample(candidates[i], NumClasses - 1, vocab, seqLen));
    }

    private static DataSample MakeSample(string text, int classIdx, Vocabulary vocab, int seqLen)
    {
        string[] tokens = Tokenizer.Tokenize(text);
        var ids = new int[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            ids[i] = vocab.GetIndex(tokens[i]);
        int[] padded = PaddingHelper.PadOrTruncate(ids, seqLen, vocab.PadIndex);
        var input = new double[seqLen];
        for (int i = 0; i < seqLen; i++)
            input[i] = padded[i];
        var label = new double[NumClasses];
        label[classIdx] = 1.0;
        return new DataSample(input, label);
    }

    private static int TechniqueIndex(string technique)
    {
        for (int i = 0; i < Techniques.Length - 1; i++)
            if (Techniques[i] == technique) return i;
        return NumClasses - 1;
    }

    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

}
