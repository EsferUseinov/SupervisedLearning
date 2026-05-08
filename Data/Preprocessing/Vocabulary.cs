namespace SupervisedLearning.Data.Preprocessing;

public class Vocabulary
{
    public const string PadToken = "<PAD>";
    public const string UnkToken = "<UNK>";

    public int PadIndex => 0;
    public int UnkIndex => 1;
    public int Size => _idxToWord.Length;

    private readonly Dictionary<string, int> _wordToIdx;
    private readonly string[] _idxToWord;

    private Vocabulary(Dictionary<string, int> wordToIdx, string[] idxToWord)
    {
        _wordToIdx = wordToIdx;
        _idxToWord = idxToWord;
    }

    public int GetIndex(string word) =>
        _wordToIdx.TryGetValue(word, out int idx) ? idx : UnkIndex;

    public static Vocabulary Build(IEnumerable<string> texts, int maxVocabSize = 5000)
    {
        var freq = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string text in texts)
        {
            foreach (string token in Tokenizer.Tokenize(text))
            {
                freq.TryGetValue(token, out int count);
                freq[token] = count + 1;
            }
        }

        string[] topWords = freq
            .OrderByDescending(kv => kv.Value)
            .Take(maxVocabSize - 2)
            .Select(kv => kv.Key)
            .ToArray();

        var idxToWord = new string[topWords.Length + 2];
        idxToWord[0] = PadToken;
        idxToWord[1] = UnkToken;
        for (int i = 0; i < topWords.Length; i++)
            idxToWord[i + 2] = topWords[i];

        var wordToIdx = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < idxToWord.Length; i++)
            wordToIdx[idxToWord[i]] = i;

        return new Vocabulary(wordToIdx, idxToWord);
    }
}
