namespace SupervisedLearning.Core;

using SupervisedLearning.Data.Preprocessing;

public static class VocabularySerializer
{
    public static void Save(Vocabulary vocab, string path)
    {
        using var w = new BinaryWriter(File.Open(path, FileMode.Create));
        w.Write(vocab.Size);
        for (int i = 0; i < vocab.Size; i++)
            w.Write(vocab.GetWord(i));
    }

    public static Vocabulary Load(string path)
    {
        using var r = new BinaryReader(File.Open(path, FileMode.Open));
        int size = r.ReadInt32();
        var words = new string[size];
        for (int i = 0; i < size; i++)
            words[i] = r.ReadString();
        return Vocabulary.FromWordList(words);
    }
}
