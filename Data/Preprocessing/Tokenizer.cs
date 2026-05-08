namespace SupervisedLearning.Data.Preprocessing;

public static class Tokenizer
{
    private static readonly char[] Separators =
        " \t\n\r.,!?;:\"'()[]{}/-_@#*+=<>\\|~`".ToCharArray();

    public static string[] Tokenize(string text) =>
        text.ToLowerInvariant()
            .Split(Separators, StringSplitOptions.RemoveEmptyEntries);
}
