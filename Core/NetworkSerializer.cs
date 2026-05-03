namespace SupervisedLearning.Core;

using SupervisedLearning.Core.Layers;

public static class NetworkSerializer
{
    public static void Save(Network network, string path)
    {
        using var writer = new BinaryWriter(File.Open(path, FileMode.Create));
        writer.Write(network.Layers.Count);
        foreach (var layer in network.Layers)
        {
            var dense = (DenseLayer)layer;
            writer.Write(dense.InputSize);
            writer.Write(dense.OutputSize);
            for (int i = 0; i < dense.OutputSize; i++)
            {
                for (int j = 0; j < dense.InputSize; j++)
                    writer.Write(dense.GetWeight(i, j));
                writer.Write(dense.GetBias(i));
            }
        }
    }

    public static void Load(Network network, string path)
    {
        using var reader = new BinaryReader(File.Open(path, FileMode.Open));
        int layerCount = reader.ReadInt32();
        if (layerCount != network.Layers.Count)
            throw new InvalidOperationException(
                $"Layer count mismatch: file has {layerCount}, network has {network.Layers.Count}.");
        for (int l = 0; l < layerCount; l++)
        {
            var dense = (DenseLayer)network.Layers[l];
            int inputSize  = reader.ReadInt32();
            int outputSize = reader.ReadInt32();
            if (inputSize != dense.InputSize || outputSize != dense.OutputSize)
                throw new InvalidOperationException(
                    $"Layer {l} size mismatch: file has {inputSize}x{outputSize}, " +
                    $"network has {dense.InputSize}x{dense.OutputSize}.");
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                    dense.SetWeight(i, j, reader.ReadDouble());
                dense.SetBias(i, reader.ReadDouble());
            }
        }
    }
}
