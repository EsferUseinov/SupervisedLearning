namespace SupervisedLearning.Core;

using SupervisedLearning.Core.Layers;

public static class NetworkSerializer
{
    private const int TagDense     = 0;
    private const int TagEmbedding = 1;
    private const int TagConv1D    = 2;
    private const int TagMaxPool1D = 3;

    public static void Save(Network network, string path)
    {
        using var w = new BinaryWriter(File.Open(path, FileMode.Create));
        w.Write(network.Layers.Count);

        foreach (var layer in network.Layers)
        {
            if (layer is DenseLayer dense)
            {
                w.Write(TagDense);
                w.Write(dense.InputSize);
                w.Write(dense.OutputSize);
                for (int i = 0; i < dense.OutputSize; i++)
                {
                    for (int j = 0; j < dense.InputSize; j++)
                        w.Write(dense.GetWeight(i, j));
                    w.Write(dense.GetBias(i));
                }
            }
            else if (layer is EmbeddingLayer emb)
            {
                w.Write(TagEmbedding);
                w.Write(emb.VocabSize);
                w.Write(emb.EmbeddingDim);
                w.Write(emb.SeqLen);
                for (int i = 0; i < emb.VocabSize * emb.EmbeddingDim; i++)
                    w.Write(emb.GetTableEntry(i));
            }
            else if (layer is Conv1DLayer conv)
            {
                w.Write(TagConv1D);
                w.Write(conv.SeqLen);
                w.Write(conv.InChannels);
                w.Write(conv.FilterSize);
                w.Write(conv.NumFilters);
                for (int i = 0; i < conv.FilterWeightCount; i++)
                    w.Write(conv.GetFilter(i));
                for (int f = 0; f < conv.NumFilters; f++)
                    w.Write(conv.GetConvBias(f));
            }
            else if (layer is MaxPool1DLayer pool)
            {
                w.Write(TagMaxPool1D);
                w.Write(pool.InputSize / pool.OutputSize);
                w.Write(pool.OutputSize);
            }
            else
            {
                throw new NotSupportedException($"Unsupported layer type: {layer.GetType().Name}");
            }
        }
    }

    public static void Load(Network network, string path)
    {
        using var r = new BinaryReader(File.Open(path, FileMode.Open));
        int layerCount = r.ReadInt32();
        if (layerCount != network.Layers.Count)
            throw new InvalidOperationException(
                $"Layer count mismatch: file has {layerCount}, network has {network.Layers.Count}.");

        for (int l = 0; l < layerCount; l++)
        {
            int tag   = r.ReadInt32();
            var layer = network.Layers[l];

            if (tag == TagDense && layer is DenseLayer dense)
            {
                int inSize  = r.ReadInt32();
                int outSize = r.ReadInt32();
                if (inSize != dense.InputSize || outSize != dense.OutputSize)
                    throw new InvalidOperationException(
                        $"DenseLayer {l} size mismatch: file {inSize}x{outSize}, net {dense.InputSize}x{dense.OutputSize}.");
                for (int i = 0; i < outSize; i++)
                {
                    for (int j = 0; j < inSize; j++)
                        dense.SetWeight(i, j, r.ReadDouble());
                    dense.SetBias(i, r.ReadDouble());
                }
            }
            else if (tag == TagEmbedding && layer is EmbeddingLayer emb)
            {
                int vocabSize = r.ReadInt32();
                int embDim    = r.ReadInt32();
                int seqLen    = r.ReadInt32();
                if (vocabSize != emb.VocabSize || embDim != emb.EmbeddingDim || seqLen != emb.SeqLen)
                    throw new InvalidOperationException($"EmbeddingLayer {l} dimension mismatch.");
                for (int i = 0; i < vocabSize * embDim; i++)
                    emb.SetTableEntry(i, r.ReadDouble());
            }
            else if (tag == TagConv1D && layer is Conv1DLayer conv)
            {
                int seqLen     = r.ReadInt32();
                int inChannels = r.ReadInt32();
                int filterSize = r.ReadInt32();
                int numFilters = r.ReadInt32();
                if (seqLen != conv.SeqLen || inChannels != conv.InChannels ||
                    filterSize != conv.FilterSize || numFilters != conv.NumFilters)
                    throw new InvalidOperationException($"Conv1DLayer {l} dimension mismatch.");
                for (int i = 0; i < conv.FilterWeightCount; i++)
                    conv.SetFilter(i, r.ReadDouble());
                for (int f = 0; f < conv.NumFilters; f++)
                    conv.SetConvBias(f, r.ReadDouble());
            }
            else if (tag == TagMaxPool1D && layer is MaxPool1DLayer pool)
            {
                int inputLen    = r.ReadInt32();
                int numChannels = r.ReadInt32();
                if (inputLen * numChannels != pool.InputSize || numChannels != pool.OutputSize)
                    throw new InvalidOperationException($"MaxPool1DLayer {l} dimension mismatch.");
            }
            else
            {
                throw new InvalidOperationException(
                    $"Layer {l} type/tag mismatch: file tag={tag}, layer={layer.GetType().Name}.");
            }
        }
    }
}
