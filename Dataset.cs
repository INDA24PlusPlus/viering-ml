using NumSharp;

namespace viering_ml;

public class Dataset
{
    public NDArray images;
    public NDArray labels;
    public int size;
    
    public Dataset(int startIndex, int endIndex)
    {
        this.size = endIndex - startIndex + 1;
        
        string imagesPath = "data/train-images.idx3-ubyte";
        string labelsPath = "data/train-labels.idx1-ubyte";
        
        using (BinaryReader brImages = new (new FileStream(imagesPath, FileMode.Open)), brLabels = new (new FileStream(labelsPath, FileMode.Open)))
        {
            int magic1 = brImages.ReadInt32Endian();
            if (magic1 != 2051)
                throw new Exception($"Invalid magic number {magic1}!");
            int numImages = brImages.ReadInt32Endian();
            int numRows = brImages.ReadInt32Endian();
            int numCols = brImages.ReadInt32Endian();
            
            int magic2 = brLabels.ReadInt32Endian();
            if (magic2 != 2049)
                throw new Exception($"Invalid magic number {magic2}!");
            
            int numLabels = brLabels.ReadInt32Endian();
            
            byte[][] images = new byte[endIndex - startIndex + 1][];
            double[] labels = new double[endIndex - startIndex + 1];
            int dimensions = numRows * numCols;
            for (int i = 0; i < numImages; i++)
            {
                var temp1 = brImages.ReadBytes(dimensions);
                var temp2 = brLabels.ReadByte();

                if (i >= startIndex && i <= endIndex)
                {
                    images[i - startIndex] = temp1;
                    labels[i - startIndex] = temp2;
                }
            }
        
            double[][] imagesP2 = new double[endIndex - startIndex + 1][];
            for (int i = 0; i < images.Length; i++)
            {
                imagesP2[i] = new double[28 * 28];
                for (int j = 0; j < images[i].Length; j++)
                {
                    imagesP2[i][j] = images[i][j] / 255d;
                }
            }
        
            this.images = np.array(imagesP2)
                .reshape(endIndex - startIndex + 1, 28 * 28)
                .transpose();
            this.labels = np.array(labels).transpose();
        }
    }
}


static class BinaryReaderExtension
{
    public static int ReadInt32Endian(this BinaryReader br)
    {
        var bytes = br.ReadBytes(sizeof(Int32));
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}