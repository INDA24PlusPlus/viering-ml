using NumSharp;
using viering_ml;

NDArray ReLU(NDArray Z)
{
    return np.maximum(Z, 0); // maybe add mini incline to values where x<0
}

NDArray ReLUDeriv(NDArray Z)
{
    return np.minimum(Z, 0) + 1d; // just 1 above x=0, and 0 below that
}

NDArray SoftMax(NDArray Z)
{
    return np.exp(Z) / np.sum(np.exp(Z).astype(np.float32), axis: 0);
}

(NDArray, NDArray, NDArray, NDArray) ForwardProp(Model model, NDArray X)
{
    var z1 = model.l1Weights.dot(X) + model.l1Biases;
    var a1 = ReLU(z1);
    var z2 = model.l2Weights.dot(a1) + model.l2Biases;
    var a2 = SoftMax(z2);
    
    return (z1, a1, z2, a2);
}

NDArray OneHot(NDArray input)
{
    var oneHotY = np.zeros((10, input.size));
    for (int i = 0; i < input.size; i++)
    {
        int id = Convert.ToInt32(input[i].GetValue());
        oneHotY[id, i] = 1;
    }
    return oneHotY;
}

(NDArray, NDArray, NDArray, NDArray) BackwardProp(NDArray z1, NDArray a1, NDArray z2, NDArray a2, Model model, Dataset dataset)
{
    double m = dataset.size;
    
    var oneHotLabels = OneHot(dataset.labels);
    var dZ2 = a2 - oneHotLabels;
    var dW2 = 1d / m * dZ2.dot(a1.T);
    var db2 = 1d / m * np.sum(dZ2.astype(np.float32), axis: 0);
    var dZ1 = model.l2Weights.T.dot(dZ2) * ReLUDeriv(z1);
    var dW1 = 1d / m * dZ1.dot(dataset.images.T);
    var db1 = 1d / m * np.sum(dZ1.astype(np.float32), axis: 0);
    
    return (dW1, db1, dW2, db2);
}

Model UpdateParams(Model model, NDArray dW1, NDArray db1, NDArray dW2, NDArray db2, float learningRate)
{
    model.l1Weights -= learningRate * dW1;
    model.l1Biases -= learningRate * db1;
    model.l2Weights -= learningRate * dW2;
    model.l2Biases -= learningRate * db2;
    
    return model;
}

NDArray GetPredictions(NDArray input)
{
    return np.argmax(input, 0);
}

int GetCorrectPredictions(NDArray predictions, NDArray labels)
{
    int i = 0;
    for (var j = 0; j < predictions.size; j++)
    {
        if (predictions[j] == labels[j]) i++;
    }
    return i;
}

Model TrainModel(Dataset dataset, int l2Size, float learningRate, int epochs)
{
    var model = new Model
    {
        l1Weights = np.random.rand(l2Size, 784) - 0.5d,
        l1Biases = np.random.rand(l2Size, 1) - 0.5d,
        l2Weights = np.random.rand(10, l2Size) - 0.5d,
        l2Biases = np.random.rand(10, 1) - 0.5d
    };
    
    Console.WriteLine("Training started");

    // actual training
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        var (z1, a1, z2, a2) = ForwardProp(model, dataset.images);
        var (dW1, db1, dW2, db2) = BackwardProp(z1, a1, z2, a2, model, dataset);
        model = UpdateParams(model, dW1, db1, dW2, db2, learningRate);

        // printin progresss
        var predictions = GetPredictions(a2);
        var numCorrect = GetCorrectPredictions(predictions, dataset.labels);
        Console.WriteLine($"epoch={epoch+1} correct={numCorrect / (double)dataset.size}");
    }
    
    Console.WriteLine("Training finished");

    return model;
}

void MakePredictions(Dataset dataset, Model model)
{
    Console.WriteLine("Making predictions");
    
    var (_, _, _, a2) = ForwardProp(model, dataset.images);
    var predictions = GetPredictions(a2);
    var numCorrect = GetCorrectPredictions(predictions, dataset.labels);
    
    Console.WriteLine($"correct={numCorrect} total={dataset.size} frac={numCorrect / (double)dataset.size}");
}

var trainingDataset = new Dataset(0, 3999);
var testingDataset = new Dataset(8000, 11999);

var model = TrainModel(trainingDataset, 50, 0.01f, 100);

MakePredictions(testingDataset, model);

struct Model
{
    public NDArray l1Weights;
    public NDArray l1Biases;
    public NDArray l2Weights;
    public NDArray l2Biases;
}

