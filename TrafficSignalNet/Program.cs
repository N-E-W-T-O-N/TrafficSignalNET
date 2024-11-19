using System.Globalization;
using CsvHelper;
using MoreLinq.Extensions;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.Metrics;
//using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Optimizers;
//using Tensorflow.Operations.Losses;
using TrafficSignalNET;
//using System.Reflection;
//using Newtonsoft.Json;
using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;
//using static Tensorflow.Keras.Utils.KerasUtils;
//using Tensorflow.Keras.Callbacks;

//# import pandas
const int img_h = 32;
const int img_w = 32;
const int n_channels = 3;

Console.WriteLine(Directory.GetCurrentDirectory());
//  Working in VS
//  var BasePath = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "Data");
//  Works in CLI
//  var BasePath = Path.Combine(Directory.GetCurrentDirectory(),  "Data");
//  var ass = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
var BasePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Data"));
Console.WriteLine(BasePath);
var result = Directory.GetDirectories(BasePath);

foreach (var x in result)
{
    Console.WriteLine(x);
}

//var FilePath = Path.Combine(BasePath, "Train");

FileOperation r = new FileOperation();

List<DataRow> records = r.ReadCsv(path: Path.Join(BasePath, "Train.csv"));


// 39209
List<int> yLabels = [];
List<string> xImagePath = [];

foreach (var (index, row) in records.Select((row, index) => (index, row)))
{
    if (index % 10000 == 0)
    {
        Console.WriteLine($"[INFO] processed {index} total images");
    }

    //var (label, imagePath) = (row.ClassId, row.Path);

    yLabels.add(row.ClassId);
    xImagePath.add(Path.Combine(BasePath, row.Path));

    //imagePath = Path.Combine(BasePath, imagePath);

    //NDArray a = np.array<int>(xLabels.ToArray());
}

int[] uniqueLabels = yLabels.Distinct().ToArray();
int classCount = uniqueLabels.Count();
int totalCount = yLabels.Count();

Dictionary<int,float> classWeight = yLabels.GroupBy(x => x)
    .Select(g => new { Index = g.Key, Count = g.Count() })
    .OrderBy(x=>x.Index)
    .ToDictionary(x => x.Index, x =>  totalCount/ (float)(classCount*x.Count));
//.ToDictionary(x => x.Index, x => x.Count/ (float)totalCount );

Console.WriteLine("ClassWeight");
float total = 0f;
foreach (var (key, value) in classWeight)
{
    Console.WriteLine($"|    {key}  |   {value}");
    total += value;
}
Console.WriteLine($"|   TOTAL    | {total}");

//Create Empty
// TF message comming from here
var xTrain = np.zeros((records.Count, img_h, img_w, n_channels), dtype: tf.float32); // TotalRecords * Height * width * Channel
//var yTrain = tf.one_hot(np.array(xLabels.ToArray(),dtype:tf.int64), depth: classCount);
var yTrain = np.eye(classCount, dtype: tf.float32)[np.array(yLabels.ToArray(), tf.float32).reshape(-1)];
// Encode label to a one hot vector.

//var indexArray = np.array(xLabels.ToArray());  // N * xLabels.Total

//var one = yTrain[indexArray];

//indexArray = indexArray.reshape(-1);

//var one_hot_targets = np.eye(uniqueLabels.Length)[indexArray];
//var sh = one_hot_targets.shape;
//Load labels

//Util.ToCategorical(y_train, num_classes);
print("Load Labels To NDArray : OK!");
int i = 0;
// TO Check the Value 
foreach (var val in yTrain[0])
{
    Console.Write($"{val} ");
    i++;
}

Console.WriteLine(yTrain[0].shape);


r.LoadImage(xImagePath.ToArray(), xTrain, "Training");

TrafficSignal ts = new TrafficSignal();

ts.BuildModel(img_h, img_w, n_channels, classCount);

ts.Compile();

var startTime = DateTime.Now;
var history = ts.Train(xTrain, yTrain, classWeight);
var endTime = DateTime.Now;
var diff = endTime - startTime;
Console.WriteLine($"Execution Time {diff.Minutes} {diff.Seconds} {diff.Milliseconds}");

ts.Summary();
ts.Save("./Model");


//var hist = JsonConvert.SerializeObject(history.history);
//File.WriteAllText("History.json",hist);

r.CreateImage(history.history,path:Path.Join(BasePath,"..","ModelResult.jpg"));


/* NOW TEST THE MODEL*/

//FilePath = Path.Combine(BasePath, "Test");

records = r.ReadCsv(path: Path.Combine(BasePath ,"Test.csv"));

List<string> testImagePath =new();
List<int> textXLabels = new();

foreach (var row in records)
{
    //yTest.add(row.ClassId);
    testImagePath.add(Path.Combine(BasePath, row.Path));
    textXLabels.add(row.ClassId);
}


// compile keras model in tensorflow static graph

// prepare dataset
// normalize the input
// x_train = x_train / 255.0f;
var xTest = np.zeros((testImagePath.Count, img_h,img_w, n_channels), dtype: tf.float32);

r.LoadImage(testImagePath.ToArray(), xTest, "Testing");

var yTest=ts.Predict(xTest,1);

Console.WriteLine();
Console.WriteLine();

Console.WriteLine(yTest[0]);
Console.WriteLine();
Console.WriteLine();

Console.WriteLine(yTest);
var yNDarray = yTest.numpy();
Console.WriteLine($"{yNDarray[0]}");
Console.WriteLine();


// Create Confusion Matrix

List<List<int>> matrix = new();
for (int ind = 0; ind < classCount; ind++)
{
    matrix.add(Enumerable.Repeat(0,classCount).ToList());
}
Console.WriteLine();

for (i = 0 ;i< textXLabels.Count; i++)
{
    try
    {
        int j = yTest.numpy()[i];
        matrix[textXLabels[i]][j] += 1;
    }
    catch (Exception ex)
    {
        Console.WriteLine();
    }
}
Console.WriteLine();


//var m = JsonConvert.SerializeObject(matrix);
//File.WriteAllText("Metrix.json", m);


foreach (var row in matrix)
{
    foreach (var col in row)
    {
        Console.Write($"{col} ");
    }
    Console.WriteLine();
}
Console.WriteLine();
Console.WriteLine();

class FileOperation
{
    public void LoadImage(string[] a, NDArray b, string process)
    {
        //var graph = tf.Graph().as_default();

        //for (int i = 0; i < a.Length; i++)
        //{
        //    b[i] = ReadTensorFromImageFile(a[i], graph);
        //    Console.Write($"Loading image: {i} {a[i]}...");
        //    Console.CursorLeft = 0;
        //}
        // graph.Exit();
        // Faster Approch
        Parallel.For(0, a.Length, (i) =>
        {
            try
            {
                var graph = tf.Graph().as_default();
                b[i] = ReadTensorFromImageFile(a[i], graph);
                Console.WriteLine($"Loading image: {i} {a[i]}...");
                Console.CursorLeft = 0;
                graph.Exit();
            }
            catch(Exception ex) { Console.WriteLine(ex.Message);}
        });

        Console.WriteLine();
        Console.WriteLine($"Loaded {a.Length} images for " + process);
    }

    private NDArray ReadTensorFromImageFile(string fileName, Graph graph)
    {
        var fileReader = tf.io.read_file(fileName, "file_reader");
        var decodeImage = tf.image.decode_jpeg(fileReader, channels: 3, name: "DecodeJpeg");
        //var decodeImage = tf.image.decode_image(fileReader, channels: 3, name: "DecodeImage");
        // Change Format to Float32 bit
        var cast = tf.cast(decodeImage, tf.float32,"cast");

        //resize required one extra dims
        var dims_expander = tf.expand_dims(cast, 0);

        var resize = tf.constant(new int[] { 32, 32 }, name: "resize");

        var bilinear = tf.image.resize_bilinear(dims_expander, resize);//(dims_expander, resize);
        var sub = tf.subtract(bilinear, new float[] { 0 });
        var normalized = tf.divide(sub, new float[] { 255 });

        var sess = tf.Session(graph);
        return sess.run(normalized);

    }

    public List<DataRow> ReadCsv(string path)
    {
       
        using var reader = new StreamReader(path);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        //csv.Context.RegisterClassMap<DataRowMap>();
        return [..csv.GetRecords<DataRow>().Shuffle()]; ;
    }

    public void CreateImage(Dictionary<string, List<float>> history, string path)
    {

        foreach (var (name,data) in history)
        {

        }
    }
}

public class TrafficSignal
{
    private ILayersApi layers = tf.keras.layers;

    private IModel model { get; set; }

/// <summary>
/// Build You model
/// </summary>
/// <param name="height">Height of Image</param>
/// <param name="width">Width of Image</param>
/// <param name="depth">Number Of Channels</param>
/// <param name="classNumber">Total number of Classification labels</param>
    public void BuildModel(int height, int width, int depth, int classNumber)
    {
        var inputs = layers.Input(shape: (height, width, depth), name: "img");  //(32, 32, 3), name: "img");

        // convolutional layer
        var x = layers.Conv2D(8, (5, 5), padding: "same", activation: "relu").Apply(inputs);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Conv2D(16, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);

        x = layers.Conv2D(16, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Conv2D(32, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);

        x = layers.Conv2D(32, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Flatten().Apply(x);
        x = layers.Dense(128, activation: "relu").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.Dropout(0.5f).Apply(x);

        x = layers.Flatten().Apply(x);
        x = layers.Dense(128, activation: "relu").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.Dropout(0.5f).Apply(x);

        // output layer
        var outputs = layers.Dense(classNumber, "softmax").Apply(x);
        // build keras model
        model = tf.keras.Model(inputs, outputs, name: "traffic_resnet");
    }

/// <summary>
/// Start Tranning of Image 
/// </summary>
/// <param name="xTrain"></param>
/// <param name="yTrain"></param>
/// <param name="classWeight"></param>
/// <returns></returns>
    public ICallback Train(NDArray xTrain, NDArray yTrain,Dictionary<int,float> classWeight=null)
    {
        // training
        //model.fit(xTrain[new Slice(0, 2000)], yTrain[new Slice(0, 2000)],
        return model!.fit(xTrain, yTrain,
        batch_size: 64,
            epochs: 10,
            validation_split: 0.2f,
            class_weight: classWeight);
    }

/// <summary>
/// Summary of Model trainned
/// </summary>
/// <exception cref="NullReferenceException"></exception>
    public void Summary()
    {
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");
        
        model.summary();
    }

/// <summary>
/// Compile the Model
/// </summary>
    public void Compile()
    {
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");

        model!.compile(optimizer: tf.keras.optimizers.RMSprop(1e-3f),
            loss: tf.keras.losses.CategoricalCrossentropy(from_logits:false), // SparseCategoricalCrossentropy(from_logits: true),
            metrics: [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()]); //new[] { "acc" }); // //
    }

/// <summary>
/// Save Model Weight
/// </summary>
/// <param name="filePath"></param>

    public void Save(string filePath) //"./toy_resnet_model"
    {
        // save the model
        model!.save(filePath);
        
    }

/// <summary>
/// Predict Model based on Input
/// </summary>
/// <param name="value"></param>
/// <param name="verbose"></param>
/// <returns>Tensor of size No. of Example * labelNumber</returns>
    public Tensor Predict(Tensor value,int verbose = 0)
    {
        // var c = confusion_matrix;
        var result = model.predict(value, verbose: verbose);
        return tf.arg_max(result, 1);
    }
}


public sealed class DataRowMap : CsvHelper.Configuration.ClassMap<DataRow>
{
    // Only Required ClassId & Path
    public DataRowMap()
    {
        //Map(m => m.Width).Name("Width");
        //Map(m => m.Height).Name("Height");
        //Map(m => m.RoiX1).Name("Roi.X1");
        //Map(m => m.RoiY1).Name("Roi.Y1");
        //Map(m => m.RoiX2).Name("Roi.X2");
        //Map(m => m.RoiY2).Name("Roi.Y2");
        Map(m => m.ClassId).Name("ClassId");
        Map(m => m.Path).Name("Path");
    }
}

public class DataRow
{
    //public int Width { get; set; }
    //public int Height { get; set; }
    //public int RoiX1 { get; set; }
    //public int RoiY1 { get; set; }
    //public int RoiX2 { get; set; }
    //public int RoiY2 { get; set; }
    public int ClassId { get; set; }
    public string Path { get; set; }
}
