using Newtonsoft.Json;
using QuickChart;
var info = File.ReadAllText("History.json");
var history = JsonConvert.DeserializeObject<Dictionary<string,List<float>>>(info);

foreach(var (name,d) in history!){Console.WriteLine(name);}
string GenerateColor()
{
    Random random = new Random();
    byte[] rgb = new byte[3];

    random.NextBytes(rgb);
    return $"#{rgb[0]:X2}{rgb[1]:X2}{rgb[2]:X2}";
}

List<Dictionary<string, object>> datasets = new();
foreach (var (name, d) in history!)
{
    datasets.Add(
       new ()
       {
           {"label",name},
           {"backgroundColor",GenerateColor()},
           {"data",d},
           {"fill",false}
       }
        );
}

var chart =new
{
    type="line",
    data=new
    {
        labels=Enumerable.Range(1,10).ToArray(),
        datasets =datasets
    },
    options= new
    {
        title=new
        {
            display=true,
            text="Graph DATA"
        },
        legend= new Dictionary<string, object>
        {
            { "display", true }
        },
        scales = new
        { xAxes = new[]{new
            {
                scaleLabel=new{display=true,labelString=" Epochs --> "}
            }}
        }
    
}
};

var res = new Chart();
res.Height = 400;
res.Width  = 600;
res.Config = JsonConvert.SerializeObject(chart);
Console.WriteLine(JsonConvert.SerializeObject(chart));

Console.WriteLine(res.GetUrl());
Console.WriteLine(res.GetShortUrl());
res.ToFile("Chart.png");