using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;


var c = new ConfusionMatrix(confusionTableCounts:new double[][],isBinary:false);

var m = new MLContext();

m.MulticlassClassification.Evaluate(new DataFrame());