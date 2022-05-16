using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Evaluation.Data.Abolone;
using Evaluation.Data.Facebook;
using Evaluation.Data.Plants;
using Evaluation.Evaluations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Evaluation
{
    class Program
    {
        private const string commandLineString = "Please supply a list models to evaluate. Ex: \"Random Forest, Multilayer Perceptron, Linear Regression, K-Means\"";
        private static readonly Regex sWhitespace = new(@"\s+", RegexOptions.Compiled);

        private static readonly Dictionary<string, Func<MLContext, string, ITrainerBase>> 
            classificationModels = new()
            {
                {
                    PreprocessInput("Random Forest"),
                    (mlContext, filePath) => new TrainerBase<BinaryClassificationMetrics>(
                        PreprocessInput("Random Forest"),
                        mlContext.BinaryClassification.Trainers.FastForest(numberOfLeaves: 50, numberOfTrees: 300),
                        new AboloneBinaryData(mlContext, filePath))
                },
                { 
                    PreprocessInput("Multilayer Perceptron"),
                    (mlContext, filePath) => new TrainerBase<BinaryClassificationMetrics>(
                        PreprocessInput("Multilayer Perceptron"),
                        mlContext.BinaryClassification.Trainers.AveragedPerceptron(decreaseLearningRate: true, numberOfIterations: 100),
                        new AboloneBinaryData(mlContext, filePath))
                },
                {
                    PreprocessInput("Stochastic Dual Coordinate Ascent"),
                    (mlContext, filePath) => new TrainerBase<MulticlassClassificationMetrics>(
                        PreprocessInput("Stochastic Dual Coordinate Ascent"),
                        mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(),
                        new AboloneData(mlContext, filePath))
                },
            },
            regressionModels = new()
            {
                { 
                    PreprocessInput("Linear Regression"),
                    (mlContext, filePath) => new TrainerBase<RegressionMetrics>(
                        PreprocessInput("Linear Regression"),
                        mlContext.Regression.Trainers.FastTreeTweedie(),
                        new FacebookData(mlContext, filePath))
                }
            },
            clusteringModels = new()
            {
                { 
                    PreprocessInput("K-Means"),
                    (mlContext, filePath) => new TrainerBase<ClusteringMetrics>(
                        PreprocessInput("K-Means"),
                        mlContext.Clustering.Trainers.KMeans(numberOfClusters: 6),
                        new PlantsData(mlContext, filePath))
                }
            };
        
        private static readonly Dictionary<string, Func<MLContext, string, ITrainerBase>> models = classificationModels
            .Concat(regressionModels)
            .Concat(clusteringModels)
            .ToDictionary(model => model.Key, model => model.Value);
        private static readonly string[] modelNames = models.Select(model => model.Key).ToArray();

        private static string PreprocessInput(string input) => sWhitespace.Replace(input, "").ToLower();

        static Program() { }

        static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine(commandLineString);
                return;
            }

            string[] givenModels = args[0].Split(',').Select(model => PreprocessInput(model)).ToArray();
            if (givenModels.Except(modelNames).Any())
            {
                Console.WriteLine(commandLineString);
                return;
            }

            MLContext mLContext = new();
            string dataPath = Path.GetFullPath("Data");

            foreach (var modelName in givenModels)
            {
                Console.WriteLine($"Evaluating: {modelName}");
                Console.WriteLine("");
                var model = models[modelName]?.Invoke(mLContext, dataPath);
                model.Evaluate();
                Console.WriteLine("");
            }

            Console.WriteLine($"Finished Evaluating");
        }
    }
}
