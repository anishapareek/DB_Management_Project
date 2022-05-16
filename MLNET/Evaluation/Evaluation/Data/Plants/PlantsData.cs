using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Evaluation.Data.Plants
{
    public class PlantsData : IData<ClusteringMetrics>
    {
        private readonly MLContext MlContext;
        private readonly string filePath, folderPath, stateAbbrPath;

        public PlantsData(MLContext mLContext, string folderPath)
        {
            MlContext = mLContext;
            this.folderPath = folderPath;
            filePath = Path.Combine(folderPath, "plants.data");
            stateAbbrPath = Path.Combine(folderPath, "stateabbr.txt");
        }

        public IEnumerable<IEstimator<ITransformer>> GetTransformers()
        {
            string[] features = new string[69];

            int i = 0;
            foreach (string line in File.ReadLines(stateAbbrPath))
            {
                var lineParsed = line.Split(' ', 2);
                var abbr = lineParsed[0];
                var name = lineParsed[1];

                features[i] = abbr;
                i++;
            }

            void mapping(PlantsRecord input, PlantsTransformedRecord output)
            {
                output.Features = new float[input.States.Length];
                output.Name = input.Name;
                for (int i = 0; i < features.Length; i++)
                    output.Features[i] = input.States.Contains(features[i]) ? 1 : 0;
            }

            yield return MlContext.Transforms.CustomMapping((Action<PlantsRecord, PlantsTransformedRecord>)mapping, contractName: null);
        }

        public EstimatorChain<ITransformer> AppendCacheCheckpoint(IEstimator<ITransformer> pipeline) => pipeline.AppendCacheCheckpoint(MlContext);

        public ClusteringMetrics Evaluate(IDataView dataView)
        {
            return MlContext.Clustering.Evaluate(dataView);
        }

        public DataOperationsCatalog.TrainTestData LoadAndPrepareData()
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<PlantsRecord>(filePath, separatorChar: ',', hasHeader: false);
            
            //EstimatorChain<ITransformer> pipeline = new();
            //foreach (var estimator in GetTransformers())
            //    pipeline = pipeline.Append(estimator);
            //pipeline = AppendCacheCheckpoint(pipeline);

            //var p = pipeline.Fit(trainingDataView);
            //var t = p.Transform(trainingDataView);

            //File.WriteAllLines("plants.output",
            //    MlContext.Data.CreateEnumerable<PlantsTransformedRecord>(t, false)
            //    .Select(t => t.Name + "," + string.Join(',', t.Features)));

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        public string SaveMetrics(string modelType, TimeSpan dataLoadingTime, TimeSpan trainingTime, TimeSpan evaluationTime, ClusteringMetrics metric)
        {
            var metrics = JsonSerializer.Serialize(new
            {
                DataLoadingTime = dataLoadingTime.TotalSeconds,
                TrainingTime = trainingTime.TotalSeconds,
                EvaluationTime = evaluationTime.TotalSeconds,
                metric.AverageDistance,
            }, options: new() { WriteIndented = true });

            string metricsPath = Path.Combine(folderPath, $"plants_{modelType}.json");
            File.WriteAllText(metricsPath, metrics);

            return metrics;
        }
    }
}
