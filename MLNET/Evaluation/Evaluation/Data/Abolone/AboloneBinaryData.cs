using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Evaluation.Data.Abolone
{
    public class AboloneBinaryData : IData<BinaryClassificationMetrics>
    {
        private readonly MLContext MlContext;
        private readonly string filePath, folderPath;

        public AboloneBinaryData(MLContext mLContext, string folderPath)
        {
            MlContext = mLContext;
            this.folderPath = folderPath;
            filePath = Path.Combine(folderPath, "abalone.data");
        }

        public IEnumerable<IEstimator<ITransformer>> GetTransformers()
        {
            yield return MlContext.Transforms.Text.FeaturizeText("Sex", "Sex");
            yield return MlContext.Transforms.Concatenate("Features", "Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "ViseraWeight", "ShellWeight");
            yield return MlContext.Transforms.NormalizeMinMax("Features", "Features");
            yield return MlContext.Transforms.Expression("Label", "x => x > 10", "Rings");
        }

        public EstimatorChain<ITransformer> AppendCacheCheckpoint(IEstimator<ITransformer> pipeline) => pipeline.AppendCacheCheckpoint(MlContext);

        public BinaryClassificationMetrics Evaluate(IDataView dataView)
        {
            return MlContext.BinaryClassification.EvaluateNonCalibrated(dataView);
        }

        public DataOperationsCatalog.TrainTestData LoadAndPrepareData()
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<AboloneRecord>(filePath, separatorChar: ',', hasHeader: false);
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        public string SaveMetrics(string modelType, TimeSpan dataLoadingTime, TimeSpan trainingTime, TimeSpan evaluationTime, BinaryClassificationMetrics metric)
        {
            var metrics = JsonSerializer.Serialize(new
            {
                DataLoadingTime = dataLoadingTime.TotalSeconds,
                TrainingTime = trainingTime.TotalSeconds,
                EvaluationTime = evaluationTime.TotalSeconds,
                metric.Accuracy,
                metric.F1Score,
                metric.ConfusionMatrix,
            }, options: new() { WriteIndented = true });

            string metricsPath = Path.Combine(folderPath, $"abolone_{modelType}.json");
            File.WriteAllText(metricsPath, metrics);

            return metrics;
        }
    }
}
