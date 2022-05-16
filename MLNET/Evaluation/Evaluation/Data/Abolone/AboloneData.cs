using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Evaluation.Data.Abolone
{
    public class AboloneData : IData<MulticlassClassificationMetrics>
    {
        private readonly MLContext MlContext;
        private readonly string filePath, folderPath;

        public AboloneData(MLContext mLContext, string folderPath)
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
            yield return MlContext.Transforms.Conversion.MapValueToKey("Label", "Rings");
        }

        public EstimatorChain<ITransformer> AppendCacheCheckpoint(IEstimator<ITransformer> pipeline) => pipeline.AppendCacheCheckpoint(MlContext);

        public MulticlassClassificationMetrics Evaluate(IDataView dataView)
        {
            return MlContext.MulticlassClassification.Evaluate(dataView);
        }

        public DataOperationsCatalog.TrainTestData LoadAndPrepareData()
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<AboloneRecord>(filePath, separatorChar: ',', hasHeader: false);
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        public string SaveMetrics(string modelType, TimeSpan dataLoadingTime, TimeSpan trainingTime, TimeSpan evaluationTime, MulticlassClassificationMetrics metric)
        {
            var metrics = JsonSerializer.Serialize(new
            {
                DataLoadingTime = dataLoadingTime.TotalSeconds,
                TrainingTime = trainingTime.TotalSeconds,
                EvaluationTime = evaluationTime.TotalSeconds,
                metric.MicroAccuracy,
                metric.MacroAccuracy,
                metric.LogLoss,
                metric.LogLossReduction,
                ConfusionMatrix = metric.ConfusionMatrix.GetFormattedConfusionTable()
            }, options: new() { WriteIndented = true });

            string metricsPath = Path.Combine(folderPath, $"abolone_{modelType}.json");
            File.WriteAllText(metricsPath, metrics);

            return metrics;
        }
    }
}
