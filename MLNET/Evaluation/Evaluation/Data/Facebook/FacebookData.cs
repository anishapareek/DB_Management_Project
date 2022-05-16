using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Evaluation.Data.Facebook
{
    public class FacebookData : IData<RegressionMetrics>
    {
        private readonly MLContext MlContext;
        private readonly string filePath, folderPath;

        public FacebookData(MLContext mLContext, string folderPath)
        {
            MlContext = mLContext;
            this.folderPath = folderPath;
            filePath = Path.Combine(folderPath, "facebook.csv");
        }

        public IEnumerable<IEstimator<ITransformer>> GetTransformers()
        {
            yield return MlContext.Transforms.Concatenate("Features",
                "PagePopularity",
                "PageCheckins",
                "PageTalkingAbout",
                "Derived",
                "CC1",
                "CC2",
                "CC3",
                "CC4",
                "CC5",
                "BaseTime",
                "PostLength",
                "PostShareCount",
                "PostPromotionStatus",
                "HLocal",
                "PostPublishedWeekday",
                "BaseDateTimeWeekday");
            yield return MlContext.Transforms.NormalizeMinMax("Features", "Features");
            yield return MlContext.Transforms.CopyColumns("Label", "TargetVariable");
        }

        public EstimatorChain<ITransformer> AppendCacheCheckpoint(IEstimator<ITransformer> pipeline) => pipeline.AppendCacheCheckpoint(MlContext);

        public RegressionMetrics Evaluate(IDataView dataView)
        {
            return MlContext.Regression.Evaluate(dataView);
        }

        public DataOperationsCatalog.TrainTestData LoadAndPrepareData()
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<FacebookRecord>(filePath, separatorChar: ',', hasHeader: false);
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        public string SaveMetrics(string modelType, TimeSpan dataLoadingTime, TimeSpan trainingTime, TimeSpan evaluationTime, RegressionMetrics metric)
        {
            var metrics = JsonSerializer.Serialize(new
            {
                DataLoadingTime = dataLoadingTime.TotalSeconds,
                TrainingTime = trainingTime.TotalSeconds,
                EvaluationTime = evaluationTime.TotalSeconds,
                metric.MeanSquaredError,
                metric.MeanAbsoluteError,
                metric.RSquared,
                metric.RootMeanSquaredError,
            }, options: new() { WriteIndented = true });

            string metricsPath = Path.Combine(folderPath, $"facebook_{modelType}.json");
            File.WriteAllText(metricsPath, metrics);

            return metrics;
        }
    }
}
