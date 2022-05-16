using Evaluation.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Evaluations
{
    public class TrainerBase<TMetric> : ITrainerBase
    {
        private readonly string modelName;
        private readonly IEstimator<ITransformer> model;
        private readonly IData<TMetric> data;
        private readonly Stopwatch stopwatch;

        public TrainerBase(string modelName, IEstimator<ITransformer> model, IData<TMetric> data)
        {
            this.modelName = modelName;
            this.model = model;
            this.data = data;
            stopwatch = new();
        }

        public void Evaluate()
        {
            stopwatch.Start();
            var dataSplit = data.LoadAndPrepareData();
            stopwatch.Stop();
            TimeSpan dataLoadingTime = stopwatch.Elapsed;

            EstimatorChain<ITransformer> pipeline = new();
            foreach (var estimator in data.GetTransformers())
                pipeline = pipeline.Append(estimator);
            pipeline = data.AppendCacheCheckpoint(pipeline);
            
            var trainingPipeline = pipeline.Append(model);

            stopwatch.Start();
            var _trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
            stopwatch.Stop();
            TimeSpan trainingTime = stopwatch.Elapsed;

            var testSetTransform = _trainedModel.Transform(dataSplit.TestSet);

            stopwatch.Start();
            var metric = data.Evaluate(testSetTransform);
            stopwatch.Stop();
            TimeSpan evaluationTime = stopwatch.Elapsed;

            var metrics = data.SaveMetrics(modelName, dataLoadingTime, trainingTime, evaluationTime, metric);
            Console.WriteLine("Results: ");
            Console.WriteLine(metrics);
        }
    }
}
