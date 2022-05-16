using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Data
{
    public interface IData<TMetric>
    {
        public IEnumerable<IEstimator<ITransformer>> GetTransformers();

        public EstimatorChain<ITransformer> AppendCacheCheckpoint(IEstimator<ITransformer> pipeline);

        public DataOperationsCatalog.TrainTestData LoadAndPrepareData();

        public TMetric Evaluate(IDataView dataView);

        public string SaveMetrics(string modelType, TimeSpan dataLoadingTime, TimeSpan TrainingTime, TimeSpan EvaluationTime, TMetric metric);
    }
}
