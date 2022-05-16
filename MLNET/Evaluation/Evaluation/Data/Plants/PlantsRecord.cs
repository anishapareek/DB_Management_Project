using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Data.Plants
{
    public class PlantsRecord
    {
        [LoadColumn(0)]
        public string Name { get; set; }

        [LoadColumn(1, 69)]
        [VectorType(69)]
        public string[] States { get; set; }
    }
}
