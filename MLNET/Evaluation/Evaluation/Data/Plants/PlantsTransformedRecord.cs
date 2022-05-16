using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Data.Plants
{
    public class PlantsTransformedRecord
    {
        public string Name { get; set; }

        [VectorType(69)]
        public float[] Features { get; set; }
    }
}
