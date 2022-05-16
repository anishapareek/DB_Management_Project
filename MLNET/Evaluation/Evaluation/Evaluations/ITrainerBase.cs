using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Evaluations
{
    public interface ITrainerBase
    {
        void Evaluate();
    }
}
