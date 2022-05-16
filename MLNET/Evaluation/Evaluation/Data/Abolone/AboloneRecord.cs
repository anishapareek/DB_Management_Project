using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Data.Abolone
{
    public class AboloneRecord
    {
        [LoadColumn(0)]
        public string Sex { get; set; }

        [LoadColumn(1)]
        public float Length { get; set; }

        [LoadColumn(2)]
        public float Diameter { get; set; }

        [LoadColumn(3)]
        public float Height { get; set; }

        [LoadColumn(4)]
        public float WholeWeight { get; set; }

        [LoadColumn(5)]
        public float ShuckedWeight { get; set; }

        [LoadColumn(6)]
        public float ViseraWeight { get; set; }

        [LoadColumn(7)]
        public float ShellWeight { get; set; }

        [LoadColumn(8)]
        public int Rings { get; set; }
    }
}
