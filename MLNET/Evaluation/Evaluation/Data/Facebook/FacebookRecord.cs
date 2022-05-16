using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Evaluation.Data.Facebook
{
    public class FacebookRecord
    {
        [LoadColumn(0)]
        public float PagePopularity { get; set; }

        [LoadColumn(1)]
        public float PageCheckins { get; set; }

        [LoadColumn(2)]
        public float PageTalkingAbout { get; set; }

        [LoadColumn(3)]
        public string PageCategory { get; set; }

        [LoadColumn(4, 28)]
        [VectorType(25)]
        public float[] Derived { get; set; }

        [LoadColumn(29)]
        public float CC1 { get; set; }

        [LoadColumn(30)]
        public float CC2 { get; set; }

        [LoadColumn(31)]
        public float CC3 { get; set; }

        [LoadColumn(32)]
        public float CC4 { get; set; }

        [LoadColumn(33)]
        public float CC5 { get; set; }

        [LoadColumn(34)]
        public float BaseTime { get; set; }

        [LoadColumn(35)]
        public float PostLength { get; set; }

        [LoadColumn(36)]
        public float PostShareCount { get; set; }

        [LoadColumn(37)]
        public float PostPromotionStatus { get; set; }

        [LoadColumn(38)]
        public float HLocal { get; set; }

        [LoadColumn(39, 45)]
        [VectorType(7)]
        public float PostPublishedWeekday { get; set; }

        [LoadColumn(46, 52)]
        [VectorType(7)]
        public float BaseDateTimeWeekday { get; set; }

        [LoadColumn(53)]
        public float TargetVariable { get; set; }
    }
}
