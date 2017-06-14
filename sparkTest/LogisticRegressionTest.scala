/**
  * Created by cike on 17-6-14.
  */

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IndexToString, StringIndexer, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row


object LogisticRegressionTest {
	def main(args: Array[String]): Unit ={
		val spark = SparkSession
			.builder()
			.appName("LR Test")
		    .getOrCreate()

		val trainSet = spark.read
			.option("header", true)
			.csv("/home/cike/software/spark/sparkData/train_id_age_gender_edu_query.csv")

		trainSet.printSchema()

		val labelIndexer = new StringIndexer()
		    .setInputCol("gender")
		    .setOutputCol("indexedGender")
		    .fit(trainSet)		//call fit(), otherwise "labelIndexer.labels" can't be resolve

		val tokenizer = new Tokenizer()
		    .setInputCol("query")
		    .setOutputCol("words")

		val hashingTF = new HashingTF()
		    .setNumFeatures(1000)
		    .setInputCol(tokenizer.getOutputCol)
		    .setOutputCol("features")

		val lr = new LogisticRegression()
			.setLabelCol("indexedGender")
			.setFeaturesCol("features")
		    .setMaxIter(10)
		    .setRegParam(0.001)

		val labelConverter = new IndexToString()
		    .setInputCol("prediction")
		    .setOutputCol("predictedGender")
		    .setLabels(labelIndexer.labels)

		val pipeline = new Pipeline()
		    .setStages(Array(labelIndexer, tokenizer, hashingTF, lr, labelConverter))

		val model = pipeline.fit(trainSet)

		model.write.overwrite().save("/home/cike/spark-lr-model")
		pipeline.write.overwrite().save("/home/cike/unfit-lr-model")

		val sameModel = PipelineModel.load("/home/cike/spark-lr-model")

		val testSet = spark.read
			.option("header", true)
			.csv("/home/cike/software/spark/sparkData/test_id_age_gender_edu_query.csv")

		sameModel.transform(testSet)
		    .select("id", "probability", "predictedGender")
		    .collect()
		    .foreach{
				// the type of "prediction" is the same as that of trainSet.printSchema()
				case Row(id: String, prob: Vector, prediction: String) =>
					println(s"$id : prob=$prob, prediction=$prediction")
//				case what =>
//					println(what)
			}
	}
}





