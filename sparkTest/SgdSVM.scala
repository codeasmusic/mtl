
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.ml.feature.{HashingTF, IndexToString, StringIndexer, Tokenizer}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by cike on 17-6-14.
  */
object SgdBagging {
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
			.fit(trainSet)
		val indexData = labelIndexer.transform(trainSet)

		val tokenizer = new Tokenizer()
			.setInputCol("query")
			.setOutputCol("words")
		val wordsData = tokenizer.transform(indexData)

		val hashingTF = new HashingTF()
			.setNumFeatures(1000)
			.setInputCol(tokenizer.getOutputCol)
			.setOutputCol("features")
		val tfData = hashingTF.transform(wordsData)

		tfData.printSchema()
		println(tfData.first())

		val trainRDD = tfData.rdd.map(
			row => LabeledPoint(
				row.getAs[Double]("indexedGender"),
				org.apache.spark.mllib.linalg.Vectors.fromML(
					row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
			)
		)

		val svm = SVMWithSGD.train(trainRDD, numIterations=10)
//		svm.clearThreshold()	//default threshold is 0.0

		val testSet = spark.read
			.option("header", true)
			.csv("/home/cike/software/spark/sparkData/test_id_age_gender_edu_query.csv")

		val testRDD = hashingTF.transform(tokenizer.transform(labelIndexer.transform(testSet))).rdd.map(
			row => LabeledPoint(
				row.getAs[Double]("indexedGender"),
				org.apache.spark.mllib.linalg.Vectors.fromML(
					row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
			)
		)

		val scoreAndLabels = testRDD.map{
			point =>
				val prediction = svm.predict(point.features)
				(prediction, point.label)
		}
		val acc = scoreAndLabels.filter(p => p._1 == p._2).count().toDouble / scoreAndLabels.count()
		println(s"Test error: $acc")

		val labelConverter = new IndexToString()
			.setInputCol("indexedPreds")
			.setOutputCol("predictions")
			.setLabels(labelIndexer.labels)
		val indexedPredDF = spark.createDataFrame(scoreAndLabels).toDF("indexedPreds", "label")
		val predDF = labelConverter.transform(indexedPredDF)

		predDF.show()
		predDF.write
			.mode(saveMode="append")
			.option("header", true)
			.csv("/home/cike/software/spark/sparkData/predictions")

	}
}
