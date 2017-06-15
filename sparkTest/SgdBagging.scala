import org.apache.spark.ml.feature.{HashingTF, IndexToString, StringIndexer, Tokenizer}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

/**
  * Created by cike on 17-6-15.
  */
object SgdBagging {

	def main(args: Array[String]): Unit ={
		val spark = SparkSession
			.builder()
			.appName("LR Test")
			.getOrCreate()

		spark.sparkContext.setLogLevel("ERROR")

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

		val trainRDD = tfData.rdd.map(
			row => LabeledPoint(
				row.getAs[Double]("indexedGender"),
				org.apache.spark.mllib.linalg.Vectors.fromML(
					row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
			)
		)

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

		val numClassifiers = 2
		for (i <- 1 to numClassifiers){
			val svm = SVMWithSGD.train(trainRDD, numIterations=10)
			val scoreAndLabels = testRDD.map{
				point =>
					val prediction = svm.predict(point.features)
					(prediction, point.label)
			}
			val predDF = spark.createDataFrame(scoreAndLabels).toDF("predictions", "label")

			predDF.write
				.mode(saveMode="append")
				.option("header", true)
				.csv("/home/cike/software/spark/sparkData/predictions")
		}

		// code below is to find the most frequent value in each column
//		val results = for{
//			p <- predDFs
//			res = p.groupBy("column_name")
//				.count()
//				.orderBy(org.apache.spark.sql.functions.col("count").desc)
//				.first()
//			    .getAs[Double]("column_name")
//		} yield res
//
//		println(results)

	}
}
