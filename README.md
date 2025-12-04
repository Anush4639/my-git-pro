# src/utils.py
from pyspark.sql import SparkSession

def create_spark(app_name="movie-recommender", master="local[*]"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    return spark


# src/etl.py
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from utils import create_spark

def load_data(spark, ratings_path, movies_path):
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    movies  = spark.read.csv(movies_path, header=True, inferSchema=True)
    return ratings, movies

def preprocess(ratings):
    # convert timestamp to timestamp and extract day/hour
    ratings = ratings.withColumn("ts", F.to_timestamp(F.col("timestamp"))) \
                     .withColumn("year", F.year("ts")) \
                     .withColumn("month", F.month("ts")) \
                     .withColumn("day", F.dayofmonth("ts")) \
                     .withColumn("hour", F.hour("ts"))
    return ratings

def top_n_movies(ratings, movies, n=20):
    counts = ratings.groupBy("movieId") \
                    .agg(F.count("*").alias("rating_count"),
                         F.avg("rating").alias("avg_rating")) \
                    .orderBy(F.desc("rating_count"))
    return counts.join(movies, on="movieId", how="left").select("movieId","title","rating_count","avg_rating").limit(n)

def genre_popularity(ratings, movies):
    # explode genres from movies
    movies_exp = movies.withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    joined = ratings.join(movies_exp, on="movieId", how="left")
    g = joined.groupBy("genre").agg(F.count("*").alias("ratings"), F.avg("rating").alias("avg_rating")).orderBy(F.desc("ratings"))
    return g

def hourly_trend(ratings):
    return ratings.groupBy("hour").agg(F.count("*").alias("ratings"), F.avg("rating").alias("avg_rating")).orderBy("hour")

def write_parquet(df, path):
    df.write.mode("overwrite").parquet(path)

def main(ratings_path, movies_path, out_dir):
    spark = create_spark()
    ratings, movies = load_data(spark, ratings_path, movies_path)
    ratings = preprocess(ratings)
    print("Total ratings:", ratings.count())

    top20 = top_n_movies(ratings, movies, n=20)
    print("Top 20 movies by count")
    top20.show(truncate=False)

    genres = genre_popularity(ratings, movies)
    print("Top genres")
    genres.show(truncate=False)

    hourly = hourly_trend(ratings)
    print("Hourly trend")
    hourly.show()

    # write results
    write_parquet(top20, f"{out_dir}/top20_movies")
    write_parquet(genres, f"{out_dir}/genre_stats")
    write_parquet(hourly, f"{out_dir}/hourly_trend")
    # also write preprocessed ratings for training
    write_parquet(ratings.select("userId","movieId","rating","ts","year","month","day","hour"), f"{out_dir}/ratings_preprocessed")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: etl.py <ratings.csv> <movies.csv> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

# src/train_als.py
import sys
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from utils import create_spark

def load_preprocessed(spark, ratings_parquet_path):
    return spark.read.parquet(ratings_parquet_path).select("userId","movieId","rating")

def train_als(ratings):
    # ensure integer types
    ratings = ratings.withColumn("userId", F.col("userId").cast("int")).withColumn("movieId", F.col("movieId").cast("int")).withColumn("rating", F.col("rating").cast("float"))
    # split
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", nonnegative=True, rank=20, maxIter=12, regParam=0.1)
    model = als.fit(train)
    preds = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(preds)
    return model, rmse

def top_k_recommendations(model, users_df, k=10):
    # model.recommendForUserSubset requires Spark DataFrame of users
    recs = model.recommendForUserSubset(users_df, k)
    # explode recs to rows
    exploded = recs.select("userId", F.explode("recommendations").alias("rec"))
    return exploded.select("userId", F.col("rec.movieId").alias("movieId"), F.col("rec.rating").alias("pred_rating"))

def main(ratings_parquet, output_dir):
    spark = create_spark("als-train")
    ratings = load_preprocessed(spark, ratings_parquet)
    model, rmse = train_als(ratings)
    print(f"ALS RMSE: {rmse:.4f}")
    # produce top 10 for all users (be mindful of size)
    users = ratings.select("userId").distinct()
    recs = top_k_recommendations(model, users, k=10)
    recs.write.mode("overwrite").parquet(f"{output_dir}/user_recommendations")
    # Optionally save model
    model.save(f"{output_dir}/als_model")
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: train_als.py <ratings_parquet> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt



spark-submit src/etl.py data/ratings.csv data/movies.csv output




spark-submit src/train_als.py output/ratings_preprocessed output



