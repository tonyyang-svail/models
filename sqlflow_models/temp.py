import tensorflow as tf
import sqlflow_models


def gen():
    c1 = [
        [1, 5, 2],
        [3],
        [2, 4, 6, 8],
    ]
    label = [0, 1, 0]
    for i in range(len(c1)):
        yield ({"c1": c1[i]}, [label[i]])


ds = tf.data.Dataset.from_generator(gen, ({"c1": tf.int64}, tf.int64), ({"c1": tf.TensorShape([None])}, tf.TensorShape([1])))
ds = ds.batch(1)

fea = tf.feature_column.categorical_column_with_identity(
    key="c1",
    num_buckets=10
)
emb = tf.feature_column.embedding_column(
    fea,
    dimension=3)
feature_columns = [emb]
model = sqlflow_models.DNNClassifier(feature_columns=feature_columns)

model.compile(optimizer=model.default_optimizer(),
              loss=model.default_loss(),
              metrics=["accuracy"])

model.fit(ds,
          epochs=model.default_training_epochs(),
          steps_per_epoch=100, verbose=0)
