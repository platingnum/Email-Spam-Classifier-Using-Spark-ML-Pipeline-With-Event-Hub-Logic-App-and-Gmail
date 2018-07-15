# Email Spam Classifier Using Spark ML Pipeline With Event Hub, Logic App and Gmail

## Import required python libraries

```python
from datetime import datetime as dt
from more_itertools import *
from bs4 import BeautifulSoup
import json
import io
import re

import pyspark
from pyspark import since, keyword_only
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.rdd import ignore_unicode_prefix
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaTransformer, _jvm
from pyspark.ml.common import inherit_doc
from pyspark.ml import Transformer
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.mllib.linalg import (Vector, Vectors, DenseVector, SparseVector, _convert_to_vector)
from pyspark.ml.classification import LinearSVC
```

## Apply Machine Learning

### Import Training Email Dataset

```python
emails =(spark.read.format("csv")
         .option("header", "true")
         .option("inferSchema", "true")
         .option("multiLine", "true")
         .load("/FileStore/tables/email/emails.csv").where("spam=1 or spam=0"))
```

### Define a custom data transformer

```python
from pyspark.ml import Transformer

class LabelPointTF(Transformer, HasInputCol, HasOutputCol):
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(LabelPointTF, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)

  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)
  
  def _transform(self, dataset):
    t = StringType()
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    return dataset.withColumn(out_col, udf(lambda x: LabeledPoint(1, Vectors.fromML(x)), t)(in_col))
```

### Create data processing pipeline

```python
# Configure an ML pipeline, which consists of four stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
labelPointTF = LabelPointTF(inputCol=hashingTF.getOutputCol(), outputCol="vectors")
lsvc = LinearSVC(maxIter=10, regParam=0.1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, labelPointTF, lsvc])
```

### Train Email Spam Classifier Model

```python
model = pipeline.fit(emails.select("text", column("spam").alias("label").cast(IntegerType())))
```

### Prepare Event Hub connection

```python
# Event Hub Connection string
connectionString = "Endpoint=sb://<your-eventhubs-namespace>.servicebus.windows.net/;SharedAccessKeyName=<your-policy-name>;SharedAccessKey=<your-key>=;EntityPath=<your-event-hub-entity-path>"

ehConf = {}
ehConf['eventhubs.connectionString'] = connectionString
ehConf['eventhubs.consumerGroup'] = "$Default"

# Start from beginning of stream
startOffset = "-1"

# End at the current time. This datetime formatting creates the correct string format from a python datetime object
endTime = dt.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

# Create the positions
startingEventPosition = {
  "offset": startOffset,  
  "seqNo": -1,            #not in use
  "enqueuedTime": None,   #not in use
  "isInclusive": True
}

endingEventPosition = {
  "offset": None,           #not in use
  "seqNo": -1,              #not in use
  "enqueuedTime": endTime,
  "isInclusive": True
}

# Put the positions into the Event Hub config dictionary
ehConf["eventhubs.startingPosition"] = json.dumps(startingEventPosition)
ehConf["eventhubs.endingPosition"] = json.dumps(endingEventPosition)
```

### Create Streaming DataFrame from Event Hub

```python
# Source with default settings
df = (spark
      .readStream
      .format("eventhubs")
      .options(**ehConf)
      .load())
```

### Extract Text from Email HTML Body

```python
def my_textFromHtml(html):
  soup = BeautifulSoup(html, "html5lib")
  for script in soup(["script", "style"]): # remove all javascript and stylesheet code
      script.extract()
  # get text
  text = soup.get_text()
  # break into lines and remove leading and trailing space on each
  lines = (line.strip() for line in text.splitlines())
  # break multi-headlines into a line each
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  # drop blank lines
  text = '\n'.join(chunk for chunk in chunks if chunk)
  return text

udf_textFromHtml = udf(lambda x: my_textFromHtml(x), StringType())
```

```python
email_df = df.select("properties.Message_Id", "properties.From", "properties.Subject", udf_textFromHtml(df.body.cast(StringType())).alias("text"), "properties.event_time")
```

### Transform Email DataFrame to get Prediction Dataframe

```python
predictions = model.transform(email_df).select("Message_Id", column("prediction").alias("Predicted_Spam"), "From", "Subject", "text" )
```

### Display Predicted Classification

```python
display(predictions)
```
