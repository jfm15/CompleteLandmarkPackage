import json

import numpy as np

train_path = "../cv/set1/train.txt"
test_path = "../cv/set1/test.txt"
partition_name = "partition_set_1"

train_ids = np.loadtxt(train_path, dtype=str)
test_ids = np.loadtxt(test_path, dtype=str)

partition_ids = {
    "training": train_ids.tolist(),
    "validation": test_ids.tolist(),
    "testing": []
}

partition_file = open("../{}.json".format(partition_name), "w")
json.dump(partition_ids, partition_file)
partition_file.close()
