import numpy as np
import os

# Define dataset IDs
train_dataset_ids = [
    22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    37, 39, 40, 41, 42, 43, 48, 49, 50, 53, 54, 55,
    56, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
    187, 188, 275, 276, 277, 278, 285, 300, 301, 307, 308,
    310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
    337, 338, 339, 340, 342, 343, 346, 372, 375
]
test_dataset_ids = [
    1503, 23517, 1551, 1552, 183, 255, 545, 546, 475, 481, 
    516, 3, 6, 8, 10, 12, 14, 9, 11, 5
]

# Randomly split train_dataset_ids into train and validation sets
np.random.seed(42)
random_rank = np.random.permutation(train_dataset_ids)
ids_train = random_rank[:int(len(random_rank) * 0.8)]
ids_val = random_rank[int(len(random_rank) * 0.8):]

# Save the splits
output_dir = '../result/openml/default/'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'ids_train.npy'), ids_train)
np.save(os.path.join(output_dir, 'ids_val.npy'), ids_val)
np.save(os.path.join(output_dir, 'ids_test.npy'), test_dataset_ids)