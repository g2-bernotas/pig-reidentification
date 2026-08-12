# Core libraries
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data

# Local libraries
from utilities.utils import Utilities, getDevice
from models.embeddings import resnet50

"""
Infers the embeddings of the train/test portions of a dataset with a trained
checkpoint and evaluates identification performance by classifying the test
embeddings with KNN against the training embeddings.

The final accuracy is printed as "Accuracy=<value>" so that train.py can pick
it up when calling this script in a subprocess.

Minimal example:

    python test.py \
        --model_path output/inssegm/fold_0/best_model_state.pkl \
        --dataset_root data/OpenSetPigs \
        --folds_file data/OpenSetPigs/splits/50-50.json \
        --save_path output/inssegm/fold_0
"""

# For a trained model, let's evaluate it
def evaluateModel(args):
    # Load the relevant datasets (no augmentation during inference)
    train_dataset = Utilities.selectDataset(args, True, augment=False)
    test_dataset = Utilities.selectDataset(args, False, augment=False)

    # Get the embeddings and labels of the training set and testing set
    train_embeddings, train_labels = inferEmbeddings(args, train_dataset, "train")
    test_embeddings, test_labels = inferEmbeddings(args, test_dataset, "test")

    # Classify them
    accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels,
                           n_neighbors=args.neighbours)

    # Write it out to the console so that a calling process can pick it up
    sys.stdout.write(f"Accuracy={accuracy}\n")
    sys.stdout.flush()
    sys.exit(0)

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy

# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataset, split):
    # Wrap up the dataset in a PyTorch dataset loader
    data_loader = data.DataLoader(  dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False   )

    device = getDevice()

    # Define our embeddings model
    model = resnet50(   pretrained=True,
                        num_classes=dataset.getNumClasses(),
                        ckpt_path=args.model_path,
                        embedding_size=args.embedding_size   )

    # Put the model on the device and in evaluation mode
    model = model.to(device)
    model.eval()

    # Embeddings/labels to be stored for this split
    outputs_embedding = []
    labels_embedding = []

    # Iterate through the dataset and infer the embedding of every image
    with torch.no_grad():
        for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
            # Put the images on the device and scale into [0, 1], exactly as in training
            images = images.to(device).float() / 255.0

            # Get the embeddings of this batch of images
            outputs, _ = model(images)

            # Express embeddings/labels in numpy form
            outputs_embedding.append(outputs.cpu().numpy())
            labels_embedding.append(labels.long().view(len(labels)).cpu().numpy())

    outputs_embedding = np.concatenate(outputs_embedding, axis=0)
    labels_embedding = np.concatenate(labels_embedding, axis=0)

    # If we're supposed to be saving the embeddings and labels to file
    if args.save_embeddings:
        os.makedirs(args.save_path, exist_ok=True)

        # Construct the save path
        save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")

        # Save the embeddings to a numpy array
        np.savez(save_path, embeddings=outputs_embedding, labels=labels_embedding)

    return outputs_embedding, labels_embedding

# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for network testing')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model to load weights from')
    parser.add_argument('--folds_file', type=str, required=True,
                        help="The file containing known/unknown splits")
    parser.add_argument('--save_path', type=str, required=True,
                        help="Where to store the embeddings")

    parser.add_argument('--dataset', type=str, default='OpenSetPigs',
                        help='Which dataset to use')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory for the dataset')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128,
                        help='Dimensionality of the learned embedding')
    parser.add_argument('--neighbours', type=int, default=5,
                        help='Number of neighbours to use for KNN classification')

    parser.add_argument('--img_rows', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224,
                        help='Width of the input image')

    parser.add_argument('--current_fold', type=int, default=0,
                        help="The current fold we'd like to test on")
    parser.add_argument('--save_embeddings', action=argparse.BooleanOptionalAction, default=True,
                        help="Save the inferred embeddings to --save_path")
    args = parser.parse_args()

    # Let's infer some embeddings
    evaluateModel(args)
