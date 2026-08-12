# Core libraries
import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

# PyTorch stuff
import torch
from torch.utils.tensorboard import SummaryWriter

# Local libraries
from utilities.loss import *
from utilities.mining_utils import *
from utilities.utils import Utilities, getDevice

"""
Trains the metric-learning network via cross-fold validation.

The network can either be trained from scratch (ImageNet-initialised) or
fine-tuned from an existing checkpoint (--finetune_model), optionally freezing
part of the backbone (--finetune_layers) and unfreezing it on a schedule
(--unfreeze_schedule).

Minimal example:

    python train.py \
        --out_path output/inssegm \
        --dataset_root data/OpenSetPigs \
        --folds_file data/OpenSetPigs/splits/50-50.json
"""

# Set every RNG we use, for reproducibility
def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Let's cross validate
def crossValidate(args):
    # Loop through each fold for cross validation
    for k in range(args.fold_number, args.num_folds):
        print(f"Beginning training for fold {k+1} of {args.num_folds}")

        # Directory for storing data to do with this fold
        args.fold_out_path = os.path.join(args.out_path, f"fold_{k}")

        # Create a folder in the results folder for this fold as well as to store embeddings
        os.makedirs(args.fold_out_path, exist_ok=True)

        # Store the current fold
        args.current_fold = k

        # Let's train!
        trainFold(args)

# Parse an unfreeze schedule of the form "5:layer3,15:all"
def parseUnfreezeSchedule(schedule_str):
    schedule = {}
    if not schedule_str:
        return schedule

    try:
        for item in schedule_str.split(','):
            epoch, layer = item.split(':')
            schedule[int(epoch)] = layer
    except ValueError:
        print("Error parsing unfreeze schedule. Format should be 'epoch:layer,epoch:layer'")
        sys.exit(1)

    return schedule

# Train for a single fold
def trainFold(args):
    # Create a new instance of the utilities class for this fold
    utils = Utilities(args)

    # Let's prepare the objects we need for training based on command line arguments
    data_loader, model, loss_fn, optimiser = utils.setupForTraining(args)

    device = getDevice()

    # Training tracking variables
    global_step = 0
    accuracy_best = 0

    # TensorBoard logs live alongside the rest of this run's output
    writer = SummaryWriter(os.path.join(args.fold_out_path, "tensorboard"))

    # Which layers should be unfrozen at which epoch
    unfreeze_schedule = parseUnfreezeSchedule(args.unfreeze_schedule)

    # Main training loop
    for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):

        # Check if we should unfreeze layers at this epoch
        if epoch in unfreeze_schedule:
            utils.unfreeze_layers(model, unfreeze_schedule[epoch])

        # Mini-batch training loop over the training set
        for images, images_pos, images_neg, labels, labels_neg in data_loader:
            # Put the images on the device and scale them into [0, 1].
            # NOTE: no mean/std normalisation is applied - inference code must match.
            images = images.to(device).float() / 255.0
            images_pos = images_pos.to(device).float() / 255.0
            images_neg = images_neg.to(device).float() / 255.0

            labels = labels.long().to(device)
            labels_neg = labels_neg.long().to(device)

            # Zero the optimiser
            optimiser.zero_grad()

            # Get the embeddings/predictions for each
            if "Softmax" in args.loss_function:
                if args.use_arcface:
                    embed_anch, embed_pos, embed_neg, preds = model(images, images_pos, images_neg, labels, labels_neg)
                else:
                    embed_anch, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
            else:
                embed_anch, embed_pos, embed_neg = model(images, images_pos, images_neg)

            # Calculate the loss on this minibatch
            if "Softmax" in args.loss_function:
                loss, triplet_loss, loss_softmax = loss_fn(embed_anch, embed_pos, embed_neg, preds, labels, labels_neg)
            else:
                loss = loss_fn(embed_anch, embed_pos, embed_neg, labels)

            # Backprop and optimise
            loss.backward()
            optimiser.step()
            global_step += 1

            writer.add_scalar("Loss/Total", loss.item(), global_step)

            # Log the loss if its time to do so
            if global_step % args.logs_freq == 0:
                if "Softmax" in args.loss_function:
                    writer.add_scalar("Loss/Triplet", triplet_loss.item(), global_step)
                    writer.add_scalar("Loss/Softmax", loss_softmax.item(), global_step)

                    utils.logTrainInfo( epoch, global_step, loss.item(),
                                        loss_triplet=triplet_loss.item(),
                                        loss_softmax=loss_softmax.item()    )
                else:
                    # Otherwise the loss isn't decomposed into components
                    utils.logTrainInfo(epoch, global_step, loss.item())

        # Every x epochs, let's evaluate on the validation set
        if epoch % args.eval_freq == 0:
            # Temporarily save model weights for the evaluation to use
            utils.saveCheckpoint(epoch, model, optimiser, "current")

            # Test on the validation set
            accuracy_curr = utils.test(global_step)

            writer.add_scalar("Accuracy/Validation", accuracy_curr, epoch)

            # Save the model weights as the best if it surpasses the previous best results
            if accuracy_curr > accuracy_best:
                utils.saveCheckpoint(epoch, model, optimiser, "best")
                accuracy_best = accuracy_curr

    writer.close()
    print(f"Finished fold {args.current_fold}, best validation accuracy: {accuracy_best:.2f}%")

# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for network training')

    # File configuration (the only required arguments)
    parser.add_argument('--out_path', type=str, required=True,
                        help="Path to folder to store results in")
    parser.add_argument('--folds_file', type=str, required=True,
                        help="Path to json file containing the known/unknown split per fold")

    # Core settings
    parser.add_argument('--num_folds', type=int, default=1,
                        help="Number of folds to cross validate across")
    parser.add_argument('--fold_number', type=int, default=0,
                        help="The fold number to START at")
    parser.add_argument('--dataset', type=str, default='OpenSetPigs',
                        help='Which dataset to use')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory for the dataset (containing images/train and images/test)')
    parser.add_argument('--seg_dir', type=str, default=None,
                        help='Directory of segmentation masks (mirroring --dataset_root) restricting '
                             'muck augmentation to the animal itself')
    parser.add_argument('--muck_dir', type=str, default=None,
                        help='Directory of .npy noise fields for muck augmentation '
                             '(see utilities/generate_muck_noise.py). Defaults to $PIG_MUCK_NOISE_DIR')
    parser.add_argument('--save_augmented_dir', type=str, default=None,
                        help='Directory to save a sample of augmented images into for inspection')
    parser.add_argument('--model', type=str, default='TripletResnetSoftmax',
                        help='Which model to use: [TripletResnetSoftmax, TripletResnet]')

    # Arguments for fine-tuning an existing model
    parser.add_argument('--finetune_model', type=str, default=None,
                        help="Path to a trained checkpoint to fine-tune from")
    parser.add_argument('--model_num_classes', type=int, default=100,
                        help="Number of classes the checkpoint given by --finetune_model was trained with")
    parser.add_argument('--finetune_layers', type=str, default='all',
                        help="Which layers to train [layer{1-4}, fc, fc_softmax, fc_embeddings, all]")

    parser.add_argument('--triplet_selection', type=str, default='HardestNegative',
                        help=("Which triplet selection method to use:"
                              "[HardestNegative, "
                              "RandomNegative, "
                              "SemihardNegative, "
                              "AllTriplets]"))
    parser.add_argument('--loss_function', type=str, default='OnlineReciprocalSoftmaxLoss',
                        help=("Which loss function to use: "
                              "[TripletLoss, TripletSoftmaxLoss, "
                              "OnlineTripletLoss, OnlineTripletSoftmaxLoss, "
                              "OnlineReciprocalTripletLoss, "
                              "OnlineReciprocalSoftmaxLoss]"))
    parser.add_argument('--lambda_factor', type=float, default=0.01,
                        help="Weighting of the triplet term relative to the softmax term")

    # Hyperparameters
    parser.add_argument('--img_rows', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224,
                        help='Width of the input image')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128,
                        help='Dimensionality of the learned embedding')
    parser.add_argument('--num_epochs', nargs='?', type=int, default=3,
                        help='# of the epochs to train for')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Optimiser learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument('--triplet_margin', type=float, default=0.5,
                        help="Margin parameter for triplet loss")

    # Training settings
    parser.add_argument('--eval_freq', nargs='?', type=int, default=1,
                        help='Frequency for evaluating model [epochs num]')
    parser.add_argument('--logs_freq', nargs='?', type=int, default=1,
                        help='Frequency for saving logs [steps num]')
    parser.add_argument('--unfreeze_schedule', type=str, default=None,
                        help="Schedule to unfreeze layers (e.g., '5:layer3,15:all')")
    parser.add_argument('--use_arcface', action='store_true', default=False,
                        help="Use an ArcFace margin head instead of a plain softmax head")
    parser.add_argument('--disable_muck', action='store_true', default=False,
                        help="Disable the muck/dirt augmentation (the '(ND)' ablation in the paper)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Seed everything we can
    setSeed(args.seed)

    # Let's cross validate!
    crossValidate(args)
