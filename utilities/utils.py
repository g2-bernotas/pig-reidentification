# Core libraries
import os
import sys
import subprocess
import numpy as np

# PyTorch stuff
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data

# Import our own classes
from utilities.loss import *
from utilities.mining_utils import *
from models.TripletResnet import TripletResnet50
from models.TripletResnetSoftmax import TripletResnet50Softmax, ArcMarginProduct
from datasets.OpenSetPigs.OpenSetPigs import OpenSetPigs

"""
File contains a collection of utility functions used for training and evaluation
"""

# Repository root, used to locate test.py when it is called in a subprocess
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which layers get unfrozen when a given name is requested. Selecting a layer
# unfreezes it, everything after it, and the fully-connected heads.
def _layerMap(model):
    fc_layers = [model.fc, model.fc_embedding, model.fc_softmax]

    return {
        'layer1': [model.layer1, model.layer2, model.layer3, model.layer4] + fc_layers,
        'layer2': [model.layer2, model.layer3, model.layer4] + fc_layers,
        'layer3': [model.layer3, model.layer4] + fc_layers,
        'layer4': [model.layer4] + fc_layers,
        'fc': fc_layers,
        'fc_softmax': [model.fc_softmax, model.fc_embedding],
        'fc_embedding': [model.fc_embedding],
        'all': [model],
    }


class Utilities:
    # Class constructor
    def __init__(self, args):
        # Store the arguments
        self.args = args

        # Where to store training logs
        self.log_path = os.path.join(args.fold_out_path, "logs.npz")

        # Path to the most recently saved checkpoint
        self.checkpoint_path = None

        # Prepare arrays to store training information
        self.loss_steps = []
        self.losses_mean = []
        self.losses_softmax = []
        self.losses_triplet = []
        self.accuracy_steps = []
        self.accuracies = []

    # Preparations for training for a particular fold
    def setupForTraining(self, args):
        # Retrieve the correct dataset
        dataset = Utilities.selectDataset(args, True)

        # Wrap up the data in a PyTorch dataset loader
        data_loader = data.DataLoader(   dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True   )

        if args.finetune_model is not None:
            model = self.loadModelForFinetuning(args, dataset.getNumClasses())
        else:
            model = self.buildModelFromScratch(args, dataset.getNumClasses())

        # Freeze everything, then unfreeze the requested layers
        for param in model.parameters():
            param.requires_grad = False

        layer_map = _layerMap(model)
        if args.finetune_layers in layer_map:
            for layer in layer_map[args.finetune_layers]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print(f"Layer choice: \"{args.finetune_layers}\" not recognised, exiting.")
            sys.exit(1)

        # Put the model on the GPU and in training mode
        model = model.to(getDevice())
        model.train()

        # Setup the triplet selection method
        if args.triplet_selection == "HardestNegative":
            triplet_selector = HardestNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "RandomNegative":
            triplet_selector = RandomNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "SemihardNegative":
            triplet_selector = SemihardNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "AllTriplets":
            triplet_selector = AllTripletSelector()
        else:
            print(f"Triplet selection choice: \"{args.triplet_selection}\" not recognised, exiting.")
            sys.exit(1)

        # Setup the selected loss function
        if args.loss_function == "TripletLoss":
            loss_fn = TripletLoss(margin=args.triplet_margin)
        elif args.loss_function == "TripletSoftmaxLoss":
            loss_fn = TripletSoftmaxLoss(margin=args.triplet_margin, lambda_factor=args.lambda_factor)
        elif args.loss_function == "OnlineTripletLoss":
            loss_fn = OnlineTripletLoss(triplet_selector, margin=args.triplet_margin)
        elif args.loss_function == "OnlineTripletSoftmaxLoss":
            loss_fn = OnlineTripletSoftmaxLoss(triplet_selector, margin=args.triplet_margin,
                                               lambda_factor=args.lambda_factor)
        elif args.loss_function == "OnlineReciprocalTripletLoss":
            loss_fn = OnlineReciprocalTripletLoss(triplet_selector)
        elif args.loss_function == "OnlineReciprocalSoftmaxLoss":
            loss_fn = OnlineReciprocalSoftmaxLoss(triplet_selector, lambda_factor=args.lambda_factor)
        else:
            print(f"Loss function choice: \"{args.loss_function}\" not recognised, exiting.")
            sys.exit(1)

        # Create our optimiser. Reciprocal triplet losses are unbounded above, so
        # Adam is used there rather than SGD with momentum.
        if "Reciprocal" in args.loss_function:
            optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

        return data_loader, model, loss_fn, optimiser

    # Build an ImageNet-initialised model for training from scratch
    def buildModelFromScratch(self, args, num_classes):
        if args.model == "TripletResnetSoftmax":
            return TripletResnet50Softmax(  pretrained=True,
                                            num_classes=num_classes,
                                            embedding_size=args.embedding_size,
                                            use_arcface=args.use_arcface   )
        elif args.model == "TripletResnet":
            return TripletResnet50( pretrained=True,
                                    num_classes=num_classes,
                                    embedding_size=args.embedding_size   )

        print(f"Model choice: \"{args.model}\" not recognised, exiting.")
        sys.exit(1)

    # Load a previously trained model and re-shape its softmax head for the
    # (possibly different) number of classes in the current dataset
    def loadModelForFinetuning(self, args, num_classes):
        # Number of classes the checkpoint was trained with
        model_num_classes = args.model_num_classes

        if args.model == "TripletResnetSoftmax":
            model = TripletResnet50Softmax( pretrained=False,
                                            num_classes=model_num_classes,
                                            embedding_size=args.embedding_size,
                                            use_arcface=args.use_arcface   )
        elif args.model == "TripletResnet":
            model = TripletResnet50(pretrained=False,
                                    num_classes=model_num_classes,
                                    embedding_size=args.embedding_size   )
        else:
            print(f"Model choice: \"{args.model}\" not recognised, exiting.")
            sys.exit(1)

        # Load the weights. strict=False so that a mismatched head doesn't fail
        weights_init = torch.load(args.finetune_model, map_location="cpu", weights_only=False)['model_state']
        model.load_state_dict(weights_init, strict=False)

        if args.model != "TripletResnetSoftmax":
            return model

        # Re-create the softmax head for the current class count. Existing
        # classes keep their trained weights; new ones are randomly initialised.
        new_fc_weights = torch.zeros((num_classes, 1000))
        new_fc_bias = torch.zeros(num_classes)
        nn.init.xavier_uniform_(new_fc_weights)
        nn.init.zeros_(new_fc_bias)

        old_fc_weight = weights_init.get('fc_softmax.weight', None)
        old_fc_bias = weights_init.get('fc_softmax.bias', None)
        min_classes = min(model_num_classes, num_classes)

        if old_fc_weight is not None:
            new_fc_weights[:min_classes] = old_fc_weight[:min_classes]
        if old_fc_bias is not None:
            new_fc_bias[:min_classes] = old_fc_bias[:min_classes]

        # ArcFace keeps its own (bias-free, L2-normalised) weight matrix
        if args.use_arcface:
            head = ArcMarginProduct(1000, num_classes)
            with torch.no_grad():
                if old_fc_weight is not None:
                    head.weight[:min_classes] = old_fc_weight[:min_classes]
            model.fc_softmax = head
        else:
            model.fc_softmax = nn.Linear(1000, num_classes)
            model.fc_softmax.weight.data = new_fc_weights
            if model.fc_softmax.bias is not None:
                model.fc_softmax.bias.data = new_fc_bias

        return model

    # Unfreeze specific layers of the model (used by the unfreeze schedule)
    def unfreeze_layers(self, model, layer_name):
        layer_map = _layerMap(model)

        if layer_name in layer_map:
            print(f"[Info] Unfreezing layers associated with: {layer_name}")
            for layer in layer_map[layer_name]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print(f"[Warning] Layer name '{layer_name}' not found in layer map.")

    # Save a checkpoint as the current state of training
    def saveCheckpoint(self, epoch, model, optimiser, description):
        # Construct a state dictionary for the training's current state
        state = {   'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimiser.state_dict()  }

        # Construct the full path for where to save this
        self.checkpoint_path = os.path.join(self.args.fold_out_path, f"{description}_model_state.pkl")

        # And actually save it
        torch.save(state, self.checkpoint_path)

    # Save training logs to file
    def saveLogs(self):
        # Save this data to file for plotting graphs, etc.
        np.savez(   self.log_path,
                    loss_steps=self.loss_steps,
                    losses_mean=self.losses_mean,
                    losses_softmax=self.losses_softmax,
                    losses_triplet=self.losses_triplet,
                    accuracy_steps=self.accuracy_steps,
                    accuracies=self.accuracies    )

    # Log information
    def logTrainInfo(self, epoch, step, loss_mean, loss_triplet=None, loss_softmax=None):
        # Add to our arrays
        self.loss_steps.append(step)
        self.losses_mean.append(loss_mean)
        if loss_triplet is not None: self.losses_triplet.append(loss_triplet)
        if loss_softmax is not None: self.losses_softmax.append(loss_softmax)

        # Construct a message and print it to the console
        log_message = f"Epoch [{epoch+1}/{self.args.num_epochs}] Global step: {step} | loss_mean: {loss_mean:.5f}"
        if loss_triplet is not None: log_message += f", loss_triplet: {loss_triplet:.5f}"
        if loss_softmax is not None: log_message += f", loss_softmax: {loss_softmax:.5f}"
        print(log_message)

        # Save this new data to file
        self.saveLogs()

    # Evaluate the current model state by calling test.py in a subprocess and
    # parsing the accuracy it reports back
    def test(self, step):
        model_path = os.path.abspath(self.checkpoint_path)

        # Construct the subprocess call
        cmd = [
            sys.executable, os.path.join(REPO_ROOT, "test.py"),
            f"--model_path={model_path}",
            f"--dataset={self.args.dataset}",
            f"--dataset_root={self.args.dataset_root}",
            f"--batch_size={self.args.batch_size}",
            f"--num_workers={self.args.num_workers}",
            f"--embedding_size={self.args.embedding_size}",
            f"--current_fold={self.args.current_fold}",
            f"--folds_file={self.args.folds_file}",
            f"--save_path={self.args.fold_out_path}",
            f"--img_rows={self.args.img_rows}",
            f"--img_cols={self.args.img_cols}",
        ]

        # Run the command and stream the output so long evaluations show progress
        process = subprocess.Popen( cmd,
                                    cwd=REPO_ROOT,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True   )

        output = ""
        for line in process.stdout:
            print(line, end='')
            output += line

        process.wait()

        # Check if the command was successful
        if process.returncode != 0:
            raise RuntimeError(f"test.py failed with return code {process.returncode}")

        # Parse the accuracy value
        try:
            accuracy = float(output.split("Accuracy=")[1].split()[0])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse accuracy from the output: {output}") from e

        self.accuracy_steps.append(step)
        self.accuracies.append(accuracy)

        # Report the accuracy
        print(f"Accuracy: {accuracy}%")

        # Save the accuracies to file
        self.saveLogs()

        return accuracy

    """
    Static methods
    """

    # Return the selected dataset based on text choice
    @staticmethod
    def selectDataset(args, train, augment=None):
        # Which split are we after?
        split = "train" if train else "test"

        # Default: augment only when training
        if augment is None:
            augment = train

        # Load the selected dataset
        if args.dataset == "OpenSetPigs":
            # Muck augmentation is a training-time-only augmentation, and can be
            # switched off entirely to reproduce the no-dirt ("ND") ablation
            use_muck = augment and not getattr(args, 'disable_muck', False)

            dataset = OpenSetPigs(  args.current_fold,
                                    args.folds_file,
                                    root_dir=getattr(args, 'dataset_root', None),
                                    seg_dir=getattr(args, 'seg_dir', None),
                                    muck_dir=getattr(args, 'muck_dir', None),
                                    save_augmented_dir=getattr(args, 'save_augmented_dir', None),
                                    split=split,
                                    transform=True,
                                    transform_techniques=None,
                                    combine=False,
                                    suppress_info=False,
                                    augment=use_muck,
                                    img_size=(args.img_rows, args.img_cols)   )
        else:
            # To add your own dataset, write a Dataset class returning
            # (anchor, positive, negative, label_anchor, label_negative) tuples
            # (see datasets/OpenSetPigs/OpenSetPigs.py) and add a case here.
            print(f"Dataset choice: \"{args.dataset}\" not recognised, exiting.")
            sys.exit(1)

        return dataset


# Pick the best available device
def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
