# Core libraries
import os
import sys
import cv2
import json
import random
import numpy as np

# PyTorch
import torch
from torch.utils import data

# Local libraries
from utilities.ioutils import allFilesAtDirWithExt, allFoldersAtDir, loadResizeImage

# Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Loads the open-set pig re-identification dataset into a PyTorch form.

Expected directory layout under `root_dir`:

	<root_dir>/
		images/
			train/
				G19_238/  <image>.png ...
				G19_241/
				...
			test/
				G19_238/  <image>.png ...
				G19_241/
				...

The `train` and `test` folders must contain the same set of identity folders.
Which identities count as "known" (seen during training) and "unknown" (held
back for open-set evaluation) is defined per fold in the splits JSON file:

	{ "0": { "known": ["G19_238", ...], "unknown": ["G20_239", ...] }, ... }

Use `utilities/generate_splits.py` to create such a file for your own data.
"""


class OpenSetPigs(data.Dataset):
	# Class constructor
	def __init__(	self,
					fold,
					fold_file,
					root_dir,
					seg_dir=None,
					muck_dir=None,
					save_augmented_dir=None,
					split="train",
					combine=False,
					known=True,
					transform=True,
					transform_techniques=None,
					img_size=(224, 224),
					suppress_info=True,
					augment=True	):
		"""
		Class attributes
		"""

		# The root directory for the dataset itself
		if root_dir is None:
			print("No dataset root directory given (--dataset_root), exiting.")
			sys.exit(1)
		self.__root = root_dir

		# Optional directory of binary segmentation masks mirroring `root_dir`,
		# used to restrict the muck/dirt augmentation to the animal itself
		self.__seg_dir = seg_dir

		# Optional directory of pre-generated noise fields for muck augmentation
		self.__muck_dir = muck_dir

		# Optional directory to dump a few augmented samples into for inspection
		self.__save_augmented_dir = save_augmented_dir

		# The fold we're currently considering
		self.__fold = str(fold)

		# The file containing the category splits for this fold
		self.__fold_file = fold_file

		# The split we're after (e.g. train/test)
		self.__split = split

		# Whether we should just load everything
		self.__combine = combine

		# Whether we're after known or unknown categories, irrelevant if combine is true
		self.__known = known

		# Whether to transform images/labels into pyTorch form
		self.__transform = transform

		# Whether to apply muck/noise augmentation when loading images
		self.__augment = augment

		# The augmentation/tensor-conversion pipeline. Photometric and geometric
		# augmentation is only ever applied to the training split - the test
		# split is converted to a tensor and otherwise left alone so that
		# inferred embeddings are deterministic.
		if transform_techniques is not None:
			self.__transform_techniques = transform_techniques
		elif self.__split == "train":
			self.__transform_techniques = self.buildTrainTransform()
		else:
			self.__transform_techniques = A.Compose([ToTensorV2()])

		# The directory containing actual imagery
		self.__train_images_dir = os.path.join(self.__root, "images", "train")
		self.__test_images_dir = os.path.join(self.__root, "images", "test")

		# Retrieve the number of classes from these
		self.__train_folders = allFoldersAtDir(self.__train_images_dir)
		self.__test_folders = allFoldersAtDir(self.__test_images_dir)
		assert len(self.__train_folders) == len(self.__test_folders)
		self.__num_classes = len(self.__train_folders)

		# Load the folds dictionary containing known and unknown categories for each fold
		if os.path.exists(self.__fold_file):
			with open(self.__fold_file, 'rb') as handle:
				self.__folds_dict = json.load(handle)
		else:
			print(f"File path doesn't exist: {self.__fold_file}")
			sys.exit(1)

		# A quick check
		assert self.__fold in self.__folds_dict.keys()

		# The image size to resize to
		self.__img_size = img_size

		# A dictionary storing seperately the list of image filepaths per category for
		# training and testing
		self.__sorted_files = {}

		# A dictionary storing separately the complete lists of filepaths for training and
		# testing
		self.__files = {}

		"""
		Class setup
		"""

		# Create dictionaries of categories: filepaths
		train_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg", ".png") for f in self.__train_folders}
		test_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg", ".png") for f in self.__test_folders}

		# Determine which categories to use
		if self.__combine:
			# Use all folders
			categories_to_use = self.__train_folders

			# If combining, map ALL categories
			unique_categories = sorted(set(os.path.basename(cat) for cat in self.__train_folders))
			self.__category_to_label = {
				os.path.basename(cat): idx
				for idx, cat in enumerate(unique_categories)
			}

			# Keep all files
			self.__sorted_files['train'] = train_files
			self.__sorted_files['test'] = test_files

		else:
			# Filter based on known/unknown
			if self.__known:
				target_list = self.__folds_dict[self.__fold]['known']
			else:
				target_list = self.__folds_dict[self.__fold]['unknown']

			categories_to_use = [cat for cat in self.__train_folders
								if os.path.basename(cat) in target_list]

			self.__category_to_label = {
				os.path.basename(cat): idx
				for idx, cat in enumerate(sorted(categories_to_use))
			}
			self.__num_classes = len(categories_to_use)

			# Filter files
			categories_to_keep = set(os.path.basename(cat) for cat in categories_to_use)
			self.__sorted_files['train'] = {k:v for (k,v) in train_files.items() if k in categories_to_keep}
			self.__sorted_files['test'] = {k:v for (k,v) in test_files.items() if k in categories_to_keep}

		# Consolidate this into one long list of filepaths for training and testing
		train_list = [v for k,v in self.__sorted_files['train'].items()]
		test_list = [v for k,v in self.__sorted_files['test'].items()]
		self.__files['train'] = [item for sublist in train_list for item in sublist]
		self.__files['test'] = [item for sublist in test_list for item in sublist]

		# Build reverse lookup: filepath -> category (O(1) instead of O(N) scan)
		self.__filepath_to_category = {}
		for split_key in ['train', 'test']:
			for cat, fps in self.__sorted_files[split_key].items():
				for fp in fps:
					self.__filepath_to_category[fp] = cat

		# Pre-compute filepath -> date dict (avoids repeated string parsing)
		self.__filepath_to_date = {}
		all_fps = self.__files['train'] + self.__files['test']
		for fp in all_fps:
			self.__filepath_to_date[fp] = self.getDateFromFilepath(fp)

		# Counter for saving augmented images for inspection
		self.__save_augmented_counter = 0

		# Report some things
		if not suppress_info: self.printStats()

	"""
	Superclass overriding methods
	"""

	# Get the number of items for this dataset (depending on the split)
	def __len__(self):
		return len(self.__files[self.__split])

	# Index retrieval method
	def __getitem__(self, index):
		max_retries = 100
		for _ in range(max_retries):
			# Get and load the anchor image
			img_path_anchor = self.__files[self.__split][index]

			# Retrieve the class/label this index refers to
			current_category = self.__retrieveCategoryForFilepath(img_path_anchor)

			# Get a positive (another random image from this class, different day)
			img_pos, img_path_positive = self.__retrievePositive(current_category, img_path_anchor)

			# If no suitable positive is found, retry with a different anchor
			if img_pos is None:
				index = random.randint(0, len(self) - 1)
				continue

			# Get a negative (a random image from a different random class)
			img_neg, label_neg, img_path_negative = self.__retrieveNegative(current_category, img_path_anchor)

			# Handle case where __retrieveNegative might fail
			if img_neg is None:
				index = random.randint(0, len(self) - 1)
				continue

			label_anchor = np.array([self.__category_to_label[current_category]])

			if label_neg is not None and label_neg in self.__category_to_label:
				label_neg = np.array([self.__category_to_label[label_neg]])
			else:
				index = random.randint(0, len(self) - 1)
				continue

			# Load the anchor image only after we confirm we have a valid triplet
			img_anchor = self.__loadImage(img_path_anchor)

			if self.__transform and self.__transform_techniques:
				img_anchor = self.__transform_techniques(image=img_anchor)['image']
				img_pos = self.__transform_techniques(image=img_pos)['image']
				img_neg = self.__transform_techniques(image=img_neg)['image']

			# Save first few augmented triplets for visual inspection
			if self.__save_augmented_dir is not None and self.__save_augmented_counter < 10:
				self.__saveAugmentedTriplet(img_anchor, img_pos, img_neg, label_anchor)

			return img_anchor, img_pos, img_neg, label_anchor, label_neg

		# Fallback: return the anchor as its own positive and negative if nothing
		# else could be found. Not useful for learning, but prevents a crash.
		print(f"Warning: failed to find a valid triplet after {max_retries} retries. Using fallback.")
		img_path_anchor = self.__files[self.__split][index]
		img_anchor = self.__loadImage(img_path_anchor)
		current_category = self.__retrieveCategoryForFilepath(img_path_anchor)
		label_anchor = np.array([self.__category_to_label[current_category]])

		if self.__transform and self.__transform_techniques:
			img_anchor_t = self.__transform_techniques(image=img_anchor)['image']
		else:
			# Need to transform to tensor manually if not using transform techniques
			img_anchor_t = torch.from_numpy(img_anchor.transpose(2, 0, 1))

		return img_anchor_t, img_anchor_t, img_anchor_t, label_anchor, label_anchor

	"""
	Public methods
	"""

	# The augmentation pipeline used on the training split.
	# NOTE: no mean/std normalisation is applied anywhere - the network is
	# trained on raw [0, 1] tensors (the division by 255 happens in train.py),
	# so any downstream inference code must do the same.
	@staticmethod
	def buildTrainTransform():
		return A.Compose([
			# Illumination robustness
			A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
			A.ColorJitter(brightness=(0.7, 1.3), contrast=(0.6, 1.4),
						saturation=(0.3, 1.2), hue=(-0.05, 0.05)),

			# Geometric
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.3),
			A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05),
					rotate=(-45, 45), p=0.8, keep_ratio=True),
			A.ElasticTransform(alpha=30, sigma=5, p=0.3,
					border_mode=cv2.BORDER_CONSTANT),

			# Noise / occlusion
			A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 2.0), p=0.4),
			A.CoarseDropout(num_holes_range=(2, 4), hole_height_range=(15, 25),
							hole_width_range=(15, 25), fill=0, p=0.4),

			ToTensorV2()
		])

	# Extract the date (YYYYMMDD) from a filepath
	@staticmethod
	def getDateFromFilepath(filepath):
		"""Extract the date (YYYYMMDD) from a filepath.

		Supports two naming conventions:
		  1) Basler_acA4112-20uc__40331001__20240626_101442824_0562.png
		     -> split by '__', take 3rd part, split by '_', first token is the date.
		  2) G20_270_20250422_Basler_acA4112-20uc__40331001__20250422_102113964_0086.png
		     -> split by '__', take 3rd part (same logic works), but also try
		       splitting the prefix before the first '__' by '_' and look for an
		       8-digit date token as a fallback.

		Returns None if no date could be parsed, in which case positive samples
		are drawn without the "different day" preference.
		"""
		try:
			filename = os.path.splitext(os.path.basename(filepath))[0]
			parts = filename.split('__')

			# Primary: 3rd component after splitting on '__'
			if len(parts) > 2:
				date_str = parts[2].split('_')[0]
				if len(date_str) == 8 and date_str.isdigit():
					return date_str

			# Fallback: scan tokens in the prefix before the first '__'
			if len(parts) >= 1:
				for token in parts[0].split('_'):
					if len(token) == 8 and token.isdigit():
						return token

			return None
		except Exception:
			return None

	# Print stats about the current state of this dataset
	def printStats(self):
		print("Loaded the OpenSetPigs dataset__________________________________")
		print(f"Fold = {int(self.__fold)+1}, split = {self.__split}, combine = {self.__combine}, known = {self.__known}")
		print(f"Found {self.__num_classes} categories: {len(self.__folds_dict[self.__fold]['known'])} known, {len(self.__folds_dict[self.__fold]['unknown'])} unknown")
		print(f"With {len(self.__files['train'])} train images, {len(self.__files['test'])} test images")
		print(f"Unknown categories {self.__folds_dict[self.__fold]['unknown']}")
		print("_______________________________________________________________")

	"""
	(Effectively) private methods
	"""

	# Load an image (with optional muck augmentation) at the configured size
	def __loadImage(self, img_path):
		return loadResizeImage(	img_path,
								self.__img_size,
								augment=self.__augment,
								seg_dir=self.__seg_dir,
								muck_dir=self.__muck_dir,
								save_augmented_dir=self.__save_augmented_dir	)

	def __saveAugmentedTriplet(self, img_anchor, img_pos, img_neg, label_anchor):
		"""Save augmented triplet images to disk for visual inspection."""
		os.makedirs(self.__save_augmented_dir, exist_ok=True)
		idx = self.__save_augmented_counter
		self.__save_augmented_counter += 1

		label_str = str(label_anchor[0]) if hasattr(label_anchor, '__getitem__') else str(label_anchor)

		def to_numpy(img):
			if isinstance(img, torch.Tensor):
				# Clone to avoid modifying the tensor used for training
				img_np = img.clone().cpu().permute(1, 2, 0).numpy()
				# ToTensorV2 does not rescale, so the tensor may be in [0, 255]
				if img_np.max() > 1.0:
					return np.clip(img_np, 0, 255).astype(np.uint8)
				return (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
			return img

		for name, img in (("anchor", img_anchor), ("positive", img_pos), ("negative", img_neg)):
			out_path = os.path.join(self.__save_augmented_dir, f"triplet_{idx:03d}_label{label_str}_{name}.png")
			cv2.imwrite(out_path, cv2.cvtColor(to_numpy(img), cv2.COLOR_RGB2BGR))

	# Print some info about the distribution of images per category
	def __printImageDistribution(self):
		for category, filepaths in self.__sorted_files[self.__split].items():
			print(category, len(filepaths))

	# For a given filepath, return the category which contains this filepath
	def __retrieveCategoryForFilepath(self, filepath):
		return self.__filepath_to_category.get(filepath)

	# Get another positive sample from this class (preferring a different day)
	def __retrievePositive(self, category, anchor_filepath):
		possible_positives_all = list(self.__sorted_files[self.__split][category])

		# Filter out the anchor itself
		valid_positives = [p for p in possible_positives_all if p != anchor_filepath]

		if not valid_positives:
			# If the class has only one image (the anchor), we can't find a positive
			return None, None

		# Try to pick a positive from a different day than the anchor. Pigs change
		# appearance over the course of a day (dirt, posture, lighting), so
		# cross-day positives are what force the embedding to be time-invariant.
		anchor_date = self.__filepath_to_date.get(anchor_filepath)

		if anchor_date is not None:
			different_day_positives = [
				p for p in valid_positives
				if self.__filepath_to_date.get(p) != anchor_date
			]
			if different_day_positives:
				img_path_positive = random.choice(different_day_positives)
			else:
				# All positives are from the same day - fall back to any valid positive
				img_path_positive = random.choice(valid_positives)
		else:
			# Could not parse anchor date - fall back to any valid positive
			img_path_positive = random.choice(valid_positives)

		# Load and return the image
		return self.__loadImage(img_path_positive), img_path_positive

	# Get a negative sample from a different, randomly chosen class
	def __retrieveNegative(self, category, filepath):
		# Get the list of categories and remove that of the anchor
		possible_categories = list(self.__sorted_files[self.__split].keys())
		if category in possible_categories:
			possible_categories.remove(category)

		if not possible_categories:
			# Only possible with a single-identity dataset
			return None, None, None

		# Randomly select a category
		random_category = random.choice(possible_categories)

		# Randomly select a filepath in that category
		img_path_negative = random.choice(self.__sorted_files[self.__split][random_category])

		# Load and return the image along with the selected label
		return self.__loadImage(img_path_negative), random_category, img_path_negative

	"""
	Getters
	"""

	def getNumClasses(self):
		return self.__num_classes

	def getNumTrainingFiles(self):
		return len(self.__files["train"])

	def getNumTestingFiles(self):
		return len(self.__files["test"])
