# Core libraries
import os
import random
import cv2
import numpy as np

"""
File contains input/output utility functions, plus the muck (dirt) augmentation
described in the paper.

The muck augmentation darkens parts of the animal using a pre-generated
band-limited noise field, imitating the dirt that accumulates on pigs over the
course of a day. The noise fields are read as `.npy` files from a directory
which can be given
  * per call, via the `muck_dir` argument,
  * on the command line, via `--muck_dir`, or
  * through the `PIG_MUCK_NOISE_DIR` environment variable.

Generate a set of noise fields with `python utilities/generate_muck_noise.py`.
If no directory is configured (or it is empty), the augmentation is silently
skipped, so training still runs - it just corresponds to the "(ND)" / no-dirt
ablation in the paper.
"""

# Environment variable fallback for the muck noise directory
MUCK_DIR_ENV_VAR = "PIG_MUCK_NOISE_DIR"

# Cache of {directory: [npy file paths]} so the listing happens once per process
_NOISE_FILE_CACHE = {}

# Directories we have already warned about, to avoid spamming the console
_WARNED_DIRS = set()


def allFilesAtDirWithExt(directory, *file_extensions, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    files = []
    # Gather files for each extension
    for extension in file_extensions:
        if full_path:
            ext_files = [os.path.join(directory, x) for x in sorted(os.listdir(directory))
                        if x.lower().endswith(extension.lower())]
        else:
            ext_files = [x for x in sorted(os.listdir(directory))
                        if x.lower().endswith(extension.lower())]
        files.extend(ext_files)

    return files

# Similarly, create a sorted list of all folders at a given directory
def allFoldersAtDir(directory, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Find all the folders
    if full_path:
        folders = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]
    else:
        folders = [x for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]

    return folders


def resolveMuckDir(muck_dir=None):
    """Return the muck noise directory to use, or None if none is configured."""
    return muck_dir or os.environ.get(MUCK_DIR_ENV_VAR) or None


def getNoiseFiles(muck_dir=None):
    """List (and cache) the `.npy` noise fields available in `muck_dir`."""
    directory = resolveMuckDir(muck_dir)
    if directory is None:
        if MUCK_DIR_ENV_VAR not in _WARNED_DIRS:
            _WARNED_DIRS.add(MUCK_DIR_ENV_VAR)
            print(f"[Info] No muck noise directory configured (--muck_dir / "
                  f"${MUCK_DIR_ENV_VAR}); muck augmentation is disabled.")
        return []

    if directory in _NOISE_FILE_CACHE:
        return _NOISE_FILE_CACHE[directory]

    if os.path.isdir(directory):
        files = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory) if f.endswith(".npy")
        )
    else:
        files = []

    if not files and directory not in _WARNED_DIRS:
        _WARNED_DIRS.add(directory)
        print(f"[Warning] No .npy noise fields found in {directory}. "
              f"Run 'python utilities/generate_muck_noise.py --out {directory}' "
              f"or drop --muck_dir to disable muck augmentation.")

    _NOISE_FILE_CACHE[directory] = files
    return files


def add_perlin_noise(img, noise_field, pig_mask=None):
    """Darken the masked region of `img` according to `noise_field`."""
    min_val = noise_field.min()
    max_val = noise_field.max()
    range_val = max_val - min_val

    if range_val == 0:  # Avoid division by zero; noise is constant
        noise_0_to_1 = np.full_like(noise_field, 0.5, dtype=np.float32)
    else:
        noise_0_to_1 = (noise_field - min_val) / range_val

    dirt_threshold_normalized = 100.0 / 255.0
    N_factor = np.ones_like(noise_0_to_1, dtype=np.float32)
    dirt_pixels_mask = noise_0_to_1 < dirt_threshold_normalized
    N_factor[dirt_pixels_mask] = noise_0_to_1[dirt_pixels_mask] / dirt_threshold_normalized
    noise_final_rgb = np.stack([N_factor]*3, axis=-1)

    if pig_mask is not None:
        # Ensure mask matches image dimensions
        if pig_mask.shape[:2] != img.shape[:2]:
            pig_mask = cv2.resize(pig_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        mask = pig_mask
    else:
        # Without a segmentation mask, fall back to "everything that isn't the
        # black letterbox padding"
        mask = img[:,:,0] > 0

    intensity = np.random.uniform(0.2, 1.0)

    merged_image_float = img.astype(np.float32)

    # Only apply if mask has True values
    if np.any(mask):
        dirt_amount = 1.0 - noise_final_rgb[mask]
        darkening_factor = 1.0 - intensity * dirt_amount
        blended_pixels = merged_image_float[mask] * darkening_factor
        merged_image_float[mask] = np.clip(blended_pixels, 0, 255)

    return merged_image_float.astype(np.uint8)


def transform_muck_image(image):
    """Applies random flipping, translation, zooming, and rotation to a noise field.

    Combines translation, zoom, and rotation into a single affine warp for speed.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Random parameters
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    tx = random.randint(-w // 10, w // 10)
    ty = random.randint(-h // 10, h // 10)

    # Build a single affine matrix: rotation + scale + translation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    image = cv2.warpAffine(image, M, (w, h))

    # Optional flip (cheap, no warp needed)
    if random.choice([True, False]):
        image = cv2.flip(image, flipCode=random.choice([-1, 0, 1]))

    return image


def add_muck_augmentation(img, pig_mask=None, muck_dir=None):
    """Apply random muck/dirt augmentation using a pre-generated noise field.

    Loads ONE random noise file on demand (not all of them) to avoid
    duplicating large arrays across DataLoader worker processes.
    """
    npy_file_paths = getNoiseFiles(muck_dir)
    if not npy_file_paths:
        return img

    # Load a single random noise field on demand
    noise_field = np.load(random.choice(npy_file_paths))

    # Augment the noise field itself so the same file yields different dirt
    noise_field = transform_muck_image(noise_field)

    # Resize noise to match the pig image dimensions
    noise_field = cv2.resize(noise_field, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply the noise to the image
    return add_perlin_noise(img, noise_field, pig_mask=pig_mask)


def loadSegmentationMask(img_path, seg_dir, size, new_size, paste_x, paste_y):
    """Load the mask mirroring `img_path` inside `seg_dir`, letterboxed like the image."""
    path_parts = os.path.normpath(img_path).split(os.sep)
    try:
        images_idx = path_parts.index('images')
    except ValueError:
        print(f"Warning: could not derive a relative path from {img_path}")
        return None

    rel_path = os.path.join(*path_parts[images_idx+1:])
    seg_img_path = os.path.join(seg_dir, 'images', rel_path)

    if not os.path.exists(seg_img_path):
        print(f"Warning: segmentation mask not found at {seg_img_path}")
        return None

    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
    if seg_img is None:
        return None

    # Resize the mask to match the padded image size
    seg_img_resized = cv2.resize(seg_img, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
    new_mask_np = np.zeros((size[1], size[0]), dtype=np.uint8)
    new_mask_np[paste_y:paste_y + new_size[0], paste_x:paste_x + new_size[1]] = seg_img_resized
    return new_mask_np > 0


# Load an image into memory, letterboxing it to `size` on a black background
def loadResizeImage(img_path, size, augment=True, seg_dir=None, muck_dir=None, save_augmented_dir=None):
    # Load the image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Keep the original image size
    old_size = img_np.shape[:2]

    # Compute resizing ratio
    ratio = float(size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Actually resize it
    img_np = cv2.resize(img_np, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)

    # Paste into the centre of a black canvas
    new_img_np = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    paste_x, paste_y = (size[0] - new_size[1]) // 2, (size[1] - new_size[0]) // 2
    new_img_np[paste_y:paste_y + new_size[0], paste_x:paste_x + new_size[1]] = img_np

    # Add muck/dirt augmentation (only during training)
    if augment:
        pig_mask = None
        if seg_dir is not None:
            pig_mask = loadSegmentationMask(img_path, seg_dir, size, new_size, paste_x, paste_y)

        img_np_aug = add_muck_augmentation(new_img_np.copy(), pig_mask=pig_mask, muck_dir=muck_dir)

        if save_augmented_dir is not None:
            os.makedirs(save_augmented_dir, exist_ok=True)
            # Save a random subset (1% chance) to avoid filling the disk
            if random.random() < 0.01:
                base_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(save_augmented_dir, f"orig_{base_name}"), cv2.cvtColor(new_img_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_augmented_dir, f"aug_{base_name}"), cv2.cvtColor(img_np_aug, cv2.COLOR_RGB2BGR))
                if pig_mask is not None:
                    cv2.imwrite(os.path.join(save_augmented_dir, f"mask_{base_name}"), (pig_mask * 255).astype(np.uint8))

        new_img_np = img_np_aug

    return new_img_np
