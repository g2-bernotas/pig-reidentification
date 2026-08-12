# Core libraries
import os
import sys
import argparse
import numpy as np

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# Define our own plot function
def scatter(x, labels, filename):
	# Get the number of classes (number of unique labels)
	num_classes = np.unique(labels).shape[0]

	# Choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", num_classes+1))

	# Map the colours to different labels
	label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])

	# Create our figure/plot
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')

	# Plot the points
	ax.scatter(	x[:,0], x[:,1], 
				lw=0, s=40, 
				c=label_colours, 
				marker="o")

	# Do some formatting. The axes are autoscaled rather than clamped to a fixed
	# range - t-SNE output extents depend on the perplexity and the number of
	# points, so hardcoded limits can push most of the cloud out of view.
	ax.margins(0.05)
	ax.axis('off')
	plt.tight_layout()

	# Save it to file
	plt.savefig(filename)
	plt.close(f)
	print(f"Saved t-SNE plot -> {filename}")

# Load and visualise embeddings via t-SNE
def plotEmbeddings(args):
	# Ensure there's something there
	if not os.path.exists(args.embeddings_file):
		print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
		sys.exit(1)

	# Load the embeddings into memory
	embeddings = np.load(args.embeddings_file)

	print("Loaded embeddings")

	# Visualise the learned embedding via t-SNE
	visualiser = TSNE(n_components=2, perplexity=args.perplexity)

	# Reduce dimensionality
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file, next to the embeddings unless told otherwise
	out_path = args.out or os.path.splitext(args.embeddings_file)[0] + "_tsne.pdf"
	out_dir = os.path.dirname(os.path.abspath(out_path))
	os.makedirs(out_dir, exist_ok=True)
	scatter(reduction, embeddings['labels'], out_path)

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
	parser.add_argument('--embeddings_file', type=str, required=True,
						help="Path to embeddings .npz file you want to visalise")
	parser.add_argument('--perplexity', type=int, default=30,
						help="Perplexity parameter for t-SNE, consider values between 5 and 50")
	parser.add_argument('--out', type=str, default=None,
						help="Where to write the plot (default: alongside the embeddings file)")
	args = parser.parse_args()

	# Let's plot!
	plotEmbeddings(args)
	