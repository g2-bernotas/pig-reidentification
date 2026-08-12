# Core libraries
import numpy as np

# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
File contains loss functions selectable during training.

Note on labels: the dataset produces 0-based class indices, and the softmax
targets are used as-is. (An earlier version of this code guessed whether labels
were 0- or 1-based from the contents of each minibatch, which silently shifted
every target by one whenever class 0 happened not to be sampled.)
"""


# Build the cross-entropy target for a triplet minibatch, in which the
# predictions are the concatenation of (anchor, positive, negative)
def _tripletTarget(labels, labels_neg, device):
	gpu_labels = labels.view(len(labels)).to(device)
	gpu_labels_neg = labels_neg.view(len(labels_neg)).to(device)

	return torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)


class TripletLoss(nn.Module):
	def __init__(self, margin=4.0):
		super(TripletLoss, self).__init__()
		self.margin = margin
					
	def forward(self, anchor, positive, negative, labels):
		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = F.relu(distance_positive - distance_negative + self.margin)
	
		return losses.sum()

class TripletSoftmaxLoss(nn.Module):
	def __init__(self, margin=0.0, lambda_factor=0.01):
		super(TripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor

	def forward(self, anchor, positive, negative, outputs, labels, labels_neg):
		distance_positive = torch.abs(anchor - positive).sum(1)
		distance_negative = torch.abs(anchor - negative).sum(1)
		losses = F.relu(distance_positive - distance_negative + self.margin)

		target = _tripletTarget(labels, labels_neg, anchor.device)
		loss_softmax = self.loss_fn(input=outputs, target=target)
		loss_total = self.lambda_factor*losses.sum() + loss_softmax

		return loss_total, losses.sum(), loss_softmax

class OnlineTripletLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0):
		super(OnlineTripletLoss, self).__init__()
		self.margin = margin
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Keep the indices on the same device as the embeddings
			triplets = triplets.to(embeddings.device)

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Compute the losses
		losses = F.relu(ap_distances - an_distances + self.margin)

		return losses.mean()

class OnlineTripletSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0, lambda_factor=0.01):
		super(OnlineTripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Concatenate labels for the softmax/crossentropy targets
		target = _tripletTarget(labels, labels_neg, anchor_embed.device)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Keep the indices on the same device as the embeddings
			triplets = triplets.to(embeddings.device)

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = F.relu(ap_distances - an_distances + self.margin)

		# Compute softmax loss over the 0-based class targets
		loss_softmax = self.loss_fn(input=preds, target=target)

		# Compute the total loss
		loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax

# Reciprocal triplet loss from 
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalTripletLoss(nn.Module):
	def __init__(self, triplet_selector):
		super(OnlineReciprocalTripletLoss, self).__init__()
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Keep the indices on the same device as the embeddings
			triplets = triplets.to(embeddings.device)

			# Compute distances over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Actually compute reciprocal triplet loss
		losses = ap_distances + (1/an_distances)

		return losses.mean()

# Reciprocal triplet loss from 
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Concatenate labels for the softmax/crossentropy targets
		target = _tripletTarget(labels, labels_neg, anchor_embed.device)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Keep the indices on the same device as the embeddings
			triplets = triplets.to(embeddings.device)

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = ap_distances + (1/an_distances)

		# Compute softmax loss over the 0-based class targets
		loss_softmax = self.loss_fn(input=preds, target=target)

		# Compute the total loss
		loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax
