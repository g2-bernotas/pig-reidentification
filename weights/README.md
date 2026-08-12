# Released checkpoints

The trained weights are published separately from the code, at:

> <DOI: 10.5281/zenodo.20415925 / [URL](https://zenodo.org/uploads/21891666)>

Download the archive and unpack it into this directory.


| File | Crop modality | Muck aug. | Loss / head  |
|---|---|---|---|
| `instance_with_dirt.pkl` | Ins. Segm. | Yes | OnlineTripletSoftmax + ArcFace | 
| `instance_no_dirt.pkl` | Ins. Segm. | No  | OnlineTripletSoftmax + ArcFace | 
| `obb_with_dirt.pkl` | OBB        | Yes | … | 
| `obb_no_dirt.pkl` | OBB        | No | … | 
| `aabb_with_dirt.pkl` | AABB       | Yes | … | 
| `aabb_no_dirt.pkl` | AABB       | No | … | 

## Using a checkpoint

```bash
# extract embeddings for the paper's analyses
python analysis/extract_embeddings.py \
    --checkpoint weights/<modality>/best_model_state.pkl \
    --data-root  data/open_set_crops \
    --out        results/embeddings/<modality>_openset.npz

# fine-tune onto new animals
python train.py \
    --finetune_model weights/<modality>/best_model_state.pkl \
    --model_num_classes <N> --finetune_layers layer4 \
    --out_path output/finetuned \
    --dataset_root data/<your data> \
    --folds_file data/<your data>/splits/100-0.json
```

`--model_num_classes` must match the identity count the checkpoint was trained
with; the classification head is then resized to your dataset, keeping the
weights of any overlapping classes. The head is unused when only embeddings are
needed, and `analysis/extract_embeddings.py` reads the training-time class count
straight out of the checkpoint.