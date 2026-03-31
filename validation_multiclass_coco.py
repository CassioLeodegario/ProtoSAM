"""
Multi-class validation: ProtoSAM + MultiClassProtoSAM on COCO val2017.

Usage example:
    python validation_multiclass_coco.py \
        --category_ids 1 3 18 \
        --n_support 5 \
        --n_images 200
"""
import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ProtoSAM import ProtoSAM, MultiClassProtoSAM, ALPNetWrapper, InputFactory, TYPE_ALPNET
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.COCODataset import COCODataset

os.environ['TORCH_HOME'] = './pretrained_model'


def get_dice_iou(pred: torch.Tensor, gt: torch.Tensor):
    pred, gt = pred.float(), gt.float()
    if gt.sum() == 0:
        return None, None
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    iou  = (tp / (tp + fp + fn + 1e-8)).item()
    return dice, iou


def build_model(args, device):
    reload_path = None if args.reload_model_path == 'None' else args.reload_model_path
    model_config = {
        'align': True,
        'dinov2_loss': False,
        'use_coco_init': True,
        'which_model': args.modelname,
        'cls_name': 'grid_proto',
        'proto_grid_size': 8,
        'feature_hw': [args.input_size // 8, args.input_size // 8],
        'reload_model_path': reload_path,
        'lora': 0,
        'use_slice_adapter': False,
        'adapter_layers': 3,
        'debug': False,
        'use_pos_enc': False,
    }
    alpnet = FewShotSeg(args.input_size, reload_path, model_config).to(device)
    alpnet_wrapper = ALPNetWrapper(alpnet)

    protosam = ProtoSAM(
        image_size=(1024, 1024),
        coarse_segmentation_model=alpnet_wrapper,
        sam_pretrained_path=args.sam_checkpoint,
        use_bbox=True,
        use_points=True,
        use_mask=False,
        use_cca=True,
        num_points_for_sam=1,
        point_mode='both',
        use_sam_trans=True,
        coarse_pred_only=False,
        use_neg_points=False,
    ).to(device)
    protosam.eval()

    return MultiClassProtoSAM(protosam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',  default='data/COCO/val2017')
    parser.add_argument('--ann_file',   default='data/COCO/annotations/instances_val2017.json')
    parser.add_argument('--category_ids', nargs='+', type=int, default=[1, 3, 18],
                        help='COCO category IDs (1=person, 3=car, 18=dog)')
    parser.add_argument('--n_support',  type=int, default=5)
    parser.add_argument('--n_images',   type=int, default=-1,
                        help='Max query images to evaluate (-1 = all)')
    parser.add_argument('--reload_model_path', default='None')
    parser.add_argument('--sam_checkpoint',    default='pretrained_model/sam_vit_h.pth')
    parser.add_argument('--modelname',  default='dinov2_l14')
    parser.add_argument('--input_size', type=int, default=672)
    parser.add_argument('--val_wsize',  type=int, default=2)
    parser.add_argument('--gpu_id',     type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda')

    model = build_model(args, device)
    print(f"Model ready. Categories: {args.category_ids}")

    sam_trans = ResizeLongestSide(1024)

    # Support dataset — single-class mode to get binary masks per class
    support_ds = COCODataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        image_size=(1024, 1024),
        sam_trans=sam_trans,
    )

    # Query dataset — multi-class mode (full label map)
    query_ds = COCODataset(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        image_size=(1024, 1024),
        category_ids=args.category_ids,
        category_id=None,
        sam_trans=sam_trans,
    )

    # Sample support sets once for all classes
    cat_names = support_ds.get_category_names()
    print(f"\nSampling support sets (n_support={args.n_support})...")
    support_sets = {}
    for cat_id in args.category_ids:
        sup_imgs, sup_lbls, _ = support_ds.get_support(cat_id, n_support=args.n_support)
        support_sets[cat_id] = (sup_imgs, sup_lbls)
        print(f"  [{cat_id}] {cat_names.get(cat_id, '?'):15s} — {len(sup_imgs)} images")

    query_loader = DataLoader(
        query_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    n_images = args.n_images if args.n_images > 0 else len(query_loader)
    print(f"\nRunning inference on {n_images} query images...\n")

    per_class_dice = {cat_id: [] for cat_id in args.category_ids}
    per_class_iou  = {cat_id: [] for cat_id in args.category_ids}

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(query_loader, total=n_images)):
            if idx >= n_images:
                break

            query_image = sample['image'].to(device)   # (1, 3, 1024, 1024)
            gt_map = sample['label'][0].round().long() # (1024, 1024) — integer category IDs

            # Skip images with no target categories
            if not any((gt_map == c).any() for c in args.category_ids):
                continue

            # Build per-class ALPNet inputs
            class_inputs = {}
            for cat_id, (sup_imgs, sup_lbls) in support_sets.items():
                inp = InputFactory.create_input(
                    input_type=TYPE_ALPNET,
                    query_image=query_image,
                    support_images=sup_imgs,
                    support_labels=sup_lbls,
                    isval=True,
                    val_wsize=args.val_wsize,
                    original_sz=query_image.shape[-2:],
                    img_sz=query_image.shape[-2:],
                )
                inp.to(device)
                class_inputs[cat_id] = inp

            pred_map, _ = model(query_image, class_inputs)
            pred_map = pred_map.cpu()

            # Per-class metrics
            for cat_id in args.category_ids:
                gt_bin   = (gt_map   == cat_id).float()
                pred_bin = (pred_map == cat_id).float()
                dice, iou = get_dice_iou(pred_bin, gt_bin)
                if dice is not None:
                    per_class_dice[cat_id].append(dice)
                    per_class_iou[cat_id].append(iou)

    # Results
    print("\n========== Multi-Class Results ==========")
    all_dice, all_iou = [], []
    for cat_id in args.category_ids:
        name = cat_names.get(cat_id, str(cat_id))
        d = np.mean(per_class_dice[cat_id]) if per_class_dice[cat_id] else 0.0
        i = np.mean(per_class_iou[cat_id])  if per_class_iou[cat_id]  else 0.0
        n = len(per_class_dice[cat_id])
        print(f"  [{cat_id:2d}] {name:15s}  Dice={d:.4f}  IoU={i:.4f}  (n={n})")
        all_dice.append(d)
        all_iou.append(i)

    print(f"\n  Mean               Dice={np.mean(all_dice):.4f}  IoU={np.mean(all_iou):.4f}")


if __name__ == '__main__':
    main()
