import argparse
import os
import glob

def compute_metrics():
    if not os.path.exists(opt.val_path):
        print('../data/val does not exist. Before running, make sure to create the validation image folder and run detect.py on it')
        return 0

    if not os.path.exists(opt.output_path):
        print('output folder does not exists. Before running, make sure to create the validation image folder and run detect.py on it')
        return 0

    image_paths = glob.glob(f'{opt.val_path}/*') # Get paths of all images in val

    small_tp, small_fp, small_fn = 0,0,0
    large_tp, large_fp, large_fn = 0,0,0

    for img in image_paths:
        name = img.split('/')[-1]

        # Load in the ground truth bounding boxes for the current image
        label_path = f'{opt.labels_path}/{name}'.replace('.jpg', '.txt')
        if os.path.exists(label_path):
            label = open(label_path)
            gt_bboxes = [line[2:] for line in list(filter(None, label.read().split('\n')))]
            label.close()
        else:
            gt_bboxes = None

        # Load in the predicted bounding boxes for the current image
        pred_path = f'output/{name}'.replace('.jpg', '.txt')
        if os.path.exists(pred_path):
            pred = open(pred_path)
            pred_bboxes = [line[2:] for line in list(filter(None, pred.read().split('\n')))]
            pred.close()
        else:
            pred_bboxes = None

        # Skip if there are no ground truth or predictions
        if gt_bboxes == None and pred_bboxes == None:
            continue
        
        # If there are no ground truth bboxes in the image, then all predictions are FP
        if gt_bboxes == None:
            large, small = get_sizes(pred_bboxes)
            small_fp += small
            large_fp += large
            continue

        # If there are no predictions, then all ground truth boxes are FN
        if pred_bboxes == None:
            large, small = get_sizes(gt_bboxes)
            small_fn += small
            large_fn += large
            continue

        pred_matched = [0]*len(pred_bboxes)
        for i in range(len(gt_bboxes)):
            max_idx, max_iou = -1, 0
            for j in range(len(pred_bboxes)):
                if (pred_matched[j]):
                    continue
                iou = compute_iou(gt_bboxes[i], pred_bboxes[j])
                if (max_iou < iou and iou > opt.iou_thres):
                    max_idx, max_iou = j, iou
            if max_iou == 0:
                if (is_small(gt_bboxes[i])):
                    small_fn += 1
                else:
                    large_fn += 1
            else:
                pred_matched[max_idx] = 1
                if (is_small(gt_bboxes[i])):
                    small_tp += 1
                else:
                    large_tp += 1

        # Predictions that didn't correspond to a grouth truth label
        unmatched_pred = [pred_bboxes[i] for i in range(len(pred_bboxes)) if pred_matched[i] == 0]
        for bbox in unmatched_pred:
            if (is_small(bbox)):
                small_fp += 1
            else:
                large_fp += 1

    small_r = small_tp / (small_tp + small_fn)
    small_p = small_tp / (small_tp + small_fp)
    small_f1 = (2 * small_r * small_p) / (small_r + small_p)

    large_r = large_tp / (large_tp + large_fn)
    large_p = large_tp / (large_tp + large_fp)
    large_f1 = (2 * large_r * large_p) / (large_r + large_p)

    total_tp = small_tp + large_tp
    total_fn = small_fn + large_fn
    total_fp = small_fp + large_fp

    total_r = total_tp / (total_tp + total_fn)
    total_p = total_tp / (total_tp + total_fp)
    total_f1 = (2 * total_r * total_p) / (total_r + total_p)

    print(f'Total TP: {total_tp}, Total FN: {total_fn}, Total FP: {total_fp}')
    print(f'Total Precision: {total_p:.3}, Total Recall: {total_r:.3}, Total F1: {total_f1:.3}')

    print(f'Small TP: {small_tp}, Small FN: {small_fn}, Small FP: {small_fp}')
    print(f'Small Precision: {small_p:.3}, Small Recall: {small_r:.3}, Small F1: {small_f1:.3}')

    print(f'Large TP: {large_tp}, Large FN: {large_fn}, Large FP: {large_fp}')
    print(f'Large Precision: {large_p:.3}, Large Recall: {large_r:.3}, Large F1: {large_f1:.3}')

    if (opt.save_txt):
        with open('small_large_metrics.txt', 'w') as f:
            f.write(f'Total TP: {total_tp}, Total FN: {total_fn}, Total FP: {total_fp}\n')
            f.write(f'Total Precision: {total_p:.3}, Total Recall: {total_r:.3}, Total F1: {total_f1:.3}\n')
            f.write(f'Small TP: {small_tp}, Small FN: {small_fn}, Small FP: {small_fp}\n')
            f.write(f'Small Precision: {small_p:.3}, Small Recall: {small_r:.3}, Small F1: {small_f1:.3}\n')
            f.write(f'Large TP: {large_tp}, Large FN: {large_fn}, Large FP: {large_fp}\n')
            f.write(f'Large Precision: {large_p:.3}, Large Recall: {large_r:.3}, Large F1: {large_f1:.3}\n')


def compute_iou(bbox1, bbox2):
  # Parse the inputs (bbox1 and bbox2 are in string format with spaces between values)
    bbox1, bbox2 = [float(value) for value in list(filter(None, bbox1.split(' ')))], [float(value) for value in list(filter(None, bbox2.split(' ')))]

    # Convert x, y, height, width to left, right, top, bottom
    left1 = bbox1[0] - (bbox1[3]/2)
    right1 = bbox1[0] + (bbox1[3]/2)
    bottom1 = bbox1[1] - (bbox1[2]/2)
    top1 = bbox1[1] + (bbox1[2]/2)

    left2 = bbox2[0] - (bbox2[3]/2)
    right2 = bbox2[0] + (bbox2[3]/2)
    bottom2 = bbox2[1] - (bbox2[2]/2)
    top2 = bbox2[1] + (bbox2[2]/2)

    # If they don't overlap, return 0
    if min(right1, right2) -  max(left1, left2) < 0:
        return 0
    if min(top1, top2) - max(bottom1, bottom2) < 0:
        return 0

    # Compute IOU and return
    intersection = (min(right1, right2) -  max(left1, left2)) * (min(top1, top2) - max(bottom1, bottom2))
    area1 = (right1 - left1) * (top1 - bottom1)
    area2 = (right2 - left2) * (top2 - bottom2)
    union = area1 + area2 - intersection
    return intersection / union


def is_small(bbox):
    # Parse input since it is in string format with spaces between values
    bbox = [float(value) for value in list(filter(None, bbox.split(' ')))]

    # Compute area in pixels^2 and return 1 if smaller than threshold
    area = bbox[2] * bbox[3] * 608 * 608
    if area < opt.small_turbine_thres:
        return 1
    else:
        return 0


def get_sizes(bboxes):
    small = 0
    for bbox in bboxes:
        small += is_small(bbox)
    large = len(bboxes)-small
    return large, small


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='small_and_large_turbine_metrics.py')
    parser.add_argument('--iou-thres', type=float, default=0.0, help='IOU threshold for determining correct detection')
    parser.add_argument('--small-turbine-thres', type=float, default=350.0, help='Value in pixels squared that determines whether a turbine is small or not')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--val-path', type=str, default='../data/val')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('--labels-path', type=str, default='../data/labels')
    opt = parser.parse_args()
    print(opt)

    compute_metrics()
