import numpy as np
from utils import model, save_detections, DO_NOT_MERGE
from PIL import Image


def xywh_to_tlbr(cx, cy, w, h):
    """
    Convert [cx, cy, w, h] to [[x1, y1], [x2, y2]]
    where (x1,y1) is top-left, (x2,y2) is bottom-right.
    """
    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)
    return [int(x1), int(y1)], [int(x2), int(y2)]


def tup(point):
    """Converts list [x, y] into (x, y)."""
    return (point[0], point[1])


def overlap(box_a, box_b):
    """
    Returns True if box_a and box_b overlap.
    Each box is [ [x1,y1], [x2,y2], class_id ].
    We only check geometry here, ignoring class_id.
    """
    (x1a, y1a), (x2a, y2a), _ = box_a
    (x1b, y1b), (x2b, y2b), _ = box_b

    # If one box is to the left of the other
    if x1a >= x2b or x1b >= x2a:
        return False
    # If one box is above the other
    if y1a >= y2b or y1b >= y2a:
        return False
    return True


def get_all_overlaps(boxes, bounds, index):
    """
    Return all indices of 'boxes' that overlap with 'bounds'
    (which is [ [x1,y1],[x2,y2], class_id ]).
    Skip 'index' itself.
    """
    results = []
    for i, box in enumerate(boxes):
        if i == index:
            continue
        if overlap(bounds, box):
            results.append(i)
    return results


def merge_boxes(boxes, merge_margin=15):
    """
    Repeatedly merges overlapping boxes until no more merges can be done.
    'boxes' is a list of [ [x1,y1],[x2,y2], class_id ].

    Returns the final, merged list of boxes.
    """
    finished = False
    while not finished:
        finished = True  # Assume no merges will happen this pass

        index = len(boxes) - 1
        while index >= 0:
            curr = boxes[index]

            # Expand current box by merge_margin
            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= merge_margin
            tl[1] -= merge_margin
            br[0] += merge_margin
            br[1] += merge_margin

            # We'll create a "temporary" box that uses curr's class,
            # but for checking overlap, the geometry is all that matters.
            # The class_id won't matter for geometry,
            # but we need it to keep the 3-elem structure.
            tmp_box = [tl, br, curr[2]]

            overlaps = get_all_overlaps(boxes, tmp_box, index)
            if overlaps:
                # Combine everything that overlaps into one bounding box
                overlaps.append(index)
                # Gather corners
                points = []
                classes_in_group = []
                for ind in overlaps:
                    box_tl, box_br, cls_id = boxes[ind]
                    points.append(box_tl)
                    points.append(box_br)
                    classes_in_group.append(cls_id)

                pts_array = np.array(points)
                x_vals = pts_array[:, 0]
                y_vals = pts_array[:, 1]

                merged_tl = [min(x_vals), min(y_vals)]
                merged_br = [max(x_vals), max(y_vals)]

                # For the merged class_id, you can either
                # pick the most frequent ID or just pick one.
                # We'll pick the first for simplicity:
                merged_class_id = classes_in_group[0]

                # Remove old boxes
                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]
                # Add merged box
                merged_box = [merged_tl, merged_br, merged_class_id]
                boxes.append(merged_box)

                # If we performed a merge, we repeat
                finished = False
                break

            index -= 1

    return boxes


def top_down_pipeline(input_img: str) -> list:
    # 1) Load the pre-trained model predictions
    det_res = model.predict(
        input_img,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cuda:0",
    )

    # 2) Read the original image
    original_img = Image.open(input_img).convert("RGB")

    # (Optional) save YOLO output images for debugging
    save_detections(det_res, input_img)

    # 3) Parse YOLO outputs to create bounding boxes in top-left/bottom-right format
    #    We'll store them in a list of [ [x1,y1],[x2,y2] ] for merging.
    raw_boxes = []
    for data in det_res:
        cls_list = data.boxes.cls.tolist()
        conf_list = data.boxes.conf.tolist()
        xywh_list = data.boxes.xywh.tolist()

        for cls_id, conf, (cx, cy, w, h) in zip(cls_list, conf_list, xywh_list):
            # Skip if class is "abandon" (id == 2)
            if cls_id == 2:
                continue

            tl, br = xywh_to_tlbr(cx, cy, w, h)
            raw_boxes.append([tl, br, int(cls_id)])

    # 4) Separate into mergeable vs. non_mergeable
    #    e.g. you want to keep class IDs 3,4,5 separate from text
    mergeable_boxes = [b for b in raw_boxes if b[2] not in DO_NOT_MERGE]
    non_mergeable_boxes = [b for b in raw_boxes if b[2] in DO_NOT_MERGE]

    # 5) Merge only the mergeable boxes
    merged_boxes = merge_boxes(mergeable_boxes, merge_margin=15)

    # 6) Combine them back
    final_boxes = merged_boxes + non_mergeable_boxes

    # 7) Sort final boxes top-down, then left-right
    #    final_boxes is [ [x1,y1],[x2,y2], class_id ]
    def sort_key(box):
        (x1, y1), (x2, y2), cls_id = box
        return (y1, x1)

    final_boxes.sort(key=sort_key)

    return [original_img, final_boxes]
