from gsrd.risk.matching import GTObject, detection_correctness, iou_xywh


def test_iou_xywh() -> None:
    a = [0, 0, 10, 10]
    b = [5, 5, 10, 10]
    iou = iou_xywh(a, b)
    assert round(iou, 3) == 0.143


def test_detection_correctness_flat_vs_hierarchy() -> None:
    gt = [GTObject(bbox=[0, 0, 10, 10], class_name="car")]
    det_bbox = [0, 0, 10, 10]

    assert detection_correctness(det_bbox, {"car"}, gt, iou_thr=0.5, hierarchy_aware=False)
    assert not detection_correctness(det_bbox, {"car", "truck"}, gt, iou_thr=0.5, hierarchy_aware=False)
    assert detection_correctness(det_bbox, {"car", "truck"}, gt, iou_thr=0.5, hierarchy_aware=True)
