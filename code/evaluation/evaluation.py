import fiftyone as fo
from PIL import Image
from code.evaluation.detection import detect

name = "dataset-small"
dataset_dir = "/home/zenon/Documents/master-thesis/evaluation/dataset-small"

# The splits to load
splits = ["val"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
    )

classes = dataset.default_classes
predictions_view = dataset.view()

with fo.ProgressBar() as pb:
    for sample in pb(predictions_view):
        image = Image.open(sample.filepath)
        w, h = image.size
        pred = detect(sample.filepath, '../weights/yolo.onnx', '../weights/resnet.onnx')

        detections = []
        for _, row in pred.iterrows():
            xmin, xmax = int(row['xmin']), int(row['xmax'])
            ymin, ymax = int(row['ymin']), int(row['ymax'])
            rel_box = [
                xmin / w, ymin / h, (xmax - xmin) / w, (ymax - ymin) / h
            ]
            detections.append(
                fo.Detection(label=classes[int(row['cls'])],
                             bounding_box=rel_box,
                             confidence=int(row['cls_conf'])))

        sample["yolo_resnet"] = fo.Detections(detections=detections)
        sample.save()

results = predictions_view.evaluate_detections(
    "yolo_resnet",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)

# Get the 10 most common classes in the dataset
counts = dataset.count_values("ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 classes
results.print_report(classes=classes_top10)

plot = results.plot_pr_curves(classes=["Healthy", "Stressed"])
plot.show()

session = fo.launch_app(dataset)
session.view = predictions_view
session.wait()
