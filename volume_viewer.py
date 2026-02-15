from mipcandy import fast_load, visualize3d
from mipcandy.presets.segmentation import print_stats_of_class_ids

case = "006"
image = fast_load(f"S:/SharedWeights/MIPCandy/valPreloaded/images/{case}.pt")
label = fast_load(f"S:/SharedWeights/MIPCandy/valPreloaded/labels/{case}.pt")
print(image.shape, label.shape)
print(label.min(), label.max())
print_stats_of_class_ids(label, "label", 5)
visualize3d(image, blocking=True)
visualize3d(label, is_label=True, blocking=True)
output = fast_load("S:/SharedWeights/MIPCandy/UNetTrainer/20260213-12-de22-UseThisToDebug/worst_output.pt")
output = output.softmax(0)
print(output.shape)
output = output.argmax(dim=0, keepdim=True)
print(output.shape)
print(output.min(), output.max())
print_stats_of_class_ids(output, "output", 5)
visualize3d(output, blocking=True)
