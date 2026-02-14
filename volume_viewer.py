from mipcandy import fast_load, visualize3d

case = "006"
image = fast_load(f"S:/SharedWeights/MIPCandy/valPreloaded/images/{case}.pt")
label = fast_load(f"S:/SharedWeights/MIPCandy/valPreloaded/labels/{case}.pt")
visualize3d(image, blocking=True)
visualize3d(label, is_label=True, blocking=True)
