
from ultralytics import YOLO

print("Loading ASTRIX model...")

model = YOLO("yolo11n.pt")

print("\nObjects recognised by the current model:\n")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

print("\nCheck completed!")
