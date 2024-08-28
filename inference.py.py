from ultralytics import YOLO
import os

person_model_path = "C:/Users/Kavi priya/Desktop/runs/detect/person_detect_model_training/weights/best.pt"
ppe_model_path = "C:/Users/Kavi priya/Desktop/runs/detect/ppe_model_training/weights/best.pt"
test_images_path = "C:/Users/Kavi priya/Desktop/datasets/test/images"
person_output_dir = "C:/Users/Kavi priya/Desktop/runs/detect/person_detect_inference"
ppe_output_dir = "C:/Users/Kavi priya/Desktop/runs/detect/ppe_detect_inference"

os.makedirs(person_output_dir, exist_ok=True)
os.makedirs(ppe_output_dir, exist_ok=True)

def run_inference(model_path, source, output_dir):
    model = YOLO(model_path)
    results = model.predict(source=source, save=True, save_txt=True, save_conf=True, project=output_dir)
    return results
print("Running inference for Person Detection...")
person_results = run_inference(person_model_path, test_images_path, person_output_dir)
print("Person Detection Inference Results:", person_results)
print("Running inference for PPE Detection...")
ppe_results = run_inference(ppe_model_path, test_images_path, ppe_output_dir)
print("PPE Detection Inference Results:", ppe_results)

def run_validation(model_path):
    model = YOLO(model_path)
    metrics = model.val()
    return metrics
print("Validating Person Detection Model...")
person_validation_metrics = run_validation(person_model_path)
print("Person Detection Validation Metrics:", person_validation_metrics)
print("Validating PPE Detection Model...")
ppe_validation_metrics = run_validation(ppe_model_path)
print("PPE Detection Validation Metrics:", ppe_validation_metrics)
