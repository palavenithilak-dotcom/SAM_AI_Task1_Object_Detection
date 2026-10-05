# SAM AI Technologies — Task 1
## Object Detection

A Streamlit object-detection application using the pre-trained YOLO11n model.

### Features
- Upload an image
- Detect common objects using a pre-trained model
- Draw bounding boxes and labels
- Display detected objects
- Show confidence scores
- Adjustable confidence threshold

### Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

On first run, Ultralytics downloads the pretrained `yolo11n.pt` model automatically if it is not already available.
