---
title: Reading the Grid
short_description: Solar cell fault inspection with Grad-CAM
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Reading the Grid

A convolutional neural network that detects faults in solar cells from electroluminescence
(EL) images, paired with a Grad-CAM heatmap showing where the model looked.

Upload an EL cell image or try a bundled example. The model returns a healthy or faulty
prediction, a movable decision threshold, and a drag-to-compare view of the Grad-CAM
overlay against the original.

## Notes and limitations

- The model reads electroluminescence scans, not ordinary daylight photos of a panel.
- Grad-CAM is normalized per image, so it shows attention rather than fault severity.
- Faulty localization is partial. This is an inspection-assist aid, not a calibrated
  localizer.

## Data

Trained on the ELPV dataset (Buerhop-Lutz et al.; Deitsch et al.), used for non-commercial
research purposes.

## Tech

A FastAPI backend serving a fine-tuned ResNet18, with a React and TypeScript frontend,
packaged as a single Docker container on port 7860.
