# CV-For-Human-Detection

## Overview
This repo is a testing ground for computer vision models geared towards detecting human faces in images.

## Models Used
* DeepFace
* Yolov8

## Current Testing Pipeline
1. Python code downloads an image from a given URL
2. Image is fed to the Yolov8 model, and this model detects persons in the image
3. The detected persons are then fed to the DeepFace model for further analysis, such as determining gender, age, and race
4. Results are printed

## Why are we using Yolov8 to detect humans when DeepFace can do that too?
- Yolov8 has more accurate results when it comes to determining if something is a person or not.

## Why are we not using Detectron2 for this?
- Theoretically, Detectron2 would also work in this pipeline as an alternative for Yolov8. However, I'm sticking to Yolov8 because it's lighter and faster.
