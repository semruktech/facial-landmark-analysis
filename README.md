# Facial Landmark Analysis
This repository contains a Python script for performing quick facial landmark analysis on facial images. It can be used for image analysis and facial landmark analysis in dental research.

### Getting Started
These instructions will help you set up the project on your local machine.

### Prerequisites
Download this repo as a zip file and extract all files to your desired location.

To run the project, the following software needs to be installed on your machine:
- Python>=3.11
- Required Python packages:
  - mediapipe
  - opencv-contib-python
  - numpy
  - pandas
  - openpyxl

If you already have these software and packages on your machine, proceed to the usage step, otherwise follow the installation step.

### Installation
Install a Python version greater than or equal to 3.11 on your machine.

To install the required Python packages, run the following command in terminal:

```bash
pip install -r requirements.txt
```

### Usage
The images you want to analyse facial landmark for must be in the images (Supported file types: jpg, png, tif, gif) folder.
From the terminal, go to the folder where you extracted the project files. Then the project can be run simply with the following command:

```bash
python main.py
```

When the interface is running: 
- Simply click two points for the calibration.
- Enter the actual distance between this two calibration points in milimeters.
- Results will be written in the "results" folder.
