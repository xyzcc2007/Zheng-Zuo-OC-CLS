# Zheng-Zuo-OC-CLS
# Deep Learning Framework for Ovarian Cancer Classification  
*Based on RegNetY032 Backbone*

---

## Introduction
This repository provides our deep learning code based on the **RegNetY032 backbone**, developed for distinguishing between benign and malignant ovarian cancer.

For the preprocessing program, we modified the code from [MMOTU_DS2Net](https://github.com/cv516Buaa/MMOTU_DS2Net) after obtaining authorization from the original authors.  
To respect copyright, we recommend that users review the original code and obtain authorization from the authors before using it.

---

## Requirements
The code provides many optional parameters. Please refer to **`get_args()`** for details.

---

## Input Data Format
The required **Excel file** must contain at least the following columns:

- **group**  
- **img_dir** (the directory path of the images to be fed into the model)  
- **class**

In addition:

- The corresponding **mask images** should be placed in the same folder.  
- Naming convention:  
  - ROI image → `aaa_roi.png`  
  - Corresponding mask → `aaa_mask.png`  
- Excel File paths are defined in **`defined_dataset.py`**.

---

## Usage Notes
Before running the program, please ensure that the **import paths** are consistent with the directory where your program is located.

---

## License & Contact
> **Note:** This code is provided **solely for academic exchange**.  
> For any other purposes, please contact: **yanlu76@zju.edu.cn** to obtain authorization.  
> For further communication, you may also reach out to: **xyzcc2007@126.com**.

---

## Citation
If you use this code for academic research, please cite appropriately and acknowledge the authors.

