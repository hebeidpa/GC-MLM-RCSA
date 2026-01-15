# GC-MLM-RCSA

## Multimodal Radiopathomics for Postoperative Metachronous Liver Metastasis Prediction in Gastric Cancer

This repository contains the reference implementation of **GC-MLM-RCSA**, a multimodal deep learning framework for predicting **postoperative metachronous liver metastasis (MLM)** in patients with gastric cancer by integrating **radiological computed tomography (CT)** and **histopathological whole-slide images (WSIs)**.

The framework is designed for research purposes and aims to explore the complementary value of radiological and pathological information for individualized postoperative risk assessment.

---

## Introduction

Despite curative gastrectomy, a substantial proportion of gastric cancer patients develop metachronous liver metastasis, which severely affects long-term survival. Accurate postoperative risk stratification remains challenging due to the heterogeneous biological behavior of tumors.

Radiological images provide macroscopic information about tumor morphology and surrounding tissues, while histopathological whole-slide images capture microscopic tumor characteristics. However, models based on a single modality are often insufficient to comprehensively characterize tumor heterogeneity.

To address this issue, we propose a **multimodal radiopathomics framework** that jointly models radiological and pathological data to predict postoperative MLM risk.
![8e4ab20cfe75b83dbae0b345308b41d8](https://github.com/user-attachments/assets/f78a5894-6d57-44c5-9db7-060ab5fbb8ef)


---

## Framework Overview

The proposed GC-MLM-RCSA framework consists of two main components:

1. **Radiology Branch**
   - Automatic segmentation of gastric regions on CT images
   - Extraction of radiological features from segmented regions

2. **Pathology Branch**
   - Patch-level feature extraction from WSIs
   - Slide-level representation learning using multiple instance learning

---

## Workflow

## Usage

This repository consists of two main functional modules:  
(1) a **radiology module** for CT-based feature extraction, and  
(2) a **pathology module** for WSI-based feature extraction.  

The two modules are executed independently and their outputs are subsequently fused at the patient level for multimodal modeling.

---

### 1. Radiology Module (CT-based)

The radiology module is implemented using the **nnU-Net framework (v2)**, which is a widely used self-configuring method for medical image segmentation (https://github.com/MIC-DKFZ/nnUNet).  
In this work, nnU-Net is adopted for automatic gastric region segmentation on CT images.

#### Step 1: CT Preprocessing
CT images should be converted to a unified format and resolution prior to analysis.

Typical preprocessing includes:
- Intensity normalization
- Resampling to a fixed voxel spacing

#### Step 2: Gastric Region Segmentation
CT images and corresponding gastric segmentation labels should be organized according to the nnU-Net v2 dataset format.
A typical dataset structure is:
```plaintext
nnUNet_raw/
└── DatasetXXX_GastricCT/
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz
    │   ├── case_002_0000.nii.gz
    ├── labelsTr/
    │   ├── case_001.nii.gz
    │   ├── case_002.nii.gz
    └── dataset.json
```
rain an nnU-Net model using the default 3D full-resolution configuration:
```plaintext
      nnUNetv2_train DatasetXXX 3d_fullres nnUNetTrainerV2 0
```
#### Step 3: Radiological Feature Extraction
Radiological features are extracted from the segmented gastric regions.

Specifically, the segmentation masks are used as regions of interest (ROIs), and radiomics features are computed using the **PyRadiomics** module integrated in **3D Slicer**. 

The output of this module is a patient-level radiological feature representation.

---

### 2. Pathology Module (WSI-based)

The pathology module is used to process histopathological whole-slide images based on the CLAM library (https://github.com/mahmoodlab/CLAM).
#### Step 1: WSI Tiling
Each whole-slide image is divided into fixed-size patches at a predefined magnification level.

Patches containing sufficient tissue content are retained for further analysis.
```plaintext
python create_patches_fp.py --source ./WSI --save_dir ./PATCH --patch_size 256 --preset bwh_resection.csv --seg --patch --stitch
```

#### Parameters
- `source`: digitized whole slide image data in well known standard formats (.svs, .ndpi, .tiff etc.) are stored under a folder named ./WSI
- `save_dir`: Generate the path(s) for the output file(s).

#### Step 2: Patch-level Feature Encoding
Patch-level features are extracted using UNI. (UNI: https://huggingface.co/MahmoodLab/UNI).
```plaintext
python extract_features_fp.py --data_h5_dir ./PATCH --data_slide_dir ./WSI --csv_path ./csv --feat_dir ./FEATURES --batch_size 512 --slide_ext .svs
```

The above command expects the coordinates .h5 files to be stored under ./PATCH and a batch size of 512 to extract 1024-dim features from each tissue patch for each slide and produce the following folder structure:
where each .h5 file contains an array of extracted features along with their patch coordinates (note for faster training, a .pt file for each slide is also created for each slide, containing just the patch features).

#### Step 3: Slide-level Feature Aggregation

Patch-level features are aggregated into slide-level representations using mean pooling, yielding 1024 features per slide.
The output of this module is a patient-level pathological feature representation.
```plaintext
python h5_to_csv.py
```


## Input and Output Summary

### Inputs
- CT images (radiology module)
- Whole-slide images (pathology module)

### Outputs
- Radiological feature vectors (per patient)
- Pathological feature vectors (per patient)


## Notes

- The two modules can be run independently
- Feature fusion and prediction require matched patient identifiers
- Pretrained models and example datasets are not publicly released due to data privacy constraints
## Acknowledgements

We would like to express our sincere gratitude to all contributors and collaborators who supported this project. 
Special thanks to our research team members for their invaluable discussions and technical insights, which greatly enhanced the development and implementation of this work.

We also acknowledge the open-source community and developers of several foundational frameworks and libraries that made this work possible. 
In particular, we thank the authors of **UNI**, a general-purpose foundation model for computational pathology, which was used for patch-level feature extraction in the pathology branch, and **nnU-Net**, a self-configuring deep learning framework adopted for automatic gastric region segmentation in the radiology branch.

Additionally, we acknowledge the developers of PyTorch, OpenSlide, scikit-learn, and related open-source tools for their indispensable contributions.
Finally, we are grateful to the institutions and organizations that provided the datasets and computational resources necessary for completing this project.
## Reference

We thank the authors and developers of the following projects for their outstanding contributions:

- **UNI**: Towards a General-Purpose Foundation Model for Computational Pathology  
  Chen, R. J., Ding, T., Lu, M. Y., et al. *Nature Medicine*, 30, 850–862 (2024).  
  https://doi.org/10.1038/s41591-024-02857-3
        
        
        
        
        
        
        
          

- **nnU-Net**: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation  
  Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H.  
  *Nature Methods*, 18(2), 203–211 (2021).  
  https://doi.org/10.1038/s41592-020-01008-z
        
        
        
        
        
        
        
          

