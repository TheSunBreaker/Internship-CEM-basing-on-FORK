# Breast Cancer Radiomics Project

A comprehensive set of tools for neoadjuvant immunotherapy response prediction in triple negative breast cancer using radiomics extracted from MRI.

## 📋 Overview

This project provides tools for processing medical imaging data in breast cancer radiomics analysis pipelines, including:

1. **DICOM to NIfTI Conversion**: Essential preprocessing step for medical image analysis
2. **IRM to nnUNet Conversion**: Preparation of MRI data for deep learning segmentation with nnUNet

These tools are crucial steps in radiomics analysis pipelines for breast cancer research.

## 🏗️ Project Structure

```
BREAST-CANCER-RADIOMICS/
├── src/                          # Core functional modules
│   ├── dcm2nii.py              # DICOM to NIfTI conversion module
│   └── irm2nnunet.py           # IRM to nnUNet conversion module
├── dcm2nii_use_case.py          # DICOM to NIfTI batch processing example
├── irm2nnunet_use_case.py       # IRM to nnUNet conversion example
├── test_conversion.py           # Installation test script
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation configuration
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contributing guidelines
├── CHANGELOG.md                # Version history
├── .gitignore                  # Git ignore rules
└── GITHUB_SUBMISSION_GUIDE.md # GitHub submission instructions
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install SimpleITK
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## 📦 Dependencies

The main dependencies for this project are:

- **SimpleITK**: For medical image processing and DICOM handling
- **pathlib**: For cross-platform path handling (included in Python 3.4+)
- **argparse**: For command-line argument parsing (included in Python standard library)

## 🛠️ Usage

### 1. DICOM to NIfTI Conversion

#### Basic DICOM to NIfTI Conversion

```python
from src.dcm2nii import dicom2nifti

# Convert a single DICOM series
dicom2nifti("path/to/dicom/folder", "output/directory", prefix="subject_001")
```

#### Command Line Usage

```bash
# Convert a single DICOM series
python src/dcm2nii.py /path/to/dicom/folder /output/directory --prefix subject_001

# Batch process multiple subjects
python dcm2nii_use_case.py /path/to/input/root /path/to/output/root
```

#### Advanced Usage with Custom Modalities

```bash
# Process specific modalities only
python dcm2nii_use_case.py /input/root /output/root --modalities IRM TEP
```

### 2. IRM to nnUNet Conversion

#### Basic IRM to nnUNet Conversion

```python
from src.irm2nnunet import extract_irm_to_nnunet_flat

# Convert IRM files to nnUNet format
extract_irm_to_nnunet_flat(
    subjects_dir="/path/to/subjects",
    nnunet_root="/path/to/nnunet",
    dataset_id=1,
    irm_suffix="_IRM.nii.gz"
)
```

#### Command Line Usage

```bash
# Convert IRM files to nnUNet format
python irm2nnunet_use_case.py /path/to/subjects /path/to/nnunet --dataset-id 1
```

### 3. nnUNet Inference

#### Prerequisites Setup

Before running nnUNet inference, you need to set up the environment and install nnUNet:

```bash
# 0. If not created already, make an environment and activate it
python -m venv nnunet_env
source nnunet_env/bin/activate  # On Windows: nnunet_env\Scripts\activate

# 1. Install nnUNet
pip install nnunetv2

# 2. Set nnUNet folders as environment variables
export nnUNet_raw=~/nnUNet_raw
export nnUNet_preprocessed=~/nnUNet_preprocessed
export nnUNet_results=~/nnUNet_results
```

#### Dataset Preparation

```bash
# 3. Make sure the dataset.json file exists and is pasted in the nnUNet_raw folder
# The dataset.json should be in: ~/nnUNet_raw/Dataset001/dataset.json

# 4. Make sure to have the pretrained model in the nnUNet_results folder 
# with the dataset.json, dataset_fingerprint.json and plans.json files
# Structure should be: ~/nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

#### Running Inference

```bash
# 5. Run Infer command
nnUNetv2_predict \
    -d <DATASET_ID> \
    -i /path/to/imagesTs \
    -o /path/to/output \
    -c 3d_fullres \
    -f 0
```

**Parameters:**
- `-d <DATASET_ID>`: Dataset ID (e.g., 1 for Dataset001)
- `-i /path/to/imagesTs`: Path to the test images directory
- `-o /path/to/output`: Path to the output directory for predictions
- `-c 3d_fullres`: Configuration (3d_fullres for 3D full resolution)
- `-f 0`: Fold number (0 for single fold)

**Example:**
```bash
nnUNetv2_predict \
    -d 1 \
    -i ~/nnUNet_raw/Dataset001/imagesTs \
    -o ~/predictions \
    -c 3d_fullres \
    -f 0
```

## 📁 Input/Output Structures

### DICOM to NIfTI Input Structure

```
input_root/
├── subject_001/
│   ├── IRM/          # MRI DICOM files
│   ├── TEP/          # PET DICOM files
│   └── TDM/          # CT DICOM files
├── subject_002/
│   ├── IRM/
│   └── TEP/
└── ...
```

### DICOM to NIfTI Output Structure

```
output_root/
├── subject_001/
│   ├── subject_001_IRM.nii.gz
│   ├── subject_001_TEP.nii.gz
│   └── subject_001_TDM.nii.gz
├── subject_002/
│   ├── subject_002_IRM.nii.gz
│   └── subject_002_TEP.nii.gz
└── ...
```

### IRM to nnUNet Input Structure

```
subjects_dir/
├── subject_001/
│   └── subject_001_IRM.nii.gz
├── subject_002/
│   └── subject_002_IRM.nii.gz
└── ...
```

### IRM to nnUNet Output Structure

```
nnunet_root/
└── nnunetv2/
    ├── nnUNet_raw/
    │   └── Dataset001/
    │       ├── imagesTr/
    │       │   ├── subject_001_0000.nii.gz
    │       │   └── subject_002_0000.nii.gz
    │       ├── labelsTr/
    │       └── imagesTs_pred3dfullres/
    ├── nnUNet_preprocessed/
    └── nnUNet_results/
        └── Dataset001/
            └── nnUNetTrainer__nnUNetPlans__3d_fullres/
                └── fold_0/
```

## 🔧 API Reference

### `src.dcm2nii.dicom2nifti()`

Converts a DICOM series to NIfTI format.

**Parameters:**
- `dicom_folder` (str): Path to the folder containing DICOM files
- `output_dir` (str): Directory where the NIfTI file will be saved
- `prefix` (str, optional): Prefix for the output filename (default: "output")

**Returns:**
- `str`: Path to the saved NIfTI file

**Raises:**
- `FileNotFoundError`: If the DICOM folder doesn't exist
- `RuntimeError`: If no DICOM series is found in the folder

### `src.irm2nnunet.extract_irm_to_nnunet_flat()`

Converts IRM NIfTI files to nnUNet format with full directory structure.

**Parameters:**
- `subjects_dir` (str): Root directory with subject folders
- `nnunet_root` (str): Base directory where nnUNet structure will be created
- `dataset_id` (int): Dataset number (e.g., 1 → Dataset001)
- `irm_suffix` (str): Suffix for IRM files (default: "_IRM.nii.gz")

### `dcm2nii_use_case.process_subjects()`

Processes multiple subjects with batch DICOM to NIfTI conversion.

**Parameters:**
- `input_root` (str): Root input directory containing subject folders
- `output_root` (str): Directory where converted NIfTI files will be saved
- `modalities` (tuple): List of modality subfolders to check and convert

## 🧪 Testing

To test the conversion processes:

1. Create a sample directory structure with DICOM files
2. Run the conversion scripts
3. Verify the output files

```bash
# Test DICOM to NIfTI conversion
python dcm2nii_use_case.py ./test_data/input ./test_data/output

# Test IRM to nnUNet conversion
python irm2nnunet_use_case.py ./test_data/output ./test_data/nnunet --dataset-id 1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **esalasvilla** - *Initial work* - [GitHub Profile]

## 🙏 Acknowledgments

- SimpleITK development team for the excellent medical image processing library
- nnUNet development team for the segmentation framework
- The medical imaging community for open-source contributions

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@software{breast_cancer_radiomics_2025,
  author = {esalasvilla},
  title = {Breast Cancer Radiomics Project},
  year = {2025},
  url = {https://github.com/your-username/breast-cancer-radiomics}
}
```

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is designed for research purposes. Always ensure compliance with local regulations and institutional review board requirements when working with medical data. 