# Breast Cancer Radiomics Project

A comprehensive set of tools for neoadjuvant immunotherapy response prediction in triple negative breast cancer using radiomics extracted from MRI.

## 📋 Overview

This project provides tools for processing medical imaging data, specifically DICOM to NIfTI conversion, which is a crucial step in radiomics analysis pipelines for breast cancer research.

## 🏗️ Project Structure

```
final_codes/
├── src/                          # Core functional modules
│   └── dcm2nii.py              # DICOM to NIfTI conversion module
├── dcm2nii_use_case.py          # Example usage and batch processing
├── BREAST-CANCER-RADIOMICS/     # Research-specific documentation
│   └── README.md                # Detailed research methodology
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

### Basic DICOM to NIfTI Conversion

```python
from src.dcm2nii import dicom2nifti

# Convert a single DICOM series
dicom2nifti("path/to/dicom/folder", "output/directory", prefix="subject_001")
```

### Command Line Usage

```bash
# Convert a single DICOM series
python src/dcm2nii.py /path/to/dicom/folder /output/directory --prefix subject_001

# Batch process multiple subjects
python dcm2nii_use_case.py /path/to/input/root /path/to/output/root
```

### Advanced Usage with Custom Modalities

```bash
# Process specific modalities only
python dcm2nii_use_case.py /input/root /output/root --modalities IRM TEP
```

## 📁 Input Structure

The batch processing script expects the following directory structure:

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

## 📤 Output Structure

The conversion process creates the following output structure:

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

### `dcm2nii_use_case.process_subjects()`

Processes multiple subjects with batch conversion.

**Parameters:**
- `input_root` (str): Root input directory containing subject folders
- `output_root` (str): Directory where converted NIfTI files will be saved
- `modalities` (tuple): List of modality subfolders to check and convert

## 🧪 Testing

To test the conversion process:

1. Create a sample directory structure with DICOM files
2. Run the conversion script
3. Verify the output NIfTI files

```bash
# Example test run
python dcm2nii_use_case.py ./test_data/input ./test_data/output
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