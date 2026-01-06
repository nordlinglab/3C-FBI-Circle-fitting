# CIBICA: Circle Detection Method

Implementation of CIBICA (Circle Inspection using Ballot Inspection with Combinations Analysis) for robust circle detection, compared against Hough Transform.

## Files
```
├── CIBICA.py           # CIBICA implementation
├── HOUGH.py            # Hough Transform wrapper
├── preprocessing.py    # Image preprocessing functions
├── main_CIBICA.py      # Main script to reproduce results
├── Ground_Truth.csv    # Ground truth data (144 images)
├── black_sphere_ROI/   # Input images (144 .png files)
└── green_back_ROI/     # Background reference images
```

## Installation
```bash
pip install numpy opencv-python scipy pandas matplotlib
```

## Quick Start

### 1. Detect circle with CIBICA
```python
from preprocessing import preprocess_image
from CIBICA import CIBICA

# Load and preprocess image
edgels = preprocess_image('your_image_name', method='green_level', param=76, hough=False)

# Detect circle
x, y, r = CIBICA(edgels, n_triplets=500, xmax=50, ymax=50)
print(f"Circle: center=({x:.1f}, {y:.1f}), radius={r:.1f}")
```

### 2. Detect circle with HOUGH
```python
from preprocessing import preprocess_image
from HOUGH import HOUGH

# Load and preprocess image
mask = preprocess_image('your_image_name', method='green_level', param=76, hough=True)

# Detect circle
x, y, r = HOUGH(mask, minDist=300, param2=8)
print(f"Circle: center=({x:.1f}, {y:.1f}), radius={r:.1f}")
```

### 3. Reproduce paper results
```bash
python main_CIBICA.py
```

**Output:**
- `Figure_Distance_Comparison.png` - Jaccard distance comparison
- `Figure_Distance_Ratio.png` - Performance ratio
- Console statistics

## Functions

### CIBICA.py
```python
CIBICA(coord, n_triplets=500, xmax=50, ymax=50, refinement=True)
```
- **Input:** Edge point coordinates (N x 2 array)
- **Output:** (x, y, r) - circle center and radius
- **Method:** Random triplet sampling + median estimation + least squares refinement

### HOUGH.py
```python
HOUGH(img, minDist=300, param1=50, param2=8, minRadius=5, maxRadius=25)
```
- **Input:** Grayscale/binary image
- **Output:** (x, y, r) - circle center and radius
- **Method:** OpenCV HoughCircles implementation

### preprocessing.py
```python
preprocess_image(filename, method='green_level', param=76, hough=False)
```
- **Input:** Image filename (without .png)
- **method:** `'green_level'` or `'median_filter'`
- **param:** Threshold value (70-86) or kernel size (3-19)
- **hough:** If True, returns mask for HOUGH; if False, returns edgels for CIBICA
- **Output:** Edge points or binary mask

## Preprocessing Methods

**Green Level Thresholding:**
```python
edgels = preprocess_image('image_name', method='green_level', param=76, hough=False)
```
- `param`: Green threshold in HSV space (70-86 recommended)

**Median Filtering:**
```python
edgels = preprocess_image('image_name', method='median_filter', param=5, hough=False)
```
- `param`: Kernel size (must be odd: 3, 5, 7, 9, ...)

## Results

Using 144 images with 18 preprocessing configurations:

- **Mean Jaccard Distance (HOUGH):** ~0.228
- **Mean Jaccard Distance (CIBICA):** ~0.151
- **Improvement:** ~34% (CIBICA better)

Lower Jaccard Distance = Better performance

## File Format Requirements

**Ground_Truth.csv:**
```
Filename,X,Y,R
image_name_1,21.87,20.64,11.61
image_name_2,17.65,20.57,11.73
...
```

**Image folders:**
- `black_sphere_ROI/image_name.png` - Input images
- `green_back_ROI/image_name.png` - Background reference (for median filter method)

## Parameters

### CIBICA
- `n_triplets`: Number of random samples (500-10000)
  - Higher = more accurate but slower
  - 500 is good balance for speed
- `refinement`: Apply least squares refinement (recommended: True)

### HOUGH
- `param2`: Accumulator threshold (5-30)
  - Lower = more detections (may include false positives)
  - 8 is tuned for this application
- `minDist`: Minimum distance between circles (pixels)

## Citations

**CIBICA Method:**
```bibtex
@article{your_paper,
  title={Circle Detection using CIBICA},
  author={Your Name},
  year={2024}
}
```

**Hough Transform:**
```bibtex
@article{duda1972hough,
  title={Use of the Hough transformation to detect lines and curves in pictures},
  author={Duda, Richard O and Hart, Peter E},
  journal={Communications of the ACM},
  volume={15},
  number={1},
  pages={11--15},
  year={1972}
}
```

**Edge Detection:**
```bibtex
@article{canny1986computational,
  title={A computational approach to edge detection},
  author={Canny, John},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  number={6},
  pages={679--698},
  year={1986}
}
```

## Example Workflow
```python
import pandas as pd
from preprocessing import preprocess_image
from CIBICA import CIBICA
from HOUGH import HOUGH

# Load ground truth
gt = pd.read_csv('Ground_Truth.csv')

# Process one image
filename = gt.iloc[0]['Filename']
true_x, true_y, true_r = gt.iloc[0]['X'], gt.iloc[0]['Y'], gt.iloc[0]['R']

# Test CIBICA
edgels = preprocess_image(filename, method='green_level', param=76, hough=False)
x_c, y_c, r_c = CIBICA(edgels, n_triplets=500, xmax=50, ymax=50)
print(f"CIBICA: ({x_c:.2f}, {y_c:.2f}), r={r_c:.2f}")

# Test HOUGH
mask = preprocess_image(filename, method='green_level', param=76, hough=True)
x_h, y_h, r_h = HOUGH(mask)
print(f"HOUGH:  ({x_h:.2f}, {y_h:.2f}), r={r_h:.2f}")

# Ground truth
print(f"Truth:  ({true_x:.2f}, {true_y:.2f}), r={true_r:.2f}")
```

## License

MIT

## Contact

[Your contact information]
