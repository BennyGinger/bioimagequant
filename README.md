# bioimagequant

`bioimagequant` provides array-based quantitative analyses for biological
images. It does not load files or depend on FITS; calling applications own file
discovery, image metadata, scheduling, and result persistence.

The package currently exposes two analysis entry points:

```python
from bioimagequant import DistanceProfiler, ExtractData
```

- `ExtractData` measures labeled regions, their intensities, and optional
  distances from reference masks.
- `DistanceProfiler` measures intensity in distance bins around arbitrary
  reference masks, optionally restricted by ROI masks.

Both accept NumPy arrays with explicit axis strings and optional channel
identities. Acquisition interval and pixel size are optional metadata.
