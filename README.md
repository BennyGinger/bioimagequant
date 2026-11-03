# bioimagequant

`bioimagequant` provides array-based quantitative analyses for biological
images. It does not load files or depend on FITS; calling applications own file
discovery, image metadata, scheduling, and result persistence.

It is the numerical analysis layer used after image processing: labeled-object
extraction answers “what does each segmented object contain?”, while distance
profiling answers “how does a signal vary as it moves away from a reference?”.
Keeping those calculations in one package gives them shared array validation,
axis handling, physical scales, tabular output, and parallel execution without
coupling them to a particular pipeline or viewer.

The package currently exposes two analysis entry points:

```python
from bioimagequant import DistanceProfiler, ExtractData
```

- `ExtractData` measures labeled regions, their intensities, and optional
  distances from reference masks.
- `DistanceProfiler` measures intensity in distance bins around arbitrary
  reference masks, optionally restricted by ROI masks. Its `workers` argument
  can distribute independent reference frames across processes.

Both accept NumPy arrays with explicit axis strings and optional channel
identities. Acquisition interval and pixel size are optional metadata. Current
analyses are deliberately limited to 2D spatial data using YX axes; applications
must project Z before registering arrays.

## Role in FITS

FITS discovers matching image, label, reference, and ROI artifacts, projects Z
when needed, and registers their metadata with these classes. `bioimagequant`
then returns pandas tables; FITS adds experiment tags and writes the per-run and
aggregated Parquet files.

## Cell compartments

```python
from bioimagequant import CompartmentSettings, ExtractData, compartment_masks, prepare_compartments

extractor = ExtractData()
extractor.add_intensity(image, "TCYX", channel_labels=("GFP", "RFP"))
extractor.add_labels(labels, "TYX", name="tracking")
settings = CompartmentSettings(membrane_thickness=6, dilation_radius=0)
table = extractor.quantify(compartments=settings, workers=1)

# The same geometry API can serve previews or application-owned exports.
masks = compartment_masks(labels[0], settings)

# Reuse per-cell distances while tuning thickness; prepare again if dilation changes.
geometry = prepare_compartments(labels[0], dilation_radius=0)
narrow = geometry.masks(membrane_thickness=3)
wide = geometry.masks(membrane_thickness=6)
```

`quantify()` always returns a `compartment` column: `full` by default, or
`full`, `mb` and `cyto` when settings are supplied. Each object-label channel
gets its own compartments, retaining its object name, channel and numeric label.
Geometry is generated once per frame/object channel and reused across intensity
channels and properties; no files are created.

Distances are Euclidean pixels. Thickness must be finite and positive; expansion
must be finite and non-negative. Optional expansion preserves neighbouring labels
without overlap and defines the measured `full` region. Each cell's cytoplasm is
computed separately using a distance transform in a padded bounding box. Pixels
outside the image count as background. Cytoplasm pixels have distance strictly
greater than the membrane thickness; remaining cell pixels form the membrane.

If erosion empties the cytoplasm, one cell pixel nearest its full-cell centroid
is retained, choosing the first in array order on a tie. That pixel is excluded
from the membrane. A singleton cell consequently has an empty membrane: its row
is retained with zero area, perimeter and intensity measurements; undefined extra
shape properties remain missing.

Centroids and reference distances are measured from `full` after expansion and
shared across compartments. Other properties describe the respective compartment.
Areas and summed intensities partition `full`; mean intensities require area
weighting. Signed intensity values are preserved.

## Future 3D support

Adding volumetric analysis will require more than permitting a Z axis. The
package will need explicit Z/Y/X voxel spacing, volume and surface measurements,
3D-compatible default region properties, and physically scaled reference
distances. Until those behaviours are defined and tested, Z input is rejected
rather than producing measurements with ambiguous units.
