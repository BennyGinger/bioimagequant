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

## Future 3D support

Adding volumetric analysis will require more than permitting a Z axis. The
package will need explicit Z/Y/X voxel spacing, volume and surface measurements,
3D-compatible default region properties, and physically scaled reference
distances. Until those behaviours are defined and tested, Z input is rejected
rather than producing measurements with ambiguous units.
