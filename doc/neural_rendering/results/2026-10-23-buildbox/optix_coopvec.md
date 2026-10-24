# OptiX cooperative vectors: what the installed SDK documents

Source: installed OptiX SDK 9.1.0, `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\doc\`.
No `OptiX_Release_Notes.pdf` is installed with this SDK (only the Programming Guide, the API
Reference and the licence files), so the quotes come from those two documents. Page numbers are
the printed page / PDF page index.

## The answer for this box

The feature-support table (Programming Guide, ch. 12 "Limits", printed p. 134, PDF p. 140),
re-extracted with `pdftotext -table` because `-layout` interleaves the column headers:

| Hardware (RT core version) | non-RTX (0.0) | Turing (1.0) | Ampere (2.0) | Ada Lovelace (3.0) | Blackwell (4.0) |
|---|---|---|---|---|---|
| Cluster Acceleration Structures | no | yes | yes | yes | yes |
| Shader Execution Reordering | no-op | no-op | no-op | yes | yes |
| **Cooperative Vector** | no* | yes | yes | yes | yes |

> "Note: * Cooperative Vectors are enabled in select server class chips that do not have RT Cores."

So both cards are listed as supported: RTX 3090 = Ampere, RTX 4070 Ti SUPER = Ada Lovelace.
The installed documents **do not** say that cooperative vectors are hardware-accelerated on Ada
and Blackwell only, nor that Ampere takes a slower path. They give one yes/no per generation.
The claim in `00_synthesis.md` ("hardware acceleration is documented by NVIDIA for Ada and
Blackwell") is not supported by these documents; whether the 3090 is slower per matmul is a
measurement question, not a documentation one. Note the contrast in the same table: SER is
explicitly a "no-op" before Ada, while cooperative vectors are "yes" from Turing on.

## Quotes, every sentence that pairs "cooperative" with a GPU, type or Tensor Cores

Programming Guide 9.1.0, ch. 11 "Neural Rendering with Cooperative Vectors":

- printed p. 125 / PDF p. 131: "Cooperative vectors are an OptiX data type and API for
  high-performance machine learning computations during OptiX shader programs. They facilitate
  use of the Tensor Cores on NVIDIA RTX GPUs. Tensor Core operations require coordination between
  the threads in a warp."
- printed p. 125 / PDF p. 131: "Support for cooperative vectors extends to all RTX devices along
  with a selection of server class GPUs. Support can be queried using optixDeviceContextGetProperty
  with OPTIX_DEVICE_PROPERTY_COOP_VEC. Using cooperative vectors on devices that do not support it
  will generate errors; no fallback support is available."
- printed p. 127 / PDF p. 133: "To make use of the Tensor Cores when calling optixCoopVecMatMul,
  per-thread data is shuffled across the warp before the call and then unshuffled afterwards."
- printed p. 128 / PDF p. 134: "While full warps are not required to call optixCoopVecMatMul,
  performance will be best when using full warps." (Recommends SER to coalesce threads and making
  conditional calls unconditional.)
- printed p. 131 / PDF p. 137: "Only OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL matrices are
  currently supported." (for the training intrinsics)

API Reference 9.1.0 (data types):

- PDF p. 78: "Only OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 is supported." and "When matrixElementType is
  OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 or OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2 the matrixLayout must
  be either OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL or ..._TRAINING_OPTIMAL."
- PDF p. 80: "The element type must be OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16." (outer-product accumulate)
- PDF p. 81: "VecTIn::elementType must be of type OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 or
  OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32"
- PDF p. 111: "If the input elementType and output elementType are not equal, then one must be
  OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32 or OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 and the other must be a
  lower-precision floating-point type." (matrix conversion)
- PDF p. 161: `OptixCoopVecElemType` enumerates FLOAT16, FLOAT32, UINT8, INT8, UINT32, INT32,
  FLOAT8_E4M3, FLOAT8_E5M2.
- PDF p. 164: "OPTIX_DEVICE_PROPERTY_COOP_VEC Returns a flag specifying whether cooperative vector
  support is enabled for this device."
- PDF p. 165: "OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_NONE: Any use of cooperative vector host APIs or
  device intrinsics will result in an error." (other value: `..._FLAG_STANDARD`)

The documents do not state which element types are Tensor-Core accelerated per architecture
(e.g. FP8 on Ampere, which has no FP8 Tensor Cores). The authoritative per-card check is the
runtime query `OPTIX_DEVICE_PROPERTY_COOP_VEC` on each device; it has not been run (it needs a
small OptiX host program and is not part of the current Requested list).

Extraction: `pdftotext -layout` for the full text and `pdftotext -table -f 140 -l 140` for the
table; raw outputs stay on the workstation (`D:/Github/Blender-RND/optix_guide.txt`, `p140.txt`).
