# Dataset Preparation

This repository uses the Oxford VGG Affine Covariant Features benchmark for detector evaluation.

Official source:

```text
https://www.robots.ox.ac.uk/~vgg/research/affine/
```

The selected benchmark sequences contain six images with increasing geometric or photometric change and ground-truth homographies from the reference image to the remaining images.

## 1. Evaluation Set

The repository uses the following eight sequences:

| Local directory | Oxford sequence | Image format |
|---|---|---|
| `seq1` | `bikes` | PPM |
| `seq2` | `trees` | PPM |
| `seq3` | `graf` | PPM |
| `seq4` | `wall` | PPM |
| `seq5` | `bark` | PPM |
| `seq6` | `boat` | PGM |
| `seq7` | `leuven` | PPM |
| `seq8` | `ubc` | PPM |

This gives 48 evaluation images in total.

The `boat` sequence is kept in PGM form. The repository image loaders accept both PPM and PGM input, so no format conversion is required.

## 2. Local Directory Layout

The dataset is not redistributed with this repository.

Download the required sequences from the original Oxford VGG source and organize them locally under:

```text
data/VGG/
```

using the mapping above.

Each local sequence directory contains six images and five homographies:

```text
seqN/
|-- img1.<ext>
|-- img2.<ext>
|-- img3.<ext>
|-- img4.<ext>
|-- img5.<ext>
|-- img6.<ext>
|-- H1to2p
|-- H1to3p
|-- H1to4p
|-- H1to5p
`-- H1to6p
```

For example:

```text
data/VGG/seq1/img1.ppm
data/VGG/seq1/H1to2p

data/VGG/seq6/img1.pgm
data/VGG/seq6/H1to2p
```

## 3. Ground-Truth Homographies

For each sequence, `img1` is the reference image.

The files:

```text
H1to2p
H1to3p
H1to4p
H1to5p
H1to6p
```

define the transformations used by the evaluation workflow when comparing `img1` with images 2 through 6.

## 4. Verification

A complete local dataset should contain:

```text
8 sequence directories
48 images
40 homography files
```

Each sequence contains six images and five homographies.

Dataset files remain local under `data/VGG/` and are excluded from version control.

## 5. Use in This Repository

The software reference, OpenCV baseline, and FPGA host use the same dataset root:

```text
./data/VGG
```

Build and execution commands are documented in [`BUILD.md`](BUILD.md).

## 6. Dataset Provenance

The repository changes only the local directory names used by the evaluation workflow. Image contents and ground-truth homographies are not modified.

For dataset attribution and provider-specific usage information, refer to the original Oxford VGG project page.
