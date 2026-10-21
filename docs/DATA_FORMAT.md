# AVGSO Data Format

The core localization and navigation scripts read a flat directory of image frames. Each filename stores the frame order, latitude, longitude, and camera heading:

```text
frame_<frame_id>_<latitude>_<longitude>_<heading>.jpg
```

Example:

```text
frame_000001_36.045944_-79.960988_90.jpg
```

## Required fields

- `frame_id`: integer used to sort frames chronologically
- `latitude`: decimal geographic latitude
- `longitude`: decimal geographic longitude
- `heading`: camera/UAV heading in degrees; the scripts normalize circular angles during preprocessing

The current project also contains specialized line, circular, lawnmower, and mixed-route folders. The same metadata convention should be preserved when creating a new route split.

## Recommended split discipline

Use chronological or route-level separation rather than randomly mixing adjacent overlapping windows. Adjacent frames are visually and geographically correlated, so random window splitting can overestimate generalization.

The navigation scripts save a split gap and sample metadata in each run directory. Keep those files with published experiment results.

## Publishing data

The raw image collections in the working project are large. Before publishing them, verify the source imagery license, remove private or restricted assets, and provide a release archive or download instructions with checksums. The source-code repository can run dry-run dataset construction without committing the full image collection.
