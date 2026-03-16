# Dataset

The training dataset is **not included in this repository** (195 GB, 7313 files).

## Expected Layout

Place the dataset files at `dataset/` (this directory) so the paths match
the config in `training/config/exp1_dsai.yaml`:

```
dataset/
├── 1979_0000.h5          # ERA5 6-hourly atmospheric fields, HDF5
├── 1979_0001.h5
├── ...
├── 1980_NNNN.h5
├── pangu_s2s_1979-2018_mean.nc             # Upper-air normalisation mean
├── pangu_s2s_1979-2018_std.nc              # Upper-air normalisation std
├── pangu_s2s_1979-2018_surface_mean.nc     # Surface normalisation mean
├── pangu_s2s_1979-2018_surface_std.nc      # Surface normalisation std
├── pangu_s2s_1979-2018_delta_mean.nc
├── pangu_s2s_1979-2018_delta_std.nc
├── pangu_s2s_1979-2018_surface_delta_mean.nc
├── pangu_s2s_1979-2018_surface_delta_std.nc
└── 1979-2018_mean_climatology.nc           # Monthly climatology for CRPS scoring
```

## Variable Groups (defined in exp1_dsai.yaml)

| Group | Variables |
|---|---|
| Upper air (17 levels) | temperature, u/v wind, specific humidity, geopotential |
| Surface | 2m temperature, 10m u/v wind, MSLP, surface pressure |
| Diagnostic | total precipitation (24hr), top LW radiation |
| Land | soil water, soil temperature, skin temperature |
| Ocean | sea surface temperature |
| Constant boundary | land-sea mask, surface geopotential |
| Varying boundary | TOA incident solar radiation |

## Pressure Levels

```
5, 10, 20, 30, 50, 70, 100, 150, 250, 300,
400, 500, 600, 700, 850, 925, 1000 hPa  (17 levels)
```

## Source / Contact

This dataset is a pre-processed ERA5 reanalysis extract covering 1979–2018
in the Pangu S2S format (6-hourly, 1°×1° global grid, 180×360).

Contact the dataset maintainer for access or refer to the Pangu-Weather S2S
data preparation pipeline for instructions on generating it from raw ERA5.
