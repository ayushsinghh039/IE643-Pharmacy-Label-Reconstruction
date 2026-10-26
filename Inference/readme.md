# Inference and Demo

This folder contains the interface code for trying the pharmacy-label reconstruction model. The trained model files are not stored in this repository because of their size.

## Files

- `ie643_demo.py` — Python demo application.
- `IE643_launch_interface.ipynb` — notebook for launching the interface in Google Colab.
- `dataset_split_v2.csv` — split manifest, if required by the selected demo/evaluation workflow.
- `models/unet_group_safe_best.keras` — U-Net checkpoint (download separately).
- `models/best_ema.pth` — diffusion checkpoint (download separately).

## Before you start

You will need:

1. Python and the packages imported by `ie643_demo.py`, or a Google Colab runtime for the notebook workflow.
2. The two trained checkpoints listed above.
3. Any input images or dataset files required by the workflow you want to run.
4. Access to the model files through an approved shared location, such as the project Google Drive folder.

The checkpoint files are not included in GitHub. Obtain them from the project owner and place them where the code expects them. Do not commit large model files, credentials, or access tokens.

## Option A: Run in Google Colab

1. Open `IE643_launch_interface.ipynb` in Google Colab.
2. Make sure the notebook can access the project folder in Google Drive.
3. Check the configuration/path cell and update it if your Drive folder differs.
4. Confirm that `ie643_demo.py` and both checkpoint files are present at the paths configured in the notebook.
5. Run the cells from top to bottom and follow the interface link printed by the notebook.

If the notebook refers to a particular Drive folder, reproduce that folder structure or edit the paths in the notebook before running it.

## Option B: Run the Python script locally

1. Install a compatible Python version and create a virtual environment.
2. Install the packages required by the script. Check the imports in `ie643_demo.py`; package versions may need to match the environment used to train the models.
3. Download/copy the two checkpoint files into the locations expected by the script.
4. From this folder, run:

   ```bash
   python ie643_demo.py --help
   ```

5. If the help output is supported by the script, use it to check the available arguments, then launch the application with the appropriate model paths.

Example command (adjust paths and options to match the script's actual arguments):

```bash
python ie643_demo.py --unet models/unet_group_safe_best.keras --recon models/best_ema.pth
```

The example is a template; confirm the supported arguments with the script before using it.

## Dataset split CSV

`dataset_split_v2.csv` is only needed if the script or notebook reads it for the chosen workflow. It is not necessarily required for a single-image inference run. If it is needed, place it at the path specified in the configuration.

## Troubleshooting

- **Checkpoint not found:** check the path and filename, including capitalization.
- **CUDA/GPU error:** try a compatible runtime or follow the device settings in the code.
- **Missing package:** install the package corresponding to the failed import.
- **Interface does not open:** read the final notebook/script output for the local URL or shared interface link.
- **Model loading error:** verify that the checkpoint matches the model architecture and expected checkpoint format.

## Reproducibility note

The README describes the expected setup. Exact dependency versions, required data files, and launch arguments should be confirmed against the current script and notebook before use.
