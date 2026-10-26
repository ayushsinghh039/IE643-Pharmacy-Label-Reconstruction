# Inference and Demo

This folder contains the inference interface and supporting files for the IE 643 pharmacy-label reconstruction project.

## Files

- `ie643_demo.py` — Python demo application.
- `IE643_launch_interface.ipynb` — Notebook for launching or testing the interface.
- `dataset_split_v2.csv` — Dataset split information used by the project.

## Model Checkpoints

The trained model files are not included in this repository because of their large size:

- `unet_group_safe_best.keras`
- `best_ema.pth`

To run the application, obtain these checkpoints from the project owner and place them in the locations expected by the application.

## Running the Demo

1. Install the required Python dependencies.
2. Obtain the required model checkpoints.
3. Configure the model and input paths.
4. Follow the instructions in the demo script or interface notebook to launch the application.
