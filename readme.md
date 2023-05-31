# Hand sign recognition

This is my solution for the Ordina Data Science Traineeship assessment.

## Required packages

Im using python version 3.8.16, note that the solution is only tested using MacOS with the apple M1 chip.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following packages with the following commands.

```bash
pip install numpy mediapipe open-cv scikit-learn tensorflow pandas matplotlib
```



## Usage

### JSON endpoint
```bash
python predict.py model.pkl example.json
```

### Training notebook
Run all blocks in the notebook, here you can see different visualizations and metrics of the different training methods. When running the last block a demo will start, press 'q' for the demo to end.