# Human Activity Recognition (HAR)

A Streamlit demo that predicts six activities from smartphone sensor windows:
walking, walking upstairs, walking downstairs, sitting, standing, and laying.

## Run the application

From this folder, create and activate a virtual environment, install the
dependencies, and start Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses `har_model.pkl`, `features.py`, and the four `*.npy` sample files
in this folder. It offers both the UCI test samples and the included recordings.

## Optional sensor receiver

In another terminal, activate the environment and run:

```powershell
python receiver.py
```

The receiver listens on port 8000 and appends incoming messages to
`stream_log.jsonl`.

## Notebooks and datasets

The notebooks contain model training, preprocessing, and analysis workflows.
The UCI HAR dataset is excluded from Git because the running app does not need
it. To rerun notebooks that read UCI files, obtain the UCI HAR Dataset and
extract it beside the notebooks as `UCI HAR Dataset/`. `own_data/` contains the
recordings used by the preprocessing notebook.

## Main files

- `app.py`, `features.py`: Streamlit application and feature extraction
- `har_model.pkl`, `har_scaler.pkl`: saved trained model artifacts
- `demo_*.npy`, `own_*.npy`: sample windows and labels used by the app
- `receiver.py`: optional phone sensor data receiver
- `*.ipynb`: model development and analysis notebooks
