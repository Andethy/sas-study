# SAS Study


## Setup (PyCharm)

Make sure that the all `src` directories are marked as sources root. Either that or ensure the `PYTHONPATH` environment variable includes all `src` directories.

## Usage

### Run Configuration

Depending on what environment you are using, by the convention of how each individual study was set up, you will need to modify the run configuration(s).

Specifically, for each `main.py` script you are running, ensure the working directory is the root folder of that study, **not** the `src` folder (i.e., the parent of `src`).
