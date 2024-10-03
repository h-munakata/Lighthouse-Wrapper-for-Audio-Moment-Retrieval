# Feature Extractor
## What is this?
These scripts provide the procedure to extract features for lighthouse from audio files using [MS-CLAP](https://github.com/microsoft/CLAP).

If you want to extract features from your own audio files, please set the path to the audio files in `extract_feat.sh`.

## How to extract features?
1. Install the required packages:
    ```bash
    pip install -r ../requirements.txt
    ```
2. Set the config in `extract_feat.sh` and run the following command:
    ```bash
    bash extract_feat.sh
    ```
