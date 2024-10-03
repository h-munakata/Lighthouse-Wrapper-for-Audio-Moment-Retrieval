# Clotho-Moment Generetor
## What is this?
These scripts provide the procedure to generate Clotho-Moments from Clotho and Walking Tours.
- Clotho: [https://zenodo.org/records/4783391](https://zenodo.org/records/4783391)
- Walking Tour: [https://shashankvkt.github.io/dora](https://shashankvkt.github.io/dora)

## How to generate Clotho-Moments?
1. Install the required packages:
    ```bash
    pip install -r ../requirements.txt
    ```
2. Install ffmpeg.
3. Download Clotho and Walking Tour datasets.
4. Set the path to the downloaded datasets and the save directory in "config.yaml".
5. Run the following command to generate Clotho-Moments:
    ```bash
    python 1_collect_data.py
    python 2_covert_bg.py
    python 3_clip_bg.py
    python 4_clip_fg.py
    python 5_create_recipe.py
    python 6_create_dataset.py
    ```

## Reproduce the results
After executing `python 5_create_recipe.py`, you can reproduce the results by overwriting the `/SAVE_DIR/json/recipe_*.json` by the provided recipe files this repository.

Note that if your move the save directory, you need to change the path in the recipe files.
