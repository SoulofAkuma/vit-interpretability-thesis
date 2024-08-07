### Getting the Dataset
We will be using the Validation part of the ImageNet-1k Dataset in this project. You can go and download it from [Hugging Face](https://huggingface.co/datasets/imagenet-1k) where the file can be found under `data/val_images.tar.gz`. Afterward, you should unpack the archive into a folder and can now use the `scripts/prepare_data.py` script to prepare your data. You can either copy the script next to the folder containing all the images, name the image folder `val` and just run the script with `python prepare_data.py` or you can call the script with an argument containing the path to your folder with the unpacked images like `python scripts/prepare_data --image-dir C:\Path\to\unpacked\images\folder`. This will structure the images in place into respective folders with their class name and remove the class name from the image's file name. 

### Using the `TestNotebook.ipynb` Notebook
If you wish to use the test notebook, please run the command `git update-index --skip-worktree TestNotebook.ipynb` to keep the changes to your notebook on your local machine and not commit it to the repo.

### Setting up the Environment
```sh
conda create --name vit-interpretability-thesis python=3.9
conda activate vit-interpretability-thesis
python -m pip install -r requirements.txt
python -m pip install .
```