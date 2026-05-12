"""Get model weights for rf2t-micro."""

from typing import Optional

import os
import tarfile

from carabiner import print_err
import requests
from tqdm.auto import tqdm

_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"

def get_model_weights(
    model_name: Optional[str] = None,
    path: Optional[str] = None
) -> str:

    """Get the model weights filename.
    
    If not present in `path/.weights/af2-sppid`, download there.

    """
    if path is None:
        path = os.path.expanduser("~")
    if model_name is None:
        model_name = "model_1"

    weight_dir = os.path.join(os.path.realpath(path), ".weights", "af2-yunta", "params")
    weight_filename = os.path.join(weight_dir, f"params_{model_name}.npz")
    files_to_keep = (os.path.join(".", os.path.basename(weight_filename)), "LICENSE")

    if not os.path.exists(weight_filename):
        print_err(f"[INFO]  Weights file {weight_filename} does not exist. Downloading...")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        r = requests.get(_WEIGHTS_URL, stream=True)
        temp_file = os.path.join(weight_dir, "weights.tar.gz")
        try:
            with open(temp_file, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=200 * 1024**2)): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        except KeyboardInterrupt as e:
            print_err(f"[INFO] Deleting {temp_file}!")
            os.remove(temp_file)
            raise e
        with tarfile.open(temp_file) as tar:
            print_err(f"[INFO  Extracting weights from {temp_file} to {weight_dir}...")
            tar.extractall(
                path=weight_dir, 
                members=files_to_keep, 
                filter='data',
            )
        tardir = weight_dir
        os.remove(temp_file)
        for filename in files_to_keep:
            source = os.path.join(tardir, filename)
            destination = os.path.join(weight_dir, filename)
            print_err(f"[INFO] Moving {source} to {destination}.")
            os.rename(source, destination)

    print_err(f"[INFO] Model weights located at {weight_dir}.")        
    return os.path.dirname(weight_dir)  # AF2 expects the parent dir of "params" dir
