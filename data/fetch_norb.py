import gzip
import os
import struct
import urllib.request
import numpy as np

## sets up target destination dir
BASE_DIR = "norb" ## create local dir (often with "data/") to store norb arrays
os.makedirs(BASE_DIR, exist_ok=True)
## points to original NYU host locations for small NORB dataset
HOST_URL = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
FILES = {
    "trainX": "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
    "trainY": "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
    "testX": "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
    "testY": "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
}

def parse_norb_matrix(file_path):
    ## parses NYU's custom binary matrix format (.mat) from compressed gzip stream
    with gzip.open(file_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        ndim = struct.unpack("<I", f.read(4))[0]

        num_dims_to_read = max(ndim, 3)
        shape = list(struct.unpack("<" + "I" * num_dims_to_read, f.read(4 * num_dims_to_read)))
        shape = shape[:ndim]

        if magic == 0x1E3D4C51:  # float32
            dtype = np.float32
        elif magic == 0x1E3D4C54:  # int32
            dtype = np.int32
        elif magic == 0x1E3D4C55:  # uint8
            dtype = np.uint8
        else:
            raise ValueError(f"Unknown header format signature: {hex(magic)}")

        raw_data = np.frombuffer(f.read(), dtype=dtype)
        return raw_data.reshape(shape)


def main():
    NUM_CLASSES = 5  # small NORB has 5 classes (0 to 4)

    for label, filename in FILES.items():
        compressed_path = os.path.join(BASE_DIR, filename)
        final_npy_path = os.path.join(BASE_DIR, f"{label}.npy")

        if not os.path.exists(final_npy_path):
            if not os.path.exists(compressed_path):
                print(f">>> Downloading {filename} from remote host...")
                urllib.request.urlretrieve(HOST_URL + filename, compressed_path)
            print(f">>> Parsing raw matrix stream for {label}...")
            arr = parse_norb_matrix(compressed_path)

            if "X" in label:
                print(">>> Normalizing pixel values to [0.0, 1.0]...")
                arr = arr.astype(np.float32) / 255.0
                ## unpack original raw shape (24300, 2, 96, 96) -> (48600, 9216)
                print(">>> Flattening left/right stereo image pairs into 48,600 row samples...")
                arr = arr.reshape(-1, 96 * 96)

            else:
                print(">>> Converting integer labels into one-hot encoded matrix...")
                arr = arr.flatten().astype(np.int32)
                arr = np.eye(NUM_CLASSES, dtype=np.float32)[arr]
                ## duplicate labels sequentially: (24300, 5) -> (48600, 5)
                print(">>> Duplicating one-hot vectors to pair with 48,600 image split sequence...")
                arr = np.repeat(arr, 2, axis=0)

            np.save(final_npy_path, arr)
            print(f">> Saved: {final_npy_path} | Target Shape: {arr.shape}\n")

            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        else:
            print(f">> Skipping {label}.npy (already exists in target folder).")

    print(">> Dataset setup successful! All arrays matched to 48,600 samples.")

## execute data loading/processing
if __name__ == "__main__":
    main()
