from pathlib import Path
import argparse
from genial.utils.file_compression import FileCompressionHandler
from genial.utils.utils import process_pool_helper
from loguru import logger
import tqdm
from time import time


def compress_all_files(dirpath: Path, levels: int = 1, src_extension: str = ".v"):
    """
    Compress all .txt files in the given directory using the specified compression method.

    :param dirpath: Path to the directory containing text files.
    """

    if levels != 1:
        raise NotImplementedError()

    logger.info(
        f"Compressing all files of extension {src_extension} in all directories at level {levels} in {dirpath} ..."
    )
    compressed_filepaths = []
    start_time = time()
    with tqdm(total=len(list(dirpath.iterdir())), desc=f"x64| File Compression") as pbar:  # Progress bar
        compressed_filepaths = process_pool_helper(
            func=FileCompressionHandler._compress_file_in_subpath,
            func_args_gen=((subpath, src_extension) for subpath in dirpath.iterdir()),
            pbar=pbar,
        )
    end_time = time()
    logger.info(
        f"Parsed and compressed {len(compressed_filepaths)} files in {(end_time - start_time) / 60}min using 64 workers."
    )

    return compressed_filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress all text files in a directory.")
    parser.add_argument("--dirpath", type=str, required=True, help="Path to the directory containing text files.")
    # parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'bz2', 'lzma'],
    # help='Compression method to use (gzip, bz2, lzma). Default is bz2.')
    args = parser.parse_args()

    dirpath = Path(args.dirpath)
    assert dirpath.exists()
    prompt_answer = input(
        f"Wraning, this operation is going to compress and delete files in all level 1 folders of {dirpath}, do you want to continue? [Yes/No]"
    )
    if prompt_answer.lower()[0] == "y":
        compressed_files = compress_all_files(dirpath=dirpath)

    logger.info(f"Done.")
