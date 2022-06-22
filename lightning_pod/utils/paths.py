import os
import errno


def create_target_path(filepath, target_directory):
    sep = os.path.sep
    real_path = os.path.realpath(filepath).split(sep)
    real_path = list(reversed(real_path))
    if target_directory in real_path:
        target_path_idx = real_path.index(target_directory) - 1
        target_path = filepath.parents[target_path_idx]
        return target_path
    else:
        raise NotADirectoryError(
            errno.ENOENT, os.strerror(errno.ENOENT), target_directory
        )
