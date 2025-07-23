from pathlib import Path

def get_project_root() -> Path:
    """
    Gets the root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def get_includes(file: str, local: bool = False) -> list[str]:
    """
    Finds all the includes used in C/C++ in a string from a file.

    Args:
        file: The text of the code.
        local: Only gets the includes that are explicitly defined as being local
            (includes with quotes instead of ``<>``)

    Returns:
        A list of the includes (without ``#include ""``) from the given string.
    """
    import re

    pattern = '^\\W*#include\\W+(.+?)\\W*$'
    if local:
        pattern = '^\\W*#include\\s+"(.+?)"\\W*$'

    return re.findall(pattern, file, re.M)

def unique_file(filename: str) -> str:
    """
    Finds a unique name for a filename. Will append a padded 4-digit number to the
    tail of the stem of the file name.

    Args:
        filename: The filename to find a unique name.

    Returns:
        A unique filename.
    """
    import os

    i = 0
    stem, ext = os.path.splitext(filename)
    while os.path.isfile(filename):
        filename = f"{stem}_{str(i).zfill(4)}{ext}"
        i = i + 1

    return filename
