def get_project_root() -> Path:
    """
    Gets the root directory of the project.
    """
    from pathlib import Path

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
