from pathlib import Path


def get_project_root() -> Path:
    """
    Gets the root directory of the project.
    """
    return Path(__file__).parent.parent.parent
