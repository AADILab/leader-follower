"""
@title

project_properties.py

@description

Common paths and attributes used by and for this project.

"""
import shutil
from pathlib import Path


# --------------------------------------------
# Project versioning and attributes
# --------------------------------------------
name = 'leader_follower'
version = '0.1'

# --------------------------------------------
# Base paths to use to place further packages
# --------------------------------------------
source_package = Path(__file__).parent
project_path = Path(source_package).parent

# --------------------------------------------
# Paths to store program and manually generated information
# --------------------------------------------
resources_dir = Path(project_path, 'resources')
log_dir = Path(project_path, 'logs')
data_dir = Path(project_path, 'data')
config_dir = Path(project_path, 'configs')
cached_dir = Path(project_path, 'cached')
output_dir = Path(project_path, 'output')
model_dir = Path(project_path, 'models')
doc_dir = Path(project_path, 'docs')

# --------------------------------------------
# Project specific paths
# --------------------------------------------
raw_data_dir = Path(data_dir, 'raw')
sound_bytes_dir = Path(resources_dir, 'sound_bytes')

# --------------------------------------------
# Cached directories
# these are generally assumed to be in cached_dir
# no guarantee that a cached dir will exist between runs
# --------------------------------------------

# --------------------------------------------
# Resource files
# paths to specific resource and configuration files
# --------------------------------------------
secrets_path = Path(resources_dir, 'project_secrets.json')

# --------------------------------------------
# Useful properties and values about the runtime environment
# --------------------------------------------
TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()
