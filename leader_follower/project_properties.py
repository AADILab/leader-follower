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
# Base paths for relative pathing to the project base
# --------------------------------------------
source_package = Path(__file__).parent
project_path = Path(source_package).parent

# --------------------------------------------
# Paths to store assets and related resources
# --------------------------------------------
resources_dir = Path(project_path, 'resources')
data_dir = Path(project_path, 'data')
doc_dir = Path(project_path, 'docs')

# --------------------------------------------
# Output directories
# Directories to programs outputs and generated artefacts
# --------------------------------------------
output_dir = Path(project_path, 'output')
model_dir = Path(project_path, 'models')
log_dir = Path(project_path, 'logs')

# --------------------------------------------
# Cached directories
# Used for caching intermittent and temporary states or information
# to aid in computational efficiency
# no guarantee that a cached dir will exist between runs
# --------------------------------------------
cached_dir = Path(project_path, 'cached')

# --------------------------------------------
# Test directories
# Directories to store test code and resources
# --------------------------------------------
test_dir = Path(project_path, 'test')
test_config_dir = Path(test_dir, 'test')

# --------------------------------------------
# Resource files
# paths to specific resource and configuration files
# --------------------------------------------
secrets_path = Path(resources_dir, 'project_secrets.json')
config_dir = Path(project_path, 'configs')

# --------------------------------------------
# Useful properties and values about the runtime environment
# --------------------------------------------
TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()
