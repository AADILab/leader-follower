""" Load in environment from default config and run it with randomly generated policies """
from lib.file_helper import loadConfig
from lib.watch_helpers import watchConfig


# Loading in default config
config = loadConfig()
# Watch that config
watchConfig(config)
