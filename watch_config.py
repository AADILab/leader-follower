""" Load in environment from default config and run it with randomly generated policies """
from lib.file_helper import loadConfig
from lib.watch_helpers import watchConfig

# Loading in default config
config = loadConfig()
for i in range(10):
    # Watch that config
    watchConfig(config)
