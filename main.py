"""Entry point: python main.py"""

import sys
import os

# ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from ml.app import MuseWaffleApp

if __name__ == "__main__":
    app = MuseWaffleApp()
    app.run()
