"""Start SAVES Studio, the synchronization desktop app.

The previous MaD GUI viewer is still available with ``python main_madgui.py``.
"""

import sys

from studio.ui.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
