"""
SAVES Studio: desktop app to synchronize offline sensors and video after a
laboratory session.

The modules directly under ``studio`` hold no Qt code (project file, file
discovery, signal cache, synchronization engine, video decoding) so they can
be tested and reused headless; ``studio.ui`` holds the windows.
"""
