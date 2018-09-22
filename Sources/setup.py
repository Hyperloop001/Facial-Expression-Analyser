from distutils.core import setup
import py2exe, sys, os

sys.argv.append('py2exe')
setup(
    options={
        'py2exe': {
            'compressed': 2,
            'optimize': 2,
            'includes': includes,
            'excludes': excludes,
            'packages': packages,
            'dll_excludes': dll_excludes,
            'bundle_files': 1,  # 1 = .exe; 2 = .zip; 3 = separate
            'dist_dir': 'dist',  # Put .exe in dist/
            'xref': False,
            'skip_archive': False,
            'ascii': False,
            'custom_boot_script': '',
            # 'unbuffered': True,  # Immediately flush output.
        }
    },
    zipfile=None,  # Put libs into .exe to save space.
)