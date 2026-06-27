import os
import sys
import subprocess
import shutil

def main():
    print("Preparing to build standalone executable for Vision-X Dashboard...")
    
    # Check if pyinstaller is installed
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller not found. Installing pyinstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        import PyInstaller.__main__

    import customtkinter
    
    # Locate the customtkinter directory to include its assets (fonts, themes, etc.)
    ctk_path = os.path.dirname(customtkinter.__file__)
    print(f"Found CustomTkinter path: {ctk_path}")

    # Use correct path separator for PyInstaller --add-data (':' for Linux/macOS, ';' for Windows)
    sep = ';' if sys.platform.startswith('win') else ':'
    
    # Construct the PyInstaller command arguments
    # --onefile packages everything into a single binary
    # --noconsole / --windowed runs without opening a terminal window
    args = [
        'gui.py',
        '--name=Vision-X_Dashboard',
        '--onefile',
        '--windowed',
        f'--add-data={ctk_path}{sep}customtkinter',
        '--clean'
    ]
    
    print(f"Running PyInstaller with arguments: {args}")
    try:
        PyInstaller.__main__.run(args)
        print("\nBuild completed successfully!")
        
        # Output info
        output_dir = os.path.abspath('dist')
        executable_name = "Vision-X_Dashboard.exe" if sys.platform.startswith('win') else "Vision-X_Dashboard"
        executable_path = os.path.join(output_dir, executable_name)
        print(f"\nStandalone executable is located at:\n{executable_path}\n")
    except Exception as e:
        print(f"Error occurred during build: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
