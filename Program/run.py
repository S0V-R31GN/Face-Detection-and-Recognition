import subprocess, sys, os
REQUIREMENTS = ["flask", "opencv-python", "opencv-contrib-python", "numpy", "scikit-learn", "Pillow"]
def install():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *REQUIREMENTS])
def main():
    install()
    print("Starting App at http://127.0.0.1:5000")
    os.system(f"{sys.executable} app.py")
if __name__ == '__main__':
    main()
