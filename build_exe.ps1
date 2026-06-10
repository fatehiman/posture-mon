# Builds dist\PostureMonitor.exe with PyInstaller.
# Run from this folder:  .\build_exe.ps1
# Requires: py -3.10 -m pip install -r requirements.txt pyinstaller

# Regenerate the icon from the five-dot mark.
py -3.10 -c "import posture_monitor as pm; pm.make_icon_image(256).save('icon.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"

# NOTE: --collect-all mediapipe would drag in mediapipe.model_maker, which pulls
# TensorFlow / PyTorch / JAX (~2.5 GB). The pose Solutions API needs none of them,
# so we exclude those heavy frameworks. Keeps the exe to a few hundred MB.
py -3.10 -m PyInstaller --noconfirm --clean --windowed --onefile `
    --name PostureMonitor `
    --icon icon.ico `
    --collect-all mediapipe `
    --collect-all pystray `
    --collect-all comtypes `
    --exclude-module mediapipe.model_maker `
    --exclude-module tensorflow `
    --exclude-module tensorboard `
    --exclude-module torch `
    --exclude-module torchvision `
    --exclude-module torchaudio `
    --exclude-module jax `
    --exclude-module jaxlib `
    --exclude-module pandas `
    posture_monitor.py

Write-Host "`nDone -> dist\PostureMonitor.exe"
