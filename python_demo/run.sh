#！/bin/bash

if [ -d ".venv" ]; then
    echo "source .venv/bin/activate"
    source .venv/bin/activate
else
    echo "please run source ../env.sh first"
    exit 1
fi

if [ -d "models" ]; then
  echo "python3 gr.py"
  python3 gr.py
else
  echo "No models, Downloading now"
  bash tar_downloader.sh
  tar -xvf models.tar.gz
  python3 gr.py
fi