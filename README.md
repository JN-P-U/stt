# stt

git clone <repository_url>
cd whisper

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows (cmd)

pip install -r requirements.txt

# ffmpeg 설치

    - mac : brew install ffmpeg
    - linux : sudo apt update && sudo apt install ffmpeg
    - window : 직접/chocho 등을 통해 설치

# ffmpeg tjfcl ghkrdls
ffmpeg -version

# 실행
python use_whisper.py --help