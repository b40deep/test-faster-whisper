# Faster Whisper

## status
- dead at the moment. no plan to resurrect.

## tutorial
- by KARTIS on [YouTube](https://www.youtube.com/watch?v=uimBp3c3Koo)

## deps:
- `cuDNN 9.12.10`
- python btn 3.9 and 3.12 IIRC `py -3.10 -m venv .venv`
- `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129`
- then `pip install faster-whisper`
- then `pip install -r requirements.txt`

## troubles / todo:
- some words are skipped. might be my mic, or the smaller models i'm using
- hallucinations like `thank you` and the like keep streaming when nobody is speaking. Now i see why most systems like this use VAD to trigger the STT and cut off any chance of these. 
- switched to stock `Whisper` via `04 gradio docs.py`. It hallucinated and then errored out iwth Whisper's 30s limit
    `ValueError: You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features.`