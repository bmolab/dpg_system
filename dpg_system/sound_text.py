import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

try:
    from torchaudio.io import StreamReader
except ModuleNotFoundError:
    try:
        import google.colab

        print(
            """
            To enable running this notebook in Google Colab, install the requisite
            third party libraries by running the following code:

            !add-apt-repository -y ppa:savoury1/ffmpeg4
            !apt-get -qq install -y ffmpeg
            """
        )
    except ModuleNotFoundError:
        pass
    raise

import matplotlib.pyplot as plt

base_url = "https://download.pytorch.org/torchaudio/tutorial-assets"
AUDIO_URL = f"{base_url}/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
VIDEO_URL = f"{base_url}/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"

streamer = StreamReader(src="0:0", format="avfoundation", option={"framerate": "30", "pixel_format": "bgr0"},)
for i in range(streamer.num_src_streams):
    print(streamer.get_src_stream_info(i))