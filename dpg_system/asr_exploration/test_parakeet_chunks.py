import time, numpy as np, soundfile as sf, mlx.core as mx
from parakeet_mlx import from_pretrained
model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
y,sr = sf.read("speech_test.wav"); y=y.astype(np.float32)
for secs, ctx, depth in ((0.5,(256,256),1), (1.0,(256,256),1), (2.0,(256,256),1), (1.0,(128,32),1), (1.0,(64,16),1)):
    chunk=int(secs*sr); fin_prev=""; fin_rev=0; lat=[]; n=0
    with model.transcribe_stream(context_size=ctx, depth=depth) as s:
        for i in range(0, len(y), chunk):
            t=time.time(); s.add_audio(mx.array(y[i:i+chunk])); lat.append(time.time()-t); n+=1
            fin="".join(t_.text for t_ in s.finalized_tokens)
            if not fin.startswith(fin_prev): fin_rev+=1
            fin_prev=fin
    lat=np.array(lat)*1000
    print(f"chunk {secs}s ctx={ctx} depth={depth}: per-chunk ms mean {lat.mean():.0f} p95 {np.percentile(lat,95):.0f} max {lat.max():.0f} (real-time budget {secs*1000:.0f} ms); finalized revisions={fin_rev}")
    print("   final:", repr(s.result.text))
