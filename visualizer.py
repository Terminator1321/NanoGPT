class TransformerRecorder:
    def __init__(self):
        self.data = {}

    def record(self, name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]

            t = out.detach().cpu().float()

            self.data[name] = {
                "mean":   float(t.mean()),
                "std":    float(t.std()),
                "min":    float(t.min()),
                "max":    float(t.max()),
                "shape":  list(t.shape),
                "sample": (
                    t[0, -1, :32].tolist() if t.dim() >= 3
                    else t.flatten()[:32].tolist()
                ),
            }

        return hook

    def clear(self):
        self.data = {}


recorder = TransformerRecorder()


class ThinkingTimeline:
    def __init__(self):
        self.frames = []

    @property
    def timeline(self):
        return self.frames

    def add(self, stage, meta=None):
        self.frames.append({"stage": stage, "meta": meta or {}})

    def clear(self):
        self.frames = []


timeline = ThinkingTimeline()