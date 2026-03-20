import os
import shutil
import threading
import traceback
import numpy as np
from PIL import Image
from dpg_system.node import Node
from dpg_system.conversion_utils import *

# Allow MPS to use all available memory instead of capping at ~36 GiB
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')


def register_vision_describe_nodes():
    Node.app.register_node('vision_describe', MoondreamDescribeNode.factory)
    Node.app.register_node('vision_describe_smol', SmolVLMDescribeNode.factory)
    Node.app.register_node('vision_describe_qwen', QwenVLDescribeNode.factory)


# ═══════════════════════════════════════════════════════════════════════════════
#  Base class — shared threading, image prep, and execute logic
# ═══════════════════════════════════════════════════════════════════════════════

class VisionDescribeBase(Node):
    """
    Abstract base for vision description nodes.  Subclasses implement
    _load_model() and _run_inference() for their specific VLM backend.

    Common features:
      • background worker thread with drop-frame strategy
      • image downscaling / RGBA→RGB
      • device auto-detection (CUDA → MPS → CPU)
    """

    # Subclasses MUST define their own class-level singletons:
    #   _model, _processor, _model_loaded, _model_lock, _device

    # ------------------------------------------------------------------ init
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # --- inputs ---
        self.image_input = self.add_input('image', triggers_execution=True)
        self.prompt_input = self.add_input(
            'prompt', widget_type='text_input',
            default_value='Describe this scene concisely.',
        )
        self.max_tokens_input = self.add_input(
            'max_tokens', widget_type='drag_int',
            default_value=150, min=16, max=1024,
        )
        self.max_image_dim_input = self.add_input(
            'max_image_dim', widget_type='drag_int',
            default_value=384, min=128, max=1024,
        )

        # --- outputs ---
        self.description_output = self.add_output('description')

        # --- options ---
        self.device_option = self.add_option(
            'device', widget_type='combo', default_value='auto',
        )
        self.device_option.widget.combo_items = ['auto', 'cpu', 'mps', 'cuda']

        # Subclass hook — add model-specific options
        self._add_options()

        # --- threading state ---
        self._pending_frame = None
        self._pending_lock = threading.Lock()
        self._worker_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()

    # ---------------------------------------------- subclass hooks (override)
    def _add_options(self):
        """Override to add model-specific options (mode, model size, etc.)."""
        pass

    def _load_model(self, device, dtype):
        """Load the model/processor into cls._model. Must set cls._model_loaded = True."""
        raise NotImplementedError

    def _run_inference(self, pil_image, prompt, max_tokens):
        """Run inference and return the description string."""
        raise NotImplementedError

    def _get_extra_params(self):
        """Return dict of model-specific params from options (for execute)."""
        return {}

    # --------------------------------------------------------- device helpers
    @classmethod
    def _detect_device(cls, requested_device='auto'):
        import torch
        if requested_device == 'auto':
            if torch.cuda.is_available():
                cls._device = 'cuda'
                return torch.float16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                cls._device = 'mps'
                return torch.float16
            else:
                cls._device = 'cpu'
                return torch.float32
        else:
            cls._device = requested_device
            if requested_device == 'cuda':
                return torch.float16
            elif requested_device == 'mps':
                return torch.float16
            else:
                return torch.float32

    # --------------------------------------------------------- image helpers
    @staticmethod
    def _prepare_image(image_array, max_dim):
        """Downscale numpy image and convert to PIL."""
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        pil_image = Image.fromarray(image_array)

        w, h = pil_image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        return pil_image

    # ------------------------------------------------------- execute (UI thread)
    def execute(self):
        image_data = self.image_input()
        if image_data is None or not isinstance(image_data, np.ndarray):
            return

        prompt = self.prompt_input()
        if not isinstance(prompt, str) or prompt.strip() == '':
            prompt = 'Describe this scene concisely.'

        max_tokens = int(self.max_tokens_input())
        max_dim = int(self.max_image_dim_input())
        extra = self._get_extra_params()

        with self._pending_lock:
            self._pending_frame = (image_data.copy(), prompt, max_tokens, max_dim, extra)
        self._worker_event.set()

    # ------------------------------------------------------- background worker
    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._worker_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                break
            self._worker_event.clear()

            with self._pending_lock:
                pending = self._pending_frame
                self._pending_frame = None
            if pending is None:
                continue

            image_array, prompt, max_tokens, max_dim, extra = pending

            try:
                import torch

                # Ensure model is loaded
                cls = self.__class__
                if not cls._model_loaded:
                    requested_device = self.device_option()
                    dtype = cls._detect_device(requested_device)
                    self._load_model(cls._device, dtype)

                # Clear MPS cache before inference
                if cls._device == 'mps':
                    torch.mps.empty_cache()

                pil_image = self._prepare_image(image_array, max_dim)

                with torch.no_grad():
                    answer = self._run_inference(pil_image, prompt, max_tokens, **extra)

                self.description_output.send(answer)

                if cls._device == 'mps':
                    torch.mps.empty_cache()

            except Exception as e:
                node_name = self.__class__.__name__
                print(f'[{node_name}] Inference error: {e}')
                traceback.print_exc()

    # ------------------------------------------------------- cleanup
    def cleanup(self):
        self._stop_event.set()
        self._worker_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Moondream2 (2B) — existing, uses custom query() / caption() API
# ═══════════════════════════════════════════════════════════════════════════════

class MoondreamDescribeNode(VisionDescribeBase):
    """Vision description using Moondream2 (vikhyatk/moondream2, ~2B params)."""

    _model = None
    _processor = None
    _model_loaded = False
    _model_lock = threading.Lock()
    _device = None

    @staticmethod
    def factory(name, data, args=None):
        return MoondreamDescribeNode(name, data, args)

    def _add_options(self):
        self.mode_option = self.add_option(
            'mode', widget_type='combo', default_value='query',
        )
        self.mode_option.widget.combo_items = ['query', 'caption']

        self.caption_length_option = self.add_option(
            'caption_length', widget_type='combo', default_value='normal',
        )
        self.caption_length_option.widget.combo_items = ['short', 'normal', 'long']

    def _get_extra_params(self):
        return {
            'mode': self.mode_option(),
            'caption_length': self.caption_length_option(),
        }

    def _load_model(self, device, dtype):
        cls = self.__class__
        with cls._model_lock:
            if cls._model_loaded:
                return
            from transformers import AutoModelForCausalLM

            model_id = 'vikhyatk/moondream2'
            local_model_dir = os.path.join(
                os.path.dirname(__file__), 'models', 'moondream2'
            )

            load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)

            if os.path.isdir(local_model_dir) and os.listdir(local_model_dir):
                print(f'[Moondream] Loading from local: {local_model_dir} …')
                # Copy dynamic modules to HF cache if needed
                modules_cache = os.path.join(
                    os.path.expanduser('~'), '.cache', 'huggingface',
                    'modules', 'transformers_modules', 'moondream2'
                )
                if not os.path.isdir(modules_cache) or not os.path.exists(
                    os.path.join(modules_cache, 'layers.py')
                ):
                    os.makedirs(modules_cache, exist_ok=True)
                    for f in os.listdir(local_model_dir):
                        if f.endswith('.py'):
                            shutil.copy2(os.path.join(local_model_dir, f), modules_cache)
                cls._model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir, **load_kwargs
                ).to(device)
            else:
                print(f'[Moondream] Loading {model_id} from HuggingFace …')
                try:
                    cls._model = AutoModelForCausalLM.from_pretrained(
                        model_id, **load_kwargs
                    ).to(device)
                except (OSError, Exception) as e:
                    print(f'[Moondream] Online load failed ({e}), using HF cache …')
                    cls._model = AutoModelForCausalLM.from_pretrained(
                        model_id, local_files_only=True, **load_kwargs
                    ).to(device)

            cls._model.eval()
            cls._model_loaded = True
            print(f'[Moondream] Model ready on {device} ({dtype})')

    def _run_inference(self, pil_image, prompt, max_tokens, mode='query', caption_length='normal'):
        cls = self.__class__
        settings = {"max_tokens": max_tokens}
        if mode == 'caption':
            result = cls._model.caption(pil_image, length=caption_length, settings=settings)
            return result.get("caption", "")
        else:
            result = cls._model.query(pil_image, question=prompt, settings=settings)
            return result.get("answer", "")


# ═══════════════════════════════════════════════════════════════════════════════
#  SmolVLM — lightweight VLM (256M / 500M / 2.2B)
# ═══════════════════════════════════════════════════════════════════════════════

class SmolVLMDescribeNode(VisionDescribeBase):
    """Vision description using SmolVLM (HuggingFace, 256M–2.2B params)."""

    _model = None
    _processor = None
    _model_loaded = False
    _model_lock = threading.Lock()
    _device = None
    _current_size = None

    MODELS = {
        '256M': 'HuggingFaceTB/SmolVLM-256M-Instruct',
        '500M': 'HuggingFaceTB/SmolVLM-500M-Instruct',
        '2.2B': 'HuggingFaceTB/SmolVLM-Instruct',
    }

    @staticmethod
    def factory(name, data, args=None):
        return SmolVLMDescribeNode(name, data, args)

    def _add_options(self):
        self.model_size_option = self.add_option(
            'model_size', widget_type='combo', default_value='500M',
        )
        self.model_size_option.widget.combo_items = ['256M', '500M', '2.2B']

    def _get_extra_params(self):
        return {}

    def _load_model(self, device, dtype):
        cls = self.__class__
        size = self.model_size_option()
        with cls._model_lock:
            if cls._model_loaded and cls._current_size == size:
                return
            from transformers import AutoProcessor, AutoModelForVision2Seq

            model_id = cls.MODELS.get(size, cls.MODELS['500M'])
            print(f'[SmolVLM] Loading {model_id} ({size}) …')

            try:
                cls._processor = AutoProcessor.from_pretrained(model_id)
                cls._model = AutoModelForVision2Seq.from_pretrained(
                    model_id, torch_dtype=dtype,
                ).to(device)
            except (OSError, Exception) as e:
                print(f'[SmolVLM] Online load failed ({e}), trying HF cache …')
                cls._processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                cls._model = AutoModelForVision2Seq.from_pretrained(
                    model_id, torch_dtype=dtype, local_files_only=True,
                ).to(device)

            cls._model.eval()
            cls._model_loaded = True
            cls._current_size = size
            print(f'[SmolVLM] Model ready on {device} ({dtype})')

    def _run_inference(self, pil_image, prompt, max_tokens):
        cls = self.__class__

        # Build chat-format messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = cls._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = cls._processor(
            text=text_prompt, images=[pil_image], return_tensors="pt"
        ).to(cls._device)

        generated_ids = cls._model.generate(**inputs, max_new_tokens=max_tokens)
        # Decode only the generated tokens (skip the input)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        answer = cls._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Qwen2.5-VL-3B — higher quality VLM
# ═══════════════════════════════════════════════════════════════════════════════

class QwenVLDescribeNode(VisionDescribeBase):
    """Vision description using Qwen2.5-VL-3B-Instruct (high quality)."""

    _model = None
    _processor = None
    _model_loaded = False
    _model_lock = threading.Lock()
    _device = None

    @staticmethod
    def factory(name, data, args=None):
        return QwenVLDescribeNode(name, data, args)

    def _add_options(self):
        pass  # no extra options for Qwen

    def _load_model(self, device, dtype):
        import torch
        cls = self.__class__
        # Qwen2.5-VL-3B has tensors exceeding MPS's 4 GB NDArray limit
        if device == 'mps':
            print('[QwenVL] MPS not supported for Qwen2.5-VL-3B (tensor > 4 GB), falling back to CPU')
            device = 'cpu'
            dtype = torch.float32
            cls._device = 'cpu'
        with cls._model_lock:
            if cls._model_loaded:
                return
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
            print(f'[QwenVL] Loading {model_id} …')

            try:
                cls._processor = AutoProcessor.from_pretrained(model_id)
                cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=dtype,
                ).to(device)
            except (OSError, Exception) as e:
                print(f'[QwenVL] Online load failed ({e}), trying HF cache …')
                cls._processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=dtype, local_files_only=True,
                ).to(device)

            cls._model.eval()
            cls._model_loaded = True
            print(f'[QwenVL] Model ready on {device} ({dtype})')

    def _run_inference(self, pil_image, prompt, max_tokens):
        cls = self.__class__

        # Build Qwen VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Use qwen_vl_utils to process the messages
        from qwen_vl_utils import process_vision_info

        text_prompt = cls._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = cls._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(cls._device)

        generated_ids = cls._model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        answer = cls._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()
