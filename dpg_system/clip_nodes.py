from dpg_system.torch_base_nodes import *

from transformers import CLIPTokenizer, CLIPTextModel

# REQUIRES hugging face transformers and pytorch


_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP_MAX_LENGTH = 77


def register_clip_nodes():
    Node.app.register_node('clip_embedding', ClipEmbeddingNode.factory)
    Node.app.register_node('clip_embedding_length', ClipEmbeddingDistanceNode.factory)


class ClipNode(Node):
    model = None
    tokenizer = None
    inited = False

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        if not self.__class__.inited:
            try:
                self.__class__.model = CLIPTextModel.from_pretrained(_CLIP_MODEL_NAME)
                self.__class__.tokenizer = CLIPTokenizer.from_pretrained(_CLIP_MODEL_NAME)
                self.__class__.inited = True
            except Exception as e:
                print('clip_nodes: failed to load CLIP model/tokenizer:', e)
                self.__class__.model = None
                self.__class__.tokenizer = None
                # Leave inited=False so a later instance can retry once the
                # environment (network/cache) is fixed.


class ClipEmbeddingNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        if self.__class__.model is None or self.__class__.tokenizer is None:
            return
        raw = self.input()
        if raw is None:
            return
        text = any_to_string(raw)
        if not text:
            return
        try:
            with torch.no_grad():
                tokens = self.__class__.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=_CLIP_MAX_LENGTH,
                    return_tensors="pt",
                )
                outputs = self.__class__.model(**tokens)
                pooled_output = outputs.pooler_output
                embedding = pooled_output[0].detach().cpu().numpy()
        except Exception as e:
            print('clip_embedding: inference failed:', e)
            return
        self.output.send(embedding)


class ClipEmbeddingDistanceNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('input', triggers_execution=True)
        self.output = self.add_output('output')

    def execute(self):
        if self.__class__.model is None or self.__class__.tokenizer is None:
            return
        raw = self.input()
        if raw is None:
            return
        text = any_to_string(raw)
        if not text:
            return
        try:
            with torch.no_grad():
                tokens = self.__class__.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=_CLIP_MAX_LENGTH,
                    return_tensors="pt",
                )
                outputs = self.__class__.model(**tokens)
                pooled_output = outputs.pooler_output
                # The variable is named euclidean_length; previously this
                # was sum(x^2), i.e. the squared length. Take the sqrt so
                # the value actually matches its name.
                euclidean_length = torch.sqrt(torch.pow(pooled_output[0], 2).sum()).item()
        except Exception as e:
            print('clip_embedding_length: inference failed:', e)
            return
        self.output.send(euclidean_length)
