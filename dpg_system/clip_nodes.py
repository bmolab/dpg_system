from dpg_system.torch_base_nodes import *

from transformers import CLIPTokenizer, CLIPTextModel

# REQUIRES hugging face transformers and pytorch


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
            self.__class__.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.__class__.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.__class__.inited = True


class ClipEmbeddingNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input = any_to_string(self.input())
        with torch.no_grad():
            tokens = self.__class__.tokenizer([input], padding=True, return_tensors="pt")
            outputs = self.__class__.model(**tokens)
            pooled_output = outputs.pooler_output
            embedding = pooled_output[0].numpy()
        self.output.send(embedding)


class ClipEmbeddingDistanceNode(ClipNode):
    @staticmethod
    def factory(name, data, args=None):
        node = ClipEmbeddingDistanceNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input("input", triggers_execution=True)
        self.output = self.add_output("output")

    def execute(self):
        input = any_to_string(self.input())
        tokens = self.__class__.tokenizer([input], padding=True, return_tensors="pt")
        outputs = self.__class__.model(**tokens)
        pooled_output = outputs.pooler_output
        euclidean_length = torch.pow(pooled_output[0], 2).sum()

        self.output.send(euclidean_length.item())

