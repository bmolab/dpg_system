import threading
import time
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from dpg_system.node import Node
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn as nn

# https://github.com/huggingface/transformers/issues/36906
# https://github.com/huggingface/transformers/issues/36888
# lollll


def load_sae(device="cuda", repo_id="google/gemma-scope-2b-pt-res",filename="layer_20/width_16k/average_l0_71/params.npz"):
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    class JumpReLUSAE(nn.Module):
        def __init__(self, d_model, d_sae):
            # Note that we initialise these to zeros because we're loading in pre-trained weights.
            # If you want to train your own SAEs then we recommend using blah
            super().__init__()
            self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
            self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
            self.threshold = nn.Parameter(torch.zeros(d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.b_dec = nn.Parameter(torch.zeros(d_model))

        def encode(self, input_acts):
            pre_acts = input_acts @ self.W_enc + self.b_enc
            mask = (pre_acts > self.threshold)
            acts = mask * torch.nn.functional.relu(pre_acts)
            return acts

        def decode(self, acts):
            return acts @ self.W_dec + self.b_dec

        def forward(self, acts):
            acts = self.encode(acts)
            recon = self.decode(acts)
            return recon

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    return sae.to(device), params


def test_sae(sae, model, tokenizer, gather_residual_activations):
    prompt = "Would you be able to travel through time using a wormhole?"

    inputs = tokenizer.encode(prompt, return_tensors="pt",
                              add_special_tokens=True).to("cuda")
    print(inputs)

    outputs = model.generate(input_ids=inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))

    target_act = gather_residual_activations(model, 20, inputs)
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)

    with torch.no_grad():
        latent_acts = sae.encode(target_act.float())  # shape [batch, seq, 16384]
        recon = sae.decode(latent_acts)              # shape [batch, seq, 2304]

    return 1 - torch.mean((recon[:, 1:] - target_act[:, 1:].to(torch.float32)) **2) / (target_act[:, 1:].to(torch.float32).var())

model_infos = [
    {
        "name": "gemma-2-2b",
        "model": "google/gemma-2-2b",
        "tokenizer": "google/gemma-2-2b",
        "sae": "google/gemma-scope-2b-pt-res",
        "filename": "layer_20/width_16k/average_l0_71/params.npz",
        "intervention_indices": {
            11859: {"name": "strong emotions 2", "description": "strong emotions 2", "votes": 0},
        },
        "quantize_sae_vectors_4bit": False
    },
    {
        "name": "gemma-2-9b",
        "model": "unsloth/gemma-2-9b-bnb-4bit",  # 4 bit quant
        "tokenizer": "unsloth/gemma-2-9b-bnb-4bit",
        "sae": "google/gemma-scope-9b-pt-res",
        "filename": "layer_20/width_16k/average_l0_68/params.npz",
        "intervention_indices": {
            4253: {"name": "strong emotions 2", "description": "strong emotions 2", "votes": 0},
            12514: {"name": "strong emotions 3", "description": "strong emotions 3", "votes": 0},
            8398: {"name": "dogs", "description": "dogs", "votes": 0},
        },
        "quantize_sae_vectors_4bit": True
    },
    {
        "name": "gemma-2-27b",
        "model": "unsloth/gemma-2-27b-bnb-4bit",
        "tokenizer": "unsloth/gemma-2-27b-bnb-4bit",
        "sae": "google/gemma-scope-27b-pt-res",
        "filename": "layer_20/width_16k/average_l0_68/params.npz",
        "intervention_indices": {},
        "quantize_sae_vectors_4bit": True
    }
]

chosen_model = model_infos[1]

print("theoretically load the model in")
# sae, params = load_sae(repo_id=chosen_model["sae"], filename=chosen_model["filename"])
# pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
# model = AutoModelForCausalLM.from_pretrained(chosen_model["model"], device_map='auto', output_hidden_states=True)
# tokenizer = AutoTokenizer.from_pretrained(chosen_model["tokenizer"])

class GemmaNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return GemmaNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.prompt_input = self.add_input(
            'prompt',
            widget_type='text_input',
            default_value="The future of artificial intelligence is"
        )
        self.delay_input = self.add_input(
            'delay',
            widget_type='drag_float',
            min=0,
            default_value=1.0
        )
        self.temperature_input = self.add_input(
            'temperature',
            widget_type='drag_float',
            default_value=0.7,
            min=0,
            max=1,
        )
        self.top_k_input = self.add_input(
            'top_k',
            widget_type='drag_int',
            default_value=50
        )
        self.top_p_input = self.add_input(
            'top_p',
            widget_type='drag_float',
            default_value=0.9
        )
        self.intervention_active_input = self.add_input(
            'intervention_active',
            widget_type='checkbox',
            default_value=False
        )
        self.intervention_strength_input = self.add_input(
            'intervention_strength',
            widget_type='drag_int',
            default_value=0
        )
        self.interventions_input = self.add_input(
            'interventions',
            widget_type='text_input',
            default_value="[]"
        )
        self.start_button = self.add_input(
            'start',
            widget_type='button',
            triggers_execution=True
            # trigger_button=True,
        )
        self.pause_button = self.add_input(
            'pause',
            widget_type='button',
            callback=self.pause_generation
        )
        self.reset_button = self.add_input(
            'reset',
            widget_type='button',
            callback=self.reset_generation
        )
        self.output = self.add_output('generated_text')

        self.generated_text = ""
        self.paused = False
        self.reset_flag = False
        self.generation_thread = None
        self.hook_handle = None

    def execute(self):
        self.paused = False
        self.reset_flag = False
        prompt = self.prompt_input()
        delay = float(self.delay_input())
        if self.generation_thread is None or not self.generation_thread.is_alive():
            self.generation_thread = threading.Thread(
                target=self.generate_word_by_word,
                args=(prompt, delay, 0),
                daemon=True
            )
            self.generation_thread.start()

        # self.output.send(self.generated_text)

    def pause_generation(self):
        self.paused = True

    def reset_generation(self):
        self.reset_flag = True
        self.paused = False
        self.generated_text = ""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=1)
        self.generation_thread = None

    def generate_word_by_word(self, prompt, delay, depth):
        if depth > 10:
            return


        def grab_interventions():
            try:
                interventions_list = json.loads(self.interventions_input())
                if not isinstance(interventions_list, list):
                    return []
            except Exception:
                return []
            return interventions_list
        def attach_hook():
            if self.intervention_active and self.hook_handle is None:
                print("Theoretically attach a hook")
                # self.hook_handle = model.model.layers[20].register_forward_hook(self.steering_hook)
            elif not self.intervention_active and self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None

        if depth == 0:
            self.generated_text = ""

            self.interventions = grab_interventions()
            print("final interventions", self.interventions)

            self.intervention_active = bool(self.intervention_active_input())
            self.intervention_strength = float(self.intervention_strength_input())
            print(self.intervention_active, self.intervention_strength)
            attach_hook()

        print("Theoretically tokenize the prompt")
        # input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        # generated_ids = input_ids["input_ids"].clone()
        max_length = 50
        top_k_value = int(self.top_k_input())
        top_p_value = float(self.top_p_input())
        top_k_warper = TopKLogitsWarper(top_k=top_k_value)
        top_p_warper = TopPLogitsWarper(top_p=top_p_value)

        for _ in range(max_length):
            if self.paused:
                while self.paused:
                    if self.reset_flag:
                        if self.hook_handle is not None:
                            self.hook_handle.remove()
                            self.hook_handle = None
                        return
                    time.sleep(0.1)
            if self.reset_flag:
                if self.hook_handle is not None:
                    self.hook_handle.remove()
                    self.hook_handle = None
                return

            new_interventions = grab_interventions()
            if new_interventions != self.interventions:
                self.interventions = new_interventions
                print("Updated interventions:", new_interventions)

            new_active = bool(self.intervention_active_input())
            if new_active != self.intervention_active:
                self.intervention_active = new_active
                print("Updated intervention active:", new_active)
                attach_hook()

            new_strength = float(self.intervention_strength_input())
            if new_strength != self.intervention_strength:
                self.intervention_strength = new_strength
                print("Updated intervention strength:", new_strength)

            print("Theoretically generate output")
            # with torch.no_grad():
            #     outputs = model(input_ids=generated_ids)
            # next_token_logits = outputs.logits[:, -1, :] / float(self.temperature_input())
            # filtered_logits = top_k_warper(None, next_token_logits)
            # filtered_logits = top_p_warper(None, filtered_logits)
            # probabilities = torch.softmax(filtered_logits, dim=-1)
            # next_token = torch.multinomial(probabilities, num_samples=1)
            # generated_ids = torch.cat([generated_ids, next_token], dim=1)
            # new_word = tokenizer.decode(next_token.item())
            new_word = "TEST!"
            self.generated_text += new_word
            self.output.send(self.generated_text)
            # if next_token.item() == tokenizer.eos_token_id:
            #     break

            # update delay live
            delay = float(self.delay_input())
            time.sleep(delay)

        if self.hook_handle is not None and depth > 9:
            self.hook_handle.remove()
            self.hook_handle = None

        self.generate_word_by_word(self.generated_text, delay, depth + 1)

    def steering_hook(self, module, inputs, outputs):
        if self.intervention_active and self.intervention_strength != 0:
            activations = outputs[0]
            modified_activations = activations.clone()
            for intervention in self.interventions:
                if len(intervention) >= 3:
                    idx = int(intervention[0])
                    votes = intervention[2]
                    if votes != 0:
                        print("theoeretically apply intervention", idx, votes)
                        # intervention_vector = pt_params['W_dec'][idx].clone()
                        # if chosen_model["quantize_sae_vectors_4bit"]:
                        #     intervention_vector = intervention_vector.to(torch.float16)
                        # strength_coefficient = self.intervention_strength * votes
                        # modified_activations = modified_activations + strength_coefficient * intervention_vector
            return (modified_activations,) + outputs[1:] if len(outputs) > 1 else (modified_activations,)
        return outputs

def register_gemma_node():
    Node.app.register_node('gemma', GemmaNode.factory)
