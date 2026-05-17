import json

import requests

from dpg_system.node import Node


# TODO, true layer support. right now layer 20 is half hard-coded


def register_neuronpedia_node():
    Node.app.register_node('neuronpedia_search', NeuronpediaSearchNode.factory)


_NEURONPEDIA_URL = "https://www.neuronpedia.org/api/explanation/search"
_REQUEST_TIMEOUT_SECONDS = 15


class NeuronpediaSearchNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NeuronpediaSearchNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.intervention_indices = {}
        self.model = "gemma-2-9b"
        self.layer = "20-gemmascope-res-16k"

        self.search_input = self.add_input(
            'search text',
            widget_type='text_input'
        )
        self.model_input = self.add_input(
            'model',
            widget_type='text_input',
            default_value=self.model
        )
        self.layer_input = self.add_input(
            'layer',
            widget_type='text_input',
            default_value=self.layer
        )
        self.trigger = self.add_input(
            'search',
            widget_type='button',
            triggers_execution=True
            # trigger_button=True,
        )
        self.output = self.add_output('results')

    def _read_text(self, widget, fallback=''):
        try:
            value = widget()
        except Exception:
            return fallback
        if value is None:
            return fallback
        return str(value).strip()

    def execute(self):
        search_text = self._read_text(self.search_input)
        model = self._read_text(self.model_input, self.model)
        layer = self._read_text(self.layer_input, self.layer)

        if not search_text:
            print("neuronpedia_search: please enter search text")
            return
        if not model:
            print("neuronpedia_search: model is empty")
            return
        if not layer:
            print("neuronpedia_search: layer is empty")
            return

        # Persist the latest model/layer so the input widgets keep reflecting
        # what we actually queried with.
        self.model = model
        self.layer = layer

        payload = {
            "modelId": model,
            "layers": [layer],
            "query": search_text,
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                _NEURONPEDIA_URL,
                json=payload,
                headers=headers,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout:
            print(f"neuronpedia_search: request timed out after {_REQUEST_TIMEOUT_SECONDS}s")
            return
        except requests.exceptions.RequestException as e:
            print(f"neuronpedia_search: network error: {e}")
            return

        if not response.ok:
            print(f"neuronpedia_search: HTTP {response.status_code}: {response.text[:200]}")
            return

        try:
            response_data = response.json()
        except ValueError as e:
            print(f"neuronpedia_search: failed to decode JSON: {e}")
            return

        if not isinstance(response_data, dict):
            print("neuronpedia_search: unexpected response shape (not an object)")
            return

        results_list = response_data.get('results')
        if not isinstance(results_list, list):
            print("neuronpedia_search: no features found in response")
            return

        top_indices = results_list[:3]

        for info in top_indices:
            if not isinstance(info, dict):
                continue
            idx = info.get('index')
            if idx is None:
                continue
            try:
                hash(idx)
            except TypeError:
                continue

            if idx not in self.intervention_indices:
                description = ""
                neuron = info.get('neuron')
                if isinstance(neuron, dict):
                    explanations = neuron.get('explanations')
                    if isinstance(explanations, list) and explanations:
                        first = explanations[0]
                        if isinstance(first, dict):
                            description = first.get('description', '') or ''
                self.intervention_indices[idx] = {
                    "description": description,
                    "votes": 1,
                }
            else:
                self.intervention_indices[idx]['votes'] += 1

        results = []
        for idx, info in self.intervention_indices.items():
            results.append([idx, info["description"], info["votes"]])

        try:
            encoded = json.dumps(results)
        except (TypeError, ValueError) as e:
            print(f"neuronpedia_search: failed to encode results: {e}")
            return

        print("encoded:", encoded)
        self.output.send(encoded)
