import requests
from dpg_system.node import Node
import json

# TODO, true layer support. right now layer 20 is half hard-coded

def register_neuronpedia_node():
    Node.app.register_node('neuronpedia_search', NeuronpediaSearchNode.factory)


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
        
    def execute(self):
        search_text = self.search_input()
        self.model = self.model_input()
        self.layer = self.layer_input()
        
        if not search_text:
            print("Please enter search text")
            return
            
        url = "https://www.neuronpedia.org/api/explanation/search"
        
        payload = {
            "modelId": self.model,
            "layers": [self.layer],
            "query": search_text,
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response_data = response.json()
            
            if 'results' in response_data:
                top_indices = response_data['results'][:3]
                
                for info in top_indices:
                    idx = info['index']
                    if idx not in self.intervention_indices:
                        description = ""
                        try:
                            if ('explanations' in info["neuron"] and 
                                    info["neuron"]["explanations"]):
                                first_explanation = info['neuron']['explanations'][0]
                                if 'description' in first_explanation:
                                    description = first_explanation['description']
                        except Exception as e:
                            print(f"Error getting description: {e}")
                            
                        self.intervention_indices[idx] = {
                            "description": description,
                            "votes": 1
                        }
                    else:
                        self.intervention_indices[idx]['votes'] += 1

                results = []
                for idx, info in self.intervention_indices.items():
                    results.append([idx, info["description"], info["votes"]])
                
                print("encoded:", json.dumps(results))
                self.output.send(json.dumps(results))
            else:
                print("No features found in response")
                
        except Exception as e:
            print(f"Error searching for features: {str(e)}")
