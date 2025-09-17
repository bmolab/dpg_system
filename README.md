# dpg_system
system for supporting ui and nodes using dearpygui

dpg_system creates an enhanced dearpygui-based environment for building out node-based playgrounds for quick ui work in python. 

There is a growing collections of nodes for numpy functions, pytorch functions, torchvision, torchaudio, Kornia image processing, opencv image processing, spacy, and the framework can be expanded to support other libraries as well.

![](fft_sample_clip.jpg)
*An example patch doing windowed fft and ifft using pytorch*

__Requirements__

If you have conda installed, you can simply run the install.sh script which creates the conda environment, installs the required conda and pip packages and downloads spacy en_core_web_lg
```
cd dpg_system
./install.sh
```
You can also create the conda environment manually instead:
```
cd dpg_system
conda env create --file environment.yml
```
This will not install spacy en_core_web_lg, so you would need to do:
```
conda install -c conda-forge spacy
python -m spacy download en_core_web_lg
```

For torchaudio nodes, you need to install pyaudio

for windows:
```
python -m pip install pyaudio
```

for macOS (assumes you have homebrew installed):
```
brew install portaudio
pip install pyaudio
```

for Linux:
```
sudo apt install python3-pyaudio
```

__Simple Example__

This example creates and runs an instance of dgp_system.App. When run, you will see a window with a blank grid, which is the canvas on which you can place nodes. 

```
from dpg_system.dpg_app import App

dpg_app = App()
dpg_app.start()
dpg_app.run_loop()
```

To place a node, type 'n' with your mouse in the position you want the node.
Then start typing the name of the node you want to create. A list of fuzzy matches to your text will appear. You can select from this list, or keep typing.
You can accept the top suggestion by pressing return or space. Use space if you want to add additional arguments (i.e. to create a node that adds 4 to the input, type '+', <space>, '4' <return>)

The nodes can be moved around freely. You can connect an output of one node to the input of another by clicking and dragging from the circle representing an output to a circle representing an input on the other node.
(Inputs are always on the left and Outputs are always on the right)

You can select a link between nodes by clicking on it, and you can delete it by pressing <backspace>

You can select multiple nodes by dragging across them.

The selected node(s) can be duplicated by pressing command or control 'd'.
Copy is command or control 'c'
Cut is command or control 'x'
Paste is command or control 'v'

Other shortcuts:
-'i' creates an integer ui node 
-'f' creates a float ui node
-'v' creates a vector ui node
-'b' creates a button node
-'t' creates a toggle (checkbox) ui node
-'m' creates a message ui node (which sends / receives strings and lists

when you have selected a node, pressing 'o' shows you any additional options that you can set for this node (pressing it again makes this disappear.)

You can save and load node patches from the 'File' menu. They are saved as json files that preserve the state of all properties and options.

You can have multiple node windows in separate tabs in the dpg_system window.

__Custom Nodes__

A simple node can be created by creating a sub-class of Node.

```
class AdditionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AdditionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("in", triggers_execution=True)
        self.operand = 0
        if len(args) > 0:
            self.operand = any_as_float(args[0])
        self.operand_input = self.add_input("operand", widget_type='drag_float', default_value=self.operand)
        self.output = self.add_output("sum")

    def execute(self):
        if self.input.fresh_input:
            data = self.input()
            operand = self.operand_input()
            sum = data + operand
            self.output.send(sum)
            
```

You would also have to register this node thus:

```
Node.app.register_node("add", AdditionNode.factory)
```
So that you can create this node by name.

The static method at the start creates a 'factory' for creating these nodes. This will always be the same except that the name of the node being created must match the name of the class.

```
@staticmethod
def factory(name, data, args=None):
    node = AdditionNode(name, data, args)
    return node
```

The init method must call __init__ for the superclass (Node), then creates inputs and outputs, sets internal values, etc.

```
def __init__(self, label: str, data, args):
    super().__init__(label, data, args)
    self.input = self.add_input("in", triggers_execution=True)
    self.operand = 0
    if len(args) > 0:
        self.operand = any_to_float(args[0])
    self.operand_input = self.add_input("operand", widget_type='drag_float', default_value=self.operand)
    self.output = self.add_output("sum")
```
    
The execute method is called when new input is received in self.input. Note that self.input is created with the argument triggers_execution=True. This indicates that any input received in that input should cause the node to execute.
    
    
```
def execute(self):
    if self.input.fresh_input:
        data = self.input()
        operand = self.operand_input()
        sum = data + operand
        self.output.send(sum)
```
    
Note also that the self.operand_input is not created with 'trigger_execution=True', meaning that input received in this input does not cause the node to execute.

Much more complicated behaviours are of course possible. For example, If you define a 'frame_task(self)' method, it is called once per update of the dpg_system, which is usually at 60 hz.

This frame_task can for example call the 'execute' function to output every single dpg_update cycle.

You can also add callbacks to the widgets so that if the user changes the value of the widget, either an internal value is updated, or perhaps the execute method is called.


NEW ADDITIONS: all nodes now check data received at inputs to see if the data starts with the name of a property, option or input. If it does, then the remainder of the data is used to set that property, etc. For example if you have a property named 'frequency' then sending the message 'frequency 1.5' to any input would set the value of frequency to 1.5.

Relatedly, when you create a new node you can add arguments that specify the values of properties. For 'metro' you can create 'metro 30' for create a metro with a period of 30 milliseconds. But you could also be more specific and write 'metro period=30 units=seconds' which would set the period to 30 and the units to seconds, so the period would be 30 seconds.

All this named property argument and message handling is done automatically. You only need to handle arguments yourself in the case that you want to accept a list of values without property names. In the case that arguments include named and unnamed arguments, the named arguments are collected separately and set, then any unnamed arguments are passed, in their relative order, as the 'args' to the node's '__init__' function for you to process.

If you do not specify the property names, then the node assumes the values are in the order expected by any argument handling that the node does in its __init__ function.


