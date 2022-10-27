# dpg_system
system for supporting ui and nodes using dearpygui

dpg_system creates an enhanced dearpygui-based environment for building out node-based playgrounds for quick ui work in python.

Simple Example:

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

The selected node(s) can be duplicated by pressing 'd'.

Other shortcuts:
'i' creates an integer ui node 
'f' creates a float ui node
'v' creates a vector ui node
'b' creates a button node
't' creates a toggle (checkbox) ui node
'm' creates a message ui node (which sends / receives strings and lists

when you have selected a node, pressing 'o' shows you any additional options that you can set for this node (pressing it again makes this disappear.)

You can save and load node patches from the 'File' menu. They are saved as json files that preserve the state of all properties and options.

You can have multiple node windows in separate tabs in the dpg_system window.

_Custom Nodes

A simple node can be created by creating a sub-class of Node.

```
class AdditionNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AdditionNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("in", trigger_node=self)
        self.operand = 0
        if args is not None and len(args) > 0:
            self.operand, value_type = decode_arg(args, 0)
        self.operand_input = self.add_input("operand", widget_type='drag_float', default_value=self.operand)
        self.output = self.add_output("sum")

    def execute(self):
        if self.input.fresh_input:
            data = self.input.get_received_data()
            operand = self.operand_input.get_widget_value()
            sum = data + operand
            self.output.send(sum)```
            
The static method at the start creates a 'factory' for creating these nodes. This will always be the same except that the name of the node being created must match the name of the class.

The init method must call __init__ for the superclass (Node), then creates inputs and outputs, sets internal values, etc.

The execute method is called when new input is received in self.input. Note that self.input is created with the argument trigger_node=self. This indicates that any input received in that input should cause the node to execute.
Note also that the self.operand_input is not created with 'trigger_node=self', meaning that input received in this input does not cause the node to execute.

Much more complicated behaviours are of course possible. For example, If you define a 'frame_task(self)' method, it is called once per update of the dpg_system, which is usually at 60 hz.

This frame_task can for example call the 'execute' function to output every single dpg_update cycle.

You can also add callbacks to the widgets so that if the user changes the value of the widget, either an internal value is updated, or perhaps the execute method is called.



