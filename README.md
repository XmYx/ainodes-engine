# ainodes-engine

Simple Qt Node engine written in Python based on NodeGraphQt

https://github.com/jchanvfx/NodeGraphQt

all nodes are in the custom_nodes directory,
each node has to inherit from the base_widgets,
and if a layout is to be added, it must be
inherited from the layouts.

TODO: Execution node 

(Currently the Diffusers_node has the run function, which emits a signal for the NodeGraph 
to execute the first node, this logic is a little flawed, we can pass the origin, and execute that,
we also have to consider the BackDrop node, which is not executable)

Please see the nodes as reference, but some main functions of the nodes are:
self.add_output(string)
self.create_property(name, value)
self.get_input(name)
self.get_output(name)
