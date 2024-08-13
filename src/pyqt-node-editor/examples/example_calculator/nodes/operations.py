from examples.example_calculator.calc_conf import register_node
from examples.example_calculator.calc_node_base import CalcNode


@register_node()
class CalcNode_Add(CalcNode):
    icon = "icons/add.png"
    op_title = "Add"
    content_label = "+"
    content_label_objname = "calc_node_bg"

    def evalOperation(self, input1, input2):
        return input1 + input2


@register_node()
class CalcNode_Sub(CalcNode):
    icon = "icons/sub.png"
    op_title = "Substract"
    content_label = "-"
    content_label_objname = "calc_node_bg"

    def evalOperation(self, input1, input2):
        return input1 - input2

@register_node()
class CalcNode_Mul(CalcNode):
    icon = "icons/mul.png"
    op_title = "Multiply"
    content_label = "*"
    content_label_objname = "calc_node_mul"

    def evalOperation(self, input1, input2):
        print('foo')
        return input1 * input2

@register_node()
class CalcNode_Div(CalcNode):
    icon = "icons/divide.png"
    op_title = "Divide"
    content_label = "/"
    content_label_objname = "calc_node_div"

    def evalOperation(self, input1, input2):
        return input1 / input2

# way how to register by function call
# register_node_now(OP_NODE_ADD, CalcNode_Add)