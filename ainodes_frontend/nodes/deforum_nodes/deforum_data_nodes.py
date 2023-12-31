# from qtpy.QtCore import QTimer, Qt
# from qtpy.QtGui import QPaintEvent
# from qtpy.QtWidgets import QVBoxLayout, QWidget
# from mpmath import sin
# from qtpy import QtCore# , Qt3DCore, Qt3DExtras

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
# from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.nodes.deforum_nodes.deforum_basenode import DeforumBaseParamsWidget, DeforumCadenceParamsWidget, DeforumHybridParamsWidget, \
    DeforumImageInitParamsWidget, DeforumHybridScheduleParamsWidget, DeforumAnimParamsWidget, DeforumTranslationScheduleWidget, \
    DeforumColorParamsWidget, DeforumDepthParamsWidget, DeforumNoiseParamsWidget, DeforumDiffusionParamsWidget, DeforumMaskingParamsWidget, \
    DeforumVideoInitParamsWidget

OP_NODE_DEFORUM_BASE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_CADENCE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_HYBRID_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_IMAGE_INIT_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_ANIM_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_TRANSLATION_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_COLOR_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_DEPTH_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_NOISE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_DIFFUSION_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_MASKING_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_VIDEO_INIT_PARAMS = get_next_opcode()

#OP_NODE_DEFORUM_MOTION = get_next_opcode()

"""class MotionWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # Initialize motion parameters
        self.zoom = 1.0
        self.angle = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.rotation = 0.0

        self.frame_number = 0

        # Set up the 3D view and scene
        self.view = Qt3DExtras.Qt3DWindow()
        self.scene = Qt3DCore.QEntity()

        # Set up the camera and transform component
        self.camera = Qt3DExtras.QFirstPersonCameraController(self.scene)
        self.camera.setAspectRatio(4 / 3)
        self.camera.setUpVector(Qt3DCore.QVector3D(0, 1, 0))
        self.camera.setViewCenter(Qt3DCore.QVector3D(0, 0, 0))

        self.transform = Qt3DCore.QTransform()
        self.scene.addComponent(self.transform)

        self.view.setRootEntity(self.scene)

        # Create a timer to update the animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)  # Adjust the interval as needed (in milliseconds)

        # Set up the layout and add the 3D view
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        # Create the point cloud entity
        self.point_cloud_entity = self.create_point_cloud()
        self.scene.addChildEntity(self.point_cloud_entity)

    def create_point_cloud(self):
        # Create a point cloud entity
        point_cloud_entity = Qt3DCore.QEntity(self.scene)

        # Create the point geometry
        point_geometry = Qt3DExtras.QSphereGeometry()
        point_geometry.setRadius(0.05)

        # Create the point material
        point_material = Qt3DExtras.QPhongMaterial()
        point_material.setDiffuse(Qt.green)

        # Create a transform component for the point cloud
        point_transform = Qt3DCore.QTransform()
        point_transform.setScale3D(Qt3DCore.QVector3D(1, 1, 1))
        point_transform.setTranslation(Qt3DCore.QVector3D(0, 0, 0))

        # Set the geometry and material to the point cloud entity
        point_mesh = Qt3DCore.QMesh(point_cloud_entity)
        point_mesh.setGeometry(point_geometry)
        point_mesh.setMaterial(point_material)
        point_mesh.addComponent(point_transform)

        return point_cloud_entity

    def update_animation(self):
        self.frame_number += 1

        # Update motion parameters based on frame number or keyframes
        self.zoom = 1.0025 + 0.002 * sin(1.25 * 3.14 * self.frame_number / 30)
        # Update other motion parameters in a similar fashion

        # Perform the necessary transformations based on motion parameters
        self.transform.setScale3D(Qt3DCore.QVector3D(self.zoom, self.zoom, self.zoom))
        self.transform.setRotationZ(self.angle)
        self.transform.setTranslation(Qt3DCore.QVector3D(self.translation_x, self.translation_y, 0))

        # Update the point cloud position
        point_transform = self.point_cloud_entity.componentsOfType(Qt3DCore.QTransform)[0]
        point_transform.setTranslation(Qt3DCore.QVector3D(self.translation_x, self.translation_y, 0))

    def paintEvent(self, event: QPaintEvent):
        # No need for painting in a 3D widget
        pass


class DeforumMotionWidget(QDMNodeContentWidget):
    params = None
    def initUI(self):
        self.createUI()
        self.create_main_layout(grid=1)
        self.main_layout.addWidget(self.widget)

    def createUI(self):
        self.widget = MotionWidget(self)"""


def merge_dicts(dict1, dict2):
    result_dict = dict1.copy()
    for key, value in dict2.items():
        if key in result_dict:
            result_dict[key] = value
        else:
            result_dict[key] = value
    return result_dict


class DeforumParamBaseNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = None
    op_title = "Deforum Args Node"
    content_label_objname = "deforum_args_node"
    category = "Deforum"

    w_value = 340
    h_value = 600

    make_dirty = True

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = self.content_class(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = self.w_value
        self.grNode.height = self.h_value
        self.content.setMinimumWidth(self.w_value)
        self.content.setMinimumHeight(self.h_value - 40)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        self.busy = True
        input_data = self.getInputData(0)
        data = self.content.get_values()

        new_data = self.getAllInputs()
        #print("new_data", new_data)

        if input_data is not None:
            data = merge_dicts(input_data, data)
        return [data]

    #@QtCore.Slot(object)

@register_node(OP_NODE_DEFORUM_BASE_PARAMS)
class DeforumBaseParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Base Params"
    content_label_objname = "deforum_base_params_node"
    op_code = OP_NODE_DEFORUM_BASE_PARAMS
    content_class = DeforumBaseParamsWidget
    h_value = 900
    w_value = 600



@register_node(OP_NODE_DEFORUM_CADENCE_PARAMS)
class DeforumCadenceParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Cadence Params"
    content_label_objname = "deforum_cadence_params_node"
    op_code = OP_NODE_DEFORUM_CADENCE_PARAMS
    content_class = DeforumCadenceParamsWidget
    w_value = 600
    h_value = 320


@register_node(OP_NODE_DEFORUM_HYBRID_PARAMS)
class DeforumHybridParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Hybrid Video Params"
    content_label_objname = "deforum_hybrid_params_node"
    op_code = OP_NODE_DEFORUM_HYBRID_PARAMS
    content_class = DeforumHybridParamsWidget
    w_value = 600




@register_node(OP_NODE_DEFORUM_IMAGE_INIT_PARAMS)
class DeforumImageInitParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Image Init Params"
    content_label_objname = "deforum_image_init_params_node"
    op_code = OP_NODE_DEFORUM_IMAGE_INIT_PARAMS
    content_class = DeforumImageInitParamsWidget
    h_value = 250


@register_node(OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS)
class DeforumHybridSchedNode(DeforumParamBaseNode):
    op_title = "Deforum Hybrid Schedule"
    content_label_objname = "deforum_hybrid_sched_node"
    op_code = OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS
    content_class = DeforumHybridScheduleParamsWidget
    h_value = 320
    w_value = 600


@register_node(OP_NODE_DEFORUM_ANIM_PARAMS)
class DeforumAnimParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Anim Params"
    content_label_objname = "deforum_anim_params_node"
    op_code = OP_NODE_DEFORUM_ANIM_PARAMS
    content_class = DeforumAnimParamsWidget
    h_value = 320
    w_value = 600


@register_node(OP_NODE_DEFORUM_TRANSLATION_PARAMS)
class DeforumTranslationNode(DeforumParamBaseNode):
    op_title = "Deforum Translation"
    content_label_objname = "deforum_translation_node"
    op_code = OP_NODE_DEFORUM_TRANSLATION_PARAMS
    content_class = DeforumTranslationScheduleWidget
    h_value = 450
    w_value = 600



@register_node(OP_NODE_DEFORUM_COLOR_PARAMS)
class DeforumColorParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Color Params"
    content_label_objname = "deforum_color_params_node"
    op_code = OP_NODE_DEFORUM_COLOR_PARAMS
    content_class = DeforumColorParamsWidget
    h_value = 300
    w_value = 600


@register_node(OP_NODE_DEFORUM_DEPTH_PARAMS)
class DeforumDepthParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Depth Params"
    content_label_objname = "deforum_depth_params_node"
    op_code = OP_NODE_DEFORUM_DEPTH_PARAMS
    content_class = DeforumDepthParamsWidget
    h_value = 325
    w_value = 600



@register_node(OP_NODE_DEFORUM_NOISE_PARAMS)
class DeforumNoiseParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Noise Params"
    content_label_objname = "deforum_noise_params_node"
    op_code = OP_NODE_DEFORUM_NOISE_PARAMS
    content_class = DeforumNoiseParamsWidget
    h_value = 420 + 65
    w_value = 600


@register_node(OP_NODE_DEFORUM_DIFFUSION_PARAMS)
class DeforumDiffusionParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Diffusion Params"
    content_label_objname = "deforum_diffusion_params_node"
    op_code = OP_NODE_DEFORUM_DIFFUSION_PARAMS
    content_class = DeforumDiffusionParamsWidget
    h_value = 440
    w_value = 600


@register_node(OP_NODE_DEFORUM_MASKING_PARAMS)
class DeforumMaskingParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Masking Params"
    content_label_objname = "deforum_masking_params_node"
    op_code = OP_NODE_DEFORUM_MASKING_PARAMS
    content_class = DeforumMaskingParamsWidget
    h_value = 420 + 100


@register_node(OP_NODE_DEFORUM_VIDEO_INIT_PARAMS)
class DeforumVideoInitParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Video Init Params"
    content_label_objname = "deforum_video_init_params_node"
    op_code = OP_NODE_DEFORUM_VIDEO_INIT_PARAMS
    content_class = DeforumVideoInitParamsWidget
    h_value = 420 - 100



