from qtpy import QtWidgets

from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.nodes.deforum_nodes.deforum_ui_data import deforum_base_params, deforum_anim_params, \
    deforum_translation_params, deforum_cadence_params, deforum_masking_params, deforum_depth_params, deforum_noise_params, \
    deforum_color_coherence_params, deforum_diffusion_schedule_params, deforum_hybrid_video_params, deforum_video_init_params, \
    deforum_image_init_params, deforum_hybrid_video_schedules


class DeforumBaseWidget(QDMNodeContentWidget):
    params = None
    def initUI(self):
        self.createUI(self.params)
        self.create_main_layout(grid=1)
    def createUI(self, params):
        for key, value in params.items():
            t = value["type"]
            if t == "dropdown":
                self.create_combo_box(value["choices"], label_text=key, object_name=f"{key}_value_combobox", accessible_name=key)
            elif t == "checkbox":
                self.create_check_box(key, accessible_name=key, object_name=f"{key}_value_checkbox", checked=value['default'])
            elif t == "lineedit":
                self.create_line_edit(key, accessible_name=key, object_name=f"{key}_value_lineedit", default=value['default'], schedule=True)
            elif t == "spinbox":
                self.create_spin_box(key, int(value["min"]), int(value["max"]), int(value["default"]), int(value["step"]), accessible_name=f"{key}_value_spinbox")
            elif t == "doublespinbox":
                self.create_double_spin_box(key, int(value["min"]), int(value["max"]), int(value["step"]), int(value["default"]), accessible_name=f"{key}_value_doublespinbox")

    def get_values(self):
        values = {}
        for widget in self.widget_list:
            if isinstance(widget, QtWidgets.QWidget):
                name = widget.objectName()
                acc_name = widget.accessibleName()
                real_name = None
                if "_value_" in name:
                    real_name = name
                    acc_name = acc_name
                elif "_value_" in acc_name:
                    real_name = acc_name
                    acc_name = name
                if real_name is not None:
                    if "_combobox" in real_name:
                        values[acc_name] = widget.currentText()
                    elif "_lineedit" in real_name:
                        values[acc_name] = widget.text()
                    elif "_checkbox" in real_name:
                        values[acc_name] = widget.isChecked()
                    elif "_spinbox" in real_name or "_doublespinbox" in real_name:
                        values[acc_name] = widget.value()
        return values




class DeforumBaseParamsWidget(DeforumBaseWidget):
    params = deforum_base_params

class DeforumAnimParamsWidget(DeforumBaseWidget):
    params = deforum_anim_params

class DeforumTranslationScheduleWidget(DeforumBaseWidget):
    params = deforum_translation_params

class DeforumCadenceParamsWidget(DeforumBaseWidget):
    params = deforum_cadence_params

class DeforumMaskingParamsWidget(DeforumBaseWidget):
    params = deforum_masking_params

class DeforumDepthParamsWidget(DeforumBaseWidget):
    params = deforum_depth_params

class DeforumNoiseParamsWidget(DeforumBaseWidget):
    params = deforum_noise_params

class DeforumColorParamsWidget(DeforumBaseWidget):
    params = deforum_color_coherence_params

class DeforumDiffusionParamsWidget(DeforumBaseWidget):
    params = deforum_diffusion_schedule_params

class DeforumHybridParamsWidget(DeforumBaseWidget):
    params = deforum_hybrid_video_params

class DeforumVideoInitParamsWidget(DeforumBaseWidget):
    params = deforum_video_init_params

class DeforumImageInitParamsWidget(DeforumBaseWidget):
    params = deforum_image_init_params

class DeforumHybridScheduleParamsWidget(DeforumBaseWidget):
    params = deforum_hybrid_video_schedules


