import bpy


def track_null_position():
    null_object = bpy.data.objects.get("NullObject")
    if null_object:
        x_values, y_values, z_values = [], [], []
        for frame in range(int(bpy.context.scene.frame_start), int(bpy.context.scene.frame_end) + 1):
            bpy.context.scene.frame_set(frame)
            position = null_object.location
            x_values.append(position.x)
            y_values.append(position.y)
            z_values.append(position.z)
        return x_values, y_values, z_values
    else:
        return [], [], []


def calculate_speed_values(x_values, y_values, z_values):
    speed_x, speed_y, speed_z = [], [], []
    for i in range(len(x_values)):
        if i == 0:
            speed_x.append(0)
            speed_y.append(0)
            speed_z.append(0)
        else:
            speed_x.append(x_values[i] - x_values[i - 1])
            speed_y.append(y_values[i] - y_values[i - 1])
            speed_z.append(z_values[i] - z_values[i - 1])
    return speed_x, speed_y, speed_z


def print_speed_lists():
    global speed_x, speed_y, speed_z
    print("Speed X:", speed_x)
    print("Speed Y:", speed_y)
    print("Speed Z:", speed_z)


def start_animation():
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    bpy.ops.screen.animation_play()


x_values, y_values, z_values = [], [], []
speed_x, speed_y, speed_z = [], [], []


def run_script():
    global x_values, y_values, z_values, speed_x, speed_y, speed_z

    x_values, y_values, z_values = track_null_position()
    speed_x, speed_y, speed_z = calculate_speed_values(x_values, y_values, z_values)
    start_animation()


def cancel_animation():
    bpy.ops.screen.animation_cancel()
    print_speed_lists()


class RunScriptOperator(bpy.types.Operator):
    bl_idname = "object.run_script_operator"
    bl_label = "Run Script"

    def execute(self, context):
        run_script()
        return {'FINISHED'}


class CancelAnimationOperator(bpy.types.Operator):
    bl_idname = "object.cancel_animation_operator"
    bl_label = "Cancel Animation"

    def execute(self, context):
        cancel_animation()
        return {'FINISHED'}


def register():
    bpy.utils.register_class(RunScriptOperator)
    bpy.utils.register_class(CancelAnimationOperator)


def unregister():
    bpy.utils.unregister_class(RunScriptOperator)
    bpy.utils.unregister_class(CancelAnimationOperator)


if __name__ == "__main__":
    register()

    bpy.ops.object.run_script_operator()
