import datetime
import os
import subprocess
from multiprocessing import Process

import cv2
import imageio
import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread
from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QVBoxLayout

from ainodes_frontend.base.qimage_ops import pixmap_to_tensor, tensor2pil

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from tqdm import tqdm
OP_NODE_VIDEO_SAVE = get_next_opcode()

class VideoOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.video = GifRecorder()
        self.current_frame = 0
        self.type_select = self.create_combo_box(items=["GIF", "mp4_ffmpeg", "mp4_fourcc", "webm_ffmpeg", "webm_cv2"], label_text="Save Format")
        # self.type_select.addItems(["GIF", "mp4_ffmpeg", "mp4_fourcc"])
        self.save_button = QPushButton("Save buffer to selected type", self)
        #self.new_button = QPushButton("New Video", self)
        #self.save_button.clicked.connect(self.loadVideo)

        self.width_value = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=512, step=64)
        self.height_value = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=512, step=64)
        self.fps = self.create_spin_box("FPS", min_val=1, max_val=4096, default_val=30, step=1)
        # self.width_value = QtWidgets.QSpinBox()
        # self.width_value.setMinimum(64)
        # self.width_value.setSingleStep(64)
        # self.width_value.setMaximum(4096)
        # self.width_value.setValue(512)

        # self.height_value = QtWidgets.QSpinBox()
        # self.height_value.setMinimum(64)
        # self.height_value.setSingleStep(64)
        # self.height_value.setMaximum(4096)
        # self.height_value.setValue(512)

        # self.fps = QtWidgets.QSpinBox()
        # self.fps.setMinimum(1)
        # self.fps.setSingleStep(1)
        # self.fps.setMaximum(4096)
        # self.fps.setValue(24)

        self.dump_at = self.create_spin_box("Dump at every:", 0, 20000, 0, 1)

        self.audio_path = self.create_line_edit("Audio Path:")

        self.checkbox = self.create_check_box("Keep Buffer")
        # self.checkbox = QtWidgets.QCheckBox("Keep Buffer")
        #
        # palette = QtGui.QPalette()
        # palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        # palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        # self.checkbox.setPalette(palette)

        # layout = QVBoxLayout()
        # layout.addWidget(self.type_select)
        # layout.addWidget(self.save_button)
        # #layout.addWidget(self.width_value)
        # #layout.addWidget(self.height_value)
        # layout.addWidget(self.fps)
        # layout.addWidget(self.dump_at)
        # layout.addWidget(self.checkbox)
        #
        # self.setLayout(layout)
        self.create_button_layout([self.save_button])
        self.create_main_layout(grid=1)


@register_node(OP_NODE_VIDEO_SAVE)
class VideoOutputNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/video_save.png"
    op_code = OP_NODE_VIDEO_SAVE
    op_title = "Video Save"
    content_label_objname = "video_output_node"
    category = "base/video"
    input_socket_name = ["EXEC", "IMAGE"]
    output_socket_name = ["EXEC", "IMAGE"]

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.filename = ""
        # self.content.eval_signal.connect(self.evalImplementation)
        pass
    def initInnerClasses(self):
        self.content = VideoOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.content.save_button.clicked.connect(self.start_new_video)
        self.grNode.height = 330
        self.grNode.width = 280

        self.content.setGeometry(10, 20, 260, 230)
        self.markInvalid(True)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        self.busy = True
        tensors = []
        if self.getInput(0) is not None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            tensors = input_node.getOutput(other_index)

            if tensors is not None and gs.should_run:

                if tensors.shape[0] > 1:
                    for tensor in tensors:
                        image = tensor2pil(tensor.unsqueeze(0))
                        frame = np.array(image)
                        self.content.video.add_frame(frame, dump=self.content.dump_at.value())
                else:
                    image = tensor2pil(tensors)
                    frame = np.array(image)
                    self.content.video.add_frame(frame, dump=self.content.dump_at.value())
                print(f"[ Current frame buffer: {len(self.content.video.frames)} ]")
        return [tensors]

    def close(self):
        self.content.video.close(self.filename)
        self.markDirty(False)
        self.markInvalid(False)
        self.start_new_video()

        if not self.content.checkbox.isChecked() == True:
            self.content.video.frames = []

    def resize(self):
        self.content.setMinimumHeight(self.content.label.pixmap().size().height())
        self.content.setMinimumWidth(self.content.label.pixmap().size().width())
        self.grNode.height = self.content.label.pixmap().size().height() + 96
        self.grNode.width = self.content.label.pixmap().size().width() + 64

        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def start_new_video(self):
        self.markDirty(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = f"output/gifs/{timestamp}.gif"
        fps = self.content.fps.value()
        type = self.content.type_select.currentText()

        dump = not self.content.checkbox.isChecked()
        audio_path = self.content.audio_path.text()
        audio_path = None if audio_path == "" else audio_path
        self.content.video.close(timestamp, fps, type, dump, audio_path)
        if dump:
            print(f"[ VIDEO SAVE NODE: Done. The frame buffer is now empty. ]")
        else:
            print(f"[ VIDEO SAVE NODE: Done. The frame buffer still has {len(self.content.video.frames)} Frames. ]")

class VideoRecorder:

    def __init__(self):
        pass
    def start_recording(self, filename, fps, width, height):
        #fourcc = cv2.VideoWriter_fourcc(*'GIF')
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        #self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def add_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)


    def close(self, filename=""):
        self.video_writer.release()
        #os.rename("test.mp4", filename)
        print(f"[ Video saved as {filename} ]")


class GifRecorder:

    def __init__(self):
        self.frames = []

    def start_recording(self, filename, fps, width, height):
        self.filename = filename
        self.fps = fps

    def add_frame(self, frame, dump):
        self.frames.append(frame)
        if len(self.frames) >= dump and dump != 0:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fps = 24
            type = 'mp4_ffmpeg'
            self.close(timestamp, fps, type, True)


    def close(self, timestamp, fps, type='GIF', dump=False, audio_path=None):

        self.worker = SaveWorker(self.frames, timestamp, fps, type, audio_path)
        self.worker.start()

        if dump == True:
            self.frames = []

        # if type == 'GIF':
        #     os.makedirs("output/gifs", exist_ok=True)
        #     filename = f"output/gifs/{timestamp}.gif"
        #     if len(self.frames) > 0:
        #
        #         self.filename = filename
        #         self.fps = fps
        #         print(f"VIDEO SAVE NODE: Video saving {len(self.frames)} frames at {self.fps}fps as {self.filename}")
        #         #imageio.mimsave(self.filename, self.frames, duration=int(1000 * 1/self.fps), subrectangles=True, quantizer='nq-fs', palettesize=8)
        #         frames_to_save = [frame for frame in tqdm(self.frames, desc="Saving GIF")]
        #
        #         imageio.mimsave(self.filename, frames_to_save, duration=int(1000 * 1 / self.fps), palettesize=8)
        #         # pils = []
        #         # for frame in self.frames:
        #         #     pils.append(Image.fromarray(frame))
        #
        #         #pils[0].save("test_gif.gif", "GIF", save_all=True, duration=40, append_images=pils[1:], loop=1, quality=1)
        #
        #     else:
        #         print("The buffer is empty, cannot save.")
        #
        # elif type == 'mp4_ffmpeg':
        #     os.makedirs("output/mp4s", exist_ok=True)
        #     filename = f"output/mp4s/{timestamp}.mp4"
        #     if len(self.frames) > 0:
        #         width = self.frames[0].shape[1]
        #         height = self.frames[0].shape[0]
        #
        #         cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
        #                'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-an',
        #                filename]
        #         video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        #         for frame in tqdm(self.frames, desc="Saving MP4 (ffmpeg)"):
        #             video_writer.stdin.write(frame.tobytes())
        #         video_writer.communicate()
        #
        #         # if audio path is provided, merge the audio and the video
        #         if audio_path is not None:
        #             try:
        #                 output_filename = f"output/mp4s/{timestamp}_with_audio.mp4"
        #                 cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_filename]
        #                 subprocess.run(cmd)
        #             except Exception as e:
        #                 print(f"Audio file merge failed from path {audio_path}\n{repr(e)}")
        #                 pass
        #     else:
        #         print("The buffer is empty, cannot save.")
        #
        # # elif type == 'mp4_ffmpeg':
        # #     os.makedirs("output/mp4s", exist_ok=True)
        # #     filename = f"output/mp4s/{timestamp}.mp4"
        # #     if len(self.frames) > 0:
        # #         width = self.frames[0].shape[1]
        # #         height = self.frames[0].shape[0]
        # #
        # #         #print(width, height)
        # #         cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
        # #                'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-an',
        # #                filename]
        # #         video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # #         for frame in self.frames:
        # #             #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # #             video_writer.stdin.write(frame.tobytes())
        # #         video_writer.communicate()
        # #     else:
        # #         print("The buffer is empty, cannot save.")
        #
        # elif type == 'mp4_fourcc':
        #     os.makedirs("output/mp4s", exist_ok=True)
        #     filename = f"output/mp4s/{timestamp}.mp4"
        #     if len(self.frames) > 0:
        #
        #         width = self.frames[0].shape[0]
        #         height = self.frames[0].shape[1]
        #         video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        #         for frame in tqdm(self.frames, desc="Saving MP4 (fourcc)"):
        #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #             video_writer.write(frame)
        #         video_writer.release()
        #     else:
        #         print("The buffer is empty, cannot save.")
        # elif type == 'webm_ffmpeg':
        #     os.makedirs("output/webms", exist_ok=True)
        #     filename = f"output/webms/{timestamp}.webm"
        #     if len(self.frames) > 0:
        #         width = self.frames[0].shape[1]
        #         height = self.frames[0].shape[0]
        #         cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
        #                'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libvpx-vp9', '-b:v', '1M', '-c:a', 'libopus',
        #                '-strict', '-2',
        #                filename]
        #         video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        #         for frame in tqdm(self.frames, desc="Saving WebM (ffmpeg)"):
        #             video_writer.stdin.write(frame.tobytes())
        #         video_writer.communicate()
        #
        #         # if audio path is provided, merge the audio and the video
        #         if audio_path is not None:
        #             try:
        #                 output_filename = f"output/webms/{timestamp}_with_audio.webm"
        #                 cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'libopus',
        #                        '-strict', '-2', output_filename]
        #                 subprocess.run(cmd)
        #             except Exception as e:
        #                 print(f"Audio file merge failed from path {audio_path}\n{repr(e)}")
        #                 pass
        # elif type == 'webm_cv2':
        #     print("saving webm")
        #
        #     self.worker = SaveWebmWorker(self.frames, timestamp, fps)
        #     self.worker.start()
        #
        #     # os.makedirs("output/webms", exist_ok=True)
        #     # filename = f"output/webms/{timestamp}.webm"
        #     # if len(self.frames) > 0:
        #     #     width = self.frames[0].shape[1]
        #     #     height = self.frames[0].shape[0]
        #     #     video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'VP90'), fps, (width, height))
        #     #     for frame in tqdm(self.frames, desc="Saving WebM (cv2)"):
        #     #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     #         video_writer.write(frame)
        #     #     video_writer.release()
        #     # else:
        #     #     print("The buffer is empty, cannot save.")


        # if dump == True:
        #     self.frames = []


class SaveWebmWorker(QThread):

    def __init__(self, frames, timestamp, fps):
        super().__init__()
        self.frames = frames
        self.timestamp = timestamp
        self.fps = fps

    def run(self):
        print("saving webm")
        os.makedirs("output/webms", exist_ok=True)
        filename = f"output/webms/{self.timestamp}.webm"
        if len(self.frames) > 0:
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]
            video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'VP90'), self.fps, (width, height))
            for frame in tqdm(self.frames, desc="Saving WebM (cv2)"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            video_writer.release()
        else:
            print("The buffer is empty, cannot save.")



class SaveWorker(QThread):
    def __init__(self, frames, timestamp, fps, type, audio_path=None):
        super().__init__()
        self.frames = frames
        self.timestamp = timestamp
        self.fps = fps
        self.type = type
        self.audio_path = audio_path

    def run(self):
        if self.type == 'GIF':
            self.save_gif()

        elif self.type == 'mp4_ffmpeg':
            self.save_mp4_ffmpeg()

        elif self.type == 'mp4_fourcc':
            self.save_mp4_fourcc()

        elif self.type == 'webm_ffmpeg':
            self.save_webm_ffmpeg()

        elif self.type == 'webm_cv2':
            self.save_webm_cv2()

    def save_gif(self):
        os.makedirs("output/gifs", exist_ok=True)
        filename = f"output/gifs/{self.timestamp}.gif"
        if len(self.frames) > 0:
            frames_to_save = [frame for frame in tqdm(self.frames, desc="Saving GIF")]
            imageio.mimsave(filename, frames_to_save, duration=int(1000 * 1 / self.fps), palettesize=8)
        else:
            print("The buffer is empty, cannot save.")

    def save_mp4_ffmpeg(self):
        os.makedirs("output/mp4s", exist_ok=True)
        filename = f"output/mp4s/{self.timestamp}.mp4"
        if len(self.frames) > 0:
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]

            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
                   'rgb24', '-r', str(self.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-an',
                   filename]
            video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in tqdm(self.frames, desc="Saving MP4 (ffmpeg)"):
                video_writer.stdin.write(frame.tobytes())
            video_writer.communicate()

            # if audio path is provided, merge the audio and the video
            if self.audio_path is not None:
                try:
                    output_filename = f"output/mp4s/{self.timestamp}_with_audio.mp4"
                    cmd = ['ffmpeg', '-y', '-i', filename, '-i', self.audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict',
                           'experimental', output_filename]
                    subprocess.run(cmd)
                except Exception as e:
                    print(f"Audio file merge failed from path {self.audio_path}\n{repr(e)}")
                    pass
        else:
            print("The buffer is empty, cannot save.")

    def save_mp4_fourcc(self):
        os.makedirs("output/mp4s", exist_ok=True)
        filename = f"output/mp4s/{self.timestamp}.mp4"
        if len(self.frames) > 0:
            # Corrected the width and height extraction
            height = self.frames[0].shape[0]
            width = self.frames[0].shape[1]

            video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            for frame in tqdm(self.frames, desc="Saving MP4 (fourcc)"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            video_writer.release()
        else:
            print("[ The buffer is empty, cannot save. ]")

    def save_webm_ffmpeg(self):
        os.makedirs("output/webms", exist_ok=True)
        filename = f"output/webms/{self.timestamp}.webm"
        if len(self.frames) > 0:
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
                   'rgb24', '-r', str(self.fps), '-i', '-', '-c:v', 'libvpx-vp9', '-b:v', '1M', '-c:a', 'libopus',
                   '-strict', '-2',
                   filename]
            video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in tqdm(self.frames, desc="Saving WebM (ffmpeg)"):
                video_writer.stdin.write(frame.tobytes())
            video_writer.communicate()

            # if audio path is provided, merge the audio and the video
            if self.audio_path is not None:
                try:
                    output_filename = f"output/webms/{self.timestamp}_with_audio.webm"
                    cmd = ['ffmpeg', '-y', '-i', filename, '-i', self.audio_path, '-c:v', 'copy', '-c:a', 'libopus',
                           '-strict', '-2', output_filename]
                    subprocess.run(cmd)
                except Exception as e:
                    print(f"Audio file merge failed from path {self.audio_path}\n{repr(e)}")
                    pass

    def save_webm_cv2(self):
        os.makedirs("output/webms", exist_ok=True)
        filename = f"output/webms/{self.timestamp}.webm"
        if len(self.frames) > 0:
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]
            video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'VP90'), self.fps, (width, height))
            for frame in tqdm(self.frames, desc="Saving WebM (cv2)"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            video_writer.release()
        else:
            print("[ The buffer is empty, cannot save. ]")
