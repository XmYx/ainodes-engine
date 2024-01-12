import datetime
import os

import pyqtgraph as pg

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSlider, QLabel
from qtpy.QtCore import QUrl

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
import torch
import typing as tp

from backend_helpers.torch_helpers.torch_gc import torch_gc

#from audiocraft.utils.autocast import TorchAutocast

#MANDATORY
OP_NODE_AUDIOCRAFT = get_next_opcode()


class MusicPlayer(QWidget):
    def __init__(self):
        super(MusicPlayer, self).__init__()
        from qtpy.QtMultimedia import QAudioOutput
        from qtpy.QtMultimedia import QMediaPlayer

        self.player = QMediaPlayer(self)
        self.audioOutput = QAudioOutput()
        self.audioOutput.setVolume(50)
        self.player.setAudioOutput(self.audioOutput)
        h_layout = QHBoxLayout()
        layout = QVBoxLayout()
        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.stop_button = QPushButton('Stop')

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.sliderMoved.connect(self.seek_position)

        self.time_label = QLabel("00:00")
        self.time_label.setAlignment(Qt.AlignCenter)

        self.play_button.clicked.connect(self.play_music)
        self.pause_button.clicked.connect(self.pause_music)
        self.stop_button.clicked.connect(self.stop_music)

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)

        h_layout.addWidget(self.play_button)
        h_layout.addWidget(self.pause_button)
        h_layout.addWidget(self.stop_button)
        layout.addLayout(h_layout)
        h_2_layout = QHBoxLayout()
        h_2_layout.addWidget(self.timeline)
        h_2_layout.addWidget(self.time_label)
        layout.addLayout(h_2_layout)
        layout.setContentsMargins(0,0,0,50)
        self.setLayout(layout)

    def set_media(self, path):
        print("SOURCE SET TO ", path)
        self.player.stop()

        self.player.setSource(QUrl())

        self.player.setSource(QUrl.fromLocalFile(path))

    def play_music(self):
        print("PLAY TRIGGERED")
        self.player.play()

    def pause_music(self):
        self.player.pause()

    def stop_music(self):
        self.player.stop()

    def seek_position(self, position):
        self.player.setPosition(position)

    def position_changed(self, position):
        self.timeline.setValue(position)
        # Update the time label
        self.time_label.setText(self.convert_milliseconds(position))

    def duration_changed(self, duration):
        self.timeline.setRange(0, duration)

    def convert_milliseconds(self, ms):
        seconds = (ms / 1000) % 60
        minutes = (ms / (1000 * 60)) % 60
        return f'{int(minutes):02d}:{int(seconds):02d}'
class AudiocraftWidget(QDMNodeContentWidget):
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.player = MusicPlayer()
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setFixedHeight(200)  # Set the desired height of the plot widget
        self.model_select = self.create_combo_box(["melody", "medium", "small", "large"], "Model Type")
        self.prompt = self.create_line_edit("Prompt", placeholder="Prompt")
        self.duration = self.create_spin_box("Duration (s)", min_val=1, max_val=600, default_val=30)
        self.topk = self.create_spin_box("Top K", min_val=0, max_val=5000, default_val=250)
        self.topp = self.create_spin_box("Top P", min_val=0, max_val=5000, default_val=0)
        self.temperature = self.create_double_spin_box("Temperature", min_val=0.0, max_val=10.0, default_val=1.0, step=0.01)
        self.cfg_scale = self.create_double_spin_box("CFG Scale", min_val=0.0, max_val=10.0, default_val=3.0, step=0.01)
        self.input_path = self.create_line_edit("Input Path:", placeholder="Leave empty for txt2audio, or use a path for audio2audio")
        self.create_main_layout(grid=1)
        self.progress_bar = self.create_progress_bar("Progress", min_val=0, max_val=100, default_val=0)

        self.grid_layout.addWidget(self.player,9,0)
        self.grid_layout.addWidget(self.audio_plot,10,0)
        self.grid_layout.addWidget(self.progress_bar,11,0)
        self.grid_layout.setContentsMargins(0,0,0,50)

    def set_progress(self, progress:int):
        self.progress_bar.setValue(progress)


#NODE CLASS
@register_node(OP_NODE_AUDIOCRAFT)
class AudioCraftNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/audiocraft.png"
    help_text = "Audiocraft - Music"
    op_code = OP_NODE_AUDIOCRAFT
    op_title = "Audiocraft Music Node"
    content_label_objname = "audiocraft_node"
    category = "base/audio"
    NodeContent_class = AudiocraftWidget
    dim = (400, 420)
    output_data_ports = [0]
    exec_port = 0

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])
        self.model = None
        self.samples = None
        self.have_samples = None
        self.loaded_model = None
        self.content.progress_signal.connect(self.content.set_progress)
        self.grNode.height = 700
        self.update_all_sockets()

    def evalImplementation_thread(self, index=0):
        data = self.getInputData(0)
        audio_path = None
        prompt = self.content.prompt.text()
        self.content.player.stop_button.click()
        hijack = None
        selected = self.content.model_select.currentText()
        if not self.model or self.loaded_model != selected:
            from audiocraft.models import MusicGen

            self.model = MusicGen.get_pretrained(selected)
            self.model.lm = self.model.lm.to("cpu")
            self.model.compression_model = self.model.compression_model.to("cpu")

            self.loaded_model = selected
            if hijack:
                self.model._generate_tokens = self.generate_tokens
        if data:
            if "prompt" in data:
                prompt = data["prompt"]
            else:
                data["prompt"] = prompt
        else:
            data = {"prompt":prompt}
        self.model.lm = self.model.lm.to("cuda")
        self.model.compression_model = self.model.compression_model.to("cuda")
        input_path = self.content.input_path.text()
        input_path = None if input_path == "" else audio_to_numpy(input_path)
        audio_path, samples = self.predict(text=prompt,
                                  melody=input_path,
                                  duration=self.content.duration.value(),
                                  topk=self.content.topk.value(),
                                  topp=self.content.topp.value(),
                                  temperature=self.content.temperature.value(),
                                  cfg_coef=self.content.cfg_scale.value())
        self.model.lm.to("cpu")
        self.model.compression_model.to("cpu")
        torch_gc()
        self.content.player.set_media(audio_path)
        return [data]

    def remove(self):
        del self.model
        torch_gc()
        super().remove()


    def predict(self, text, melody, duration, topk, topp, temperature, cfg_coef):
        from audiocraft.data.audio import audio_write

        self.model.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )

        #self.model.generation_params["max_gen_len"] = int(duration * self.model.frame_rate)
        if melody:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(self.model.device).float().t().unsqueeze(0)
            print(melody.shape)
            if melody.dim() == 2:
                melody = melody[None]
            melody = melody[..., :int(sr * self.model.lm.cfg.dataset.segment_duration)]
            output = self.model.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=sr,
                progress=True
            )
        else:
            output = self.model.generate(descriptions=[text], progress=True)

        output = output.detach().cpu().float()[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        output_dir = "output/WAVs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}.wav")
        with open(output_path, "wb") as file:
            audio_write(file.name, output, self.model.sample_rate, strategy="loudness", add_suffix=False)
        file.close()
        audio_np = output.squeeze().numpy()
        self.content.audio_plot.plot(audio_np)
        return output_path, output

    def callback(self, i, j):
        self.content.progress_signal.emit(int((100 / j) * i))


    def generate_tokens(self, attributes,
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            self.callback(generated_tokens, tokens_to_generate)
            #print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert self.model.generation_params['max_gen_len'] > prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        # generate by sampling from LM
        with self.model.autocast:
            gen_tokens = self.model.lm.generate(prompt_tokens, attributes, callback=callback, **self.model.generation_params)

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.model.compression_model.decode(gen_tokens, None)
        return gen_audio

def audio_to_numpy(path):
    # Load the audio file
    import librosa
    signal, sr = librosa.load(path, sr=None)

    return [sr, signal]