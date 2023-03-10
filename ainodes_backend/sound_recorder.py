import sounddevice as sd
from pydub import AudioSegment
import wavio as wv
class AudioRecorder:
    def __init__(self):
        self.duration = 5
        self.sample_rate = int(sd.query_devices('default')['default_samplerate'])

        print(self.sample_rate)

        self.channels = 2

    def record(self):
        print(f"Recording {self.duration} seconds of audio...")
        frames = int(self.duration * self.sample_rate)
        self.recording = sd.rec(int(self.duration * self.sample_rate),
                           samplerate=self.sample_rate, channels=2)
        sd.wait()
        #write("recording0.wav", self.sample_rate, self.recording)

        # Convert the NumPy array to audio file
        wv.write("recording.wav", self.recording, self.sample_rate, sampwidth=2)
        #self.save_to_file()

    def save_to_file(self):

        filename = "audio.mp3"
        sound = AudioSegment(
            self.recording.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=self.recording.dtype.itemsize,
            channels=self.channels
        )
        sound.export(filename, format="mp3")
        print(f"Saved audio to {filename}.")
        print("Done recording.")
