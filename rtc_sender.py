import asyncio
import fractions
import json
import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from websockets import connect
import time
from av import VideoFrame
class RandomImageTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.height = 480
        self.width = 640
        self.timestamp = 0

    async def recv(self):
        image = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self.timestamp
        frame.time_base = fractions.Fraction(1, 30)  # Assuming 30 FPS
        self.timestamp += 3000  # Increment the timestamp by 3000 for 30 FPS

        await asyncio.sleep(1 / 30)  # Sleep for 1/30 seconds to simulate 30 FPS
        return frame


async def send_receive_random_images(uri):
    pc = RTCPeerConnection()
    done = asyncio.Event()

    @pc.on("track")
    async def on_track(track):
        print("Track %s received" % track.kind)

        frame_counter = 0
        while True:
            frame = await track.recv()
            image = frame.to_ndarray(format="bgr24")
            frame_counter += 1
            print(f"Received frame {frame_counter}")  # Add this line to print a frame counter
            cv2.imshow("Received Image", image)
            cv2.waitKey(1)
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed":
            done.set()

    local_video = RandomImageTrack()
    pc.addTrack(local_video)

    while not done.is_set():
        try:
            async with connect(uri) as websocket:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)

                await websocket.send(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
                result_str = await websocket.recv()
                result = json.loads(result_str)
                print(result)
                await pc.setRemoteDescription(RTCSessionDescription(**result))

                # Wait for the connection to close
                await done.wait()
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    uri = "ws://localhost:80/ws"
    asyncio.get_event_loop().run_until_complete(send_receive_random_images(uri))