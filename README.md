<h1>aiNodes Engine</h1>

<img src="docs/main.png" alt="Main Image">

<p>aiNodes is a simple and easy-to-use Python-based AI image / motion picture generator node engine.</p>

<p>A desktop ai centered node engine with a live execution chain, python code editor node, and plug-in support, officially supported by Deforum.
We are thankful for many great functions adapted from ComfyUI, and various open-source projects, that make this framework possible to exist.
  
## 💖 Support the project
  
Please consider starring, or contributing to the project.
Updates are frequent, and the more time I can spare on the project, the greater things can be implemented. Thank you for being here!

Your support is greatly appreciated!

</p>

<h2>📚 Table of Contents</h2>

- [Introduction](#-intro)
- [Installation / Running the App](#-installation--running-the-app)
- [Contributing](#-contributing)
- [Features](#-features)
- [License](#-license)
<a name="-intro"></a>
<h2>🚀 Introduction</h2>

<ul>
  <li>Full modularity - download node packs on runtime</li>
  <li>Coloured background drop</li>
  <li>Easy node creation with IDE annotations</li>
</ul>
<a name="-installation--running-the-app"></a>
<h2>🔧 Install / Running the App</h2>

<p>To get started with aiNodes, follow the steps below:</p>

Requirements:
<ol>
  <li>Python 3.10 (https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)</li>
  <li>Git (https://github.com/git-for-windows/git/releases/download/v2.40.1.windows.1/Git-2.40.1-64-bit.exe)</li>
  <li>nVidia GPU with CUDA and drivers installed</li>
</ol>

 Windows:
<ol>
  <li>Download the 1 Click Installer from the releases on the right menu and run it in a folder of your choice</li>
  <li>It will create a virtual environment, and install all dependencies, to start next time, you can use the shortcut on your Desktop.</li>
</ol>

To update, you can run update.bat.


Linux:
```shell
git clone https://github.com/XmYx/ainodes-engine
cd ainodes-engine
bash ainodes.sh
```

MacOs:
```shell
support coming up
```
launch with:
```
source nodes_env/bin/activate
python main.py
```
Once the app is up and running, you can start check the File - Example Graphs option to start creating, and you can also access your model folders from the File menu.

<a name="-contributing"></a>
<h2>🤝 Contributing</h2>

<p>Contributions to the Ainodes Engine are welcome and appreciated. If you find any bugs or issues with the app, please feel free to open an issue or submit a pull request.</p>

<h2>🙌 Features</h2>
<a name="-features"></a>
<p>aiNodes is an open source desktop ai based image / motion generator, editor suite designed to be flexible, and with an Unreal-like execution chain. It natively supports:</p>

<ol>
  <li>Deforum</li>
  <li>Stable Diffusion 1.5 / 2.0 / 2.1</li>
  <li>Upscalers</li>
  <li>Kandinsky</li>
  <li>ControlNet</li>
  <li>LORAs</li>
  <li>Ti Embeddings</li>
  <li>Hypernetworks</li>
  <li>Background Separation</li>
  <li>Human matting / masking</li>
  <li>Compositing</li>
  <li>Drag and Drop (from discord too)</li>
  <li>Subgraphs</li>
  <li>Graph saving as metadata in the image file</li>
  <li>Graph loading from image metadata</li>
</ol>

This project came to life thanks to many great backend functions borrowed from ComfyUI, and adapted to work in this unique, 
live, controllable manner with a strict user declared execution chain, leading to data values possible to be iterated at
different points in time in your pipeline.

<a name="-license"></a>
<h2>📄 License</h2>

<p>This project is licensed under the L-GPL License. See the <a href="LICENSE">LICENSE</a> file for details.</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/XmYx/ainodes-engine" alt="Stars">
  <img src="https://img.shields.io/github/forks/XmYx/ainodes-engine" alt="Forks">
</p>
<pre>
<code>
Mention on <a href="https://vjun.io/vdmo/node-based-stable-diffusion-toolkits-2i1j">vjun.io</a>
HU Tutorial by <a href="https://youtu.be/QD7aMacE2f4">mp3pintyo</a>
</code>
</pre>
