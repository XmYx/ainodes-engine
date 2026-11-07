"""Run a saved graph with no visible window (issue #20).

aiNodes' node engine is built on ``QGraphicsScene``, so there is no way to
execute a graph without a ``QApplication``.  What there *is* is Qt's
``offscreen`` platform plugin: the whole app runs, nothing is shown, and no
display server is needed.  That is what ``--headless`` selects, which makes
``--headless --run graph.json`` usable over SSH and in containers.
"""
import os
import sys
import time

from qtpy.QtCore import QTimer


def log(message):
    """Headless output goes to the real stdout.

    The main window redirects ``sys.stdout`` into its console dock, which is
    invisible in headless mode - and the point of headless mode is to see the
    output in a terminal or a log file.
    """
    print(message, file=sys.__stdout__, flush=True)


def _editors(window):
    out = []
    for sub in window.mdiArea.subWindowList():
        widget = sub.widget()
        if widget is not None and hasattr(widget, "scene"):
            out.append(widget)
    return out


def _report(editor):
    """Say what each node produced, so a headless run is not a black box."""
    log("[headless] results:")
    for node in editor.scene.nodes:
        values = []
        for index in range(len(getattr(node, "outputs", []))):
            value = node.getOutput(index)
            if value is None:
                continue
            shape = getattr(getattr(value, "shape", None), "__iter__", None)
            if isinstance(value, dict) and "samples" in value:
                values.append(f"{node.output_socket_name[index]}="
                              f"latent{tuple(value['samples'].shape)}")
            elif shape is not None:
                values.append(f"{node.output_socket_name[index]}="
                              f"tensor{tuple(value.shape)}")
            else:
                values.append(f"{node.output_socket_name[index]}="
                              f"{type(value).__name__}")
        state = "invalid" if node.isInvalid() else ("dirty" if node.isDirty() else "ok")
        log(f"  {node.title}: {state}" + (f" -> {', '.join(values)}" if values else ""))


def run_headless(app, window, args):
    """Execute ``args.run`` ``args.run_loops`` times, then return an exit code."""
    graph = args.run
    if not graph:
        log("--headless needs --run GRAPH (there is nothing to execute otherwise)")
        return 2
    if not os.path.isfile(graph):
        log(f"graph not found: {graph}")
        return 2

    log(f"[headless] platform plugin: {os.environ.get('QT_QPA_PLATFORM', 'default')}")
    window.resize(1600, 900)   # the scene needs a sane viewport even offscreen
    window.fileOpen(graph)

    editors = _editors(window)
    if not editors:
        log(f"[headless] {graph} did not load into an editor")
        return 1
    editor = editors[-1]
    node_count = len(editor.scene.nodes)
    log(f"[headless] loaded {graph}: {node_count} node(s)")
    if not node_count:
        log("[headless] nothing to run")
        return 1

    state = {"runs_left": max(1, int(args.run_loops)), "code": 0,
             "started": time.time()}
    runner = editor.scene.noderunner

    def finish(code):
        state["code"] = code
        app.quit()

    def on_run_finished(_runner):
        state["runs_left"] -= 1
        elapsed = time.time() - state["started"]
        log(f"[headless] run finished after {elapsed:.1f}s "
            f"({state['runs_left']} left)")
        if state["runs_left"] <= 0:
            _report(editor)
            finish(0)
        else:
            QTimer.singleShot(0, start_run)

    runner.finished_callbacks.append(on_run_finished)

    def start_run():
        state["started"] = time.time()
        for node in editor.scene.nodes:
            node.markDirty(True)
        runner.start(loop=False)

    if args.timeout and args.timeout > 0:
        def on_timeout():
            log(f"[headless] timed out after {args.timeout}s")
            runner.stop()
            finish(124)

        QTimer.singleShot(int(args.timeout * 1000), on_timeout)

    QTimer.singleShot(0, start_run)
    app.exec()
    sys.stdout.flush()
    return state["code"]
