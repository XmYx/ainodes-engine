"""Theme / palette handling.

One palette per theme drives *both* the Qt stylesheet and the colours the
node canvas paints with (``QGraphicsItem.paint`` cannot read a stylesheet, so
before this module the canvas was hardcoded dark and the "light" theme was a
dark canvas inside light chrome - see issue #28).

Adding a theme means adding an entry to :data:`PALETTES`; the stylesheet is
rendered from ``ainodes_frontend/qss/theme.qss.tmpl`` with ``@token@``
placeholders substituted from the palette.
"""
import os

from qtpy.QtGui import QColor

from ainodes_frontend import singleton as gs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QSS_DIR = os.path.join(HERE, "qss")
TEMPLATE = os.path.join(QSS_DIR, "theme.qss.tmpl")

DEFAULT_THEME = "dark"

PALETTES = {
    "dark": {
        # window chrome
        "window": "#1e1f22",
        "surface": "#2b2d30",
        "surface_alt": "#323438",
        "surface_hover": "#3a3d42",
        "text": "#e8e8ea",
        "text_dim": "#a0a3a8",
        "text_disabled": "#6b6e74",
        "border": "#44474d",
        "accent": "#ff8c00",
        "accent_hover": "#ffa333",
        "accent_text": "#1a1a1a",
        "selection": "#3a5c86",
        "selection_text": "#ffffff",
        # editable fields - the ones that were unreadable
        "field": "#1a1b1e",
        "field_text": "#f2f2f4",
        "field_border": "#4e525a",
        "field_placeholder": "#8b8f96",
        # node canvas
        "scene_bg": "#232528",
        "grid_light": "#2b2e32",
        "grid_dark": "#1b1d20",
        "scene_text": "#cccccc",
        "node_bg": "#e3212428",
        "node_title_bg": "#ff2f3237",
        "node_title_text": "#f5f5f7",
        "node_outline": "#7f000000",
        "node_selected": "#ffffa637",
        "node_hovered": "#ff37a6ff",
        "node_dirty": "#ffff5555",
        "node_invalid": "#ffff0000",
        "edge": "#d0c8c8c8",
        "edge_selected": "#ffffa637",
        "edge_hovered": "#ff37a6ff",
        "socket_outline": "#ff000000",
        "console_bg": "#15161a",
        "console_text": "#dcdcdc",
    },
    "light": {
        "window": "#f3f3f5",
        "surface": "#ffffff",
        "surface_alt": "#e9e9ec",
        "surface_hover": "#dcdce1",
        "text": "#1f2023",
        "text_dim": "#5a5d63",
        "text_disabled": "#a2a5ab",
        "border": "#c4c6cc",
        "accent": "#d2691e",
        "accent_hover": "#e8873c",
        "accent_text": "#ffffff",
        "selection": "#b6d4f5",
        "selection_text": "#10131a",
        "field": "#ffffff",
        "field_text": "#16181c",
        "field_border": "#b3b6bd",
        "field_placeholder": "#80848c",
        "scene_bg": "#e4e5e9",
        "grid_light": "#d5d7dd",
        "grid_dark": "#c3c6cd",
        "scene_text": "#44474d",
        "node_bg": "#f0fafafc",
        "node_title_bg": "#ffdfe1e6",
        "node_title_text": "#1b1d21",
        "node_outline": "#7f20242c",
        "node_selected": "#ffc2620e",
        "node_hovered": "#ff1f7fd0",
        "node_dirty": "#ffd03030",
        "node_invalid": "#ffb00000",
        "edge": "#d0404449",
        "edge_selected": "#ffc2620e",
        "edge_hovered": "#ff1f7fd0",
        "socket_outline": "#ff2a2d33",
        "console_bg": "#fbfbfd",
        "console_text": "#1f2023",
    },
}

#: Socket colours, per data type index.  Shared by both themes but with a
#: lighter outline in light mode so small sockets stay visible.
SOCKET_COLORS = [
    "#FFFF7700",  # 0 exec
    "#FF52e220",  # 1
    "#FF0056a6",  # 2 latent
    "#FFa86db1",  # 3 conditioning
    "#FFb54747",  # 4 model
    "#FFdbe220",  # 5 image
    "#FF888888",  # 6 data
    "#FFaa55ff",  # 7 any
]

_current = None


def available_themes():
    return sorted(PALETTES)


def current_theme():
    return _current or getattr(gs, "theme", None) or DEFAULT_THEME


def palette(theme=None):
    """The colour dict for ``theme`` (falls back to the default theme)."""
    theme = (theme or current_theme() or DEFAULT_THEME).lower()
    return PALETTES.get(theme, PALETTES[DEFAULT_THEME])


def color(key, theme=None):
    """A ``QColor`` for a palette key.

    Keys may hold ``#rrggbb`` or ``#aarrggbb``; ``QColor`` understands both
    as long as we tell it which, so ``#aarrggbb`` is converted explicitly.
    """
    value = palette(theme).get(key)
    if value is None:
        return QColor("#ff00ff")  # loud magenta: a missing key should be obvious
    return to_qcolor(value)


def to_qcolor(value):
    if isinstance(value, QColor):
        return value
    text = str(value)
    if text.startswith("#") and len(text) == 9:
        alpha = int(text[1:3], 16)
        col = QColor(f"#{text[3:]}")
        col.setAlpha(alpha)
        return col
    return QColor(text)


def socket_color(index):
    colors = SOCKET_COLORS
    try:
        return to_qcolor(colors[int(index) % len(colors)])
    except Exception:
        return to_qcolor(colors[-1])


def is_dark(theme=None):
    bg = color("window", theme)
    return bg.lightness() < 128


def _render_template(theme):
    pal = palette(theme)
    if not os.path.isfile(TEMPLATE):
        return ""
    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        text = fh.read()
    for key, value in pal.items():
        # the canvas uses #aarrggbb, which Qt stylesheets do not accept
        css = value if len(value) != 9 else f"#{value[3:]}"
        text = text.replace(f"@{key}@", css)
    text = text.replace("@icons@", QSS_DIR.replace(os.sep, "/"))
    return text


def stylesheet(theme=None):
    """The full Qt stylesheet for ``theme``.

    A user-provided ``ainodes_frontend/qss/<theme>.qss`` is appended after the
    generated sheet, so custom themes can still override single rules.
    """
    theme = (theme or current_theme()).lower()
    sheet = _render_template(theme)
    override = os.path.join(QSS_DIR, f"{theme}.qss")
    if os.path.isfile(override):
        with open(override, "r", encoding="utf-8") as fh:
            sheet += "\n\n/* --- " + theme + ".qss overrides --- */\n" + fh.read()
    return sheet


def detect_system_theme():
    """Best effort "what is the desktop using" check.

    Qt 6.5+ reports this directly; otherwise fall back to the GTK theme name
    on Linux and the registry on Windows.  Anything unknown means dark.
    """
    try:
        from qtpy.QtGui import QGuiApplication
        from qtpy.QtCore import Qt

        app = QGuiApplication.instance()
        if app is not None and hasattr(app, "styleHints"):
            hints = app.styleHints()
            scheme = getattr(hints, "colorScheme", None)
            if callable(scheme):
                value = scheme()
                if value == Qt.ColorScheme.Light:
                    return "light"
                if value == Qt.ColorScheme.Dark:
                    return "dark"
    except Exception:
        pass

    import platform

    system = platform.system().lower()
    if system == "linux":
        name = (
            os.environ.get("GTK_THEME", "")
            or _gsettings("org.gnome.desktop.interface", "color-scheme")
            or _gsettings("org.gnome.desktop.interface", "gtk-theme")
        ).lower()
        if name:
            return "light" if "dark" not in name else "dark"
    elif system == "windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if value else "dark"
        except Exception:
            pass
    elif system == "darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=5,
            )
            return "dark" if "dark" in out.stdout.lower() else "light"
        except Exception:
            pass
    return DEFAULT_THEME


def _gsettings(schema, key):
    import subprocess

    try:
        out = subprocess.run(["gsettings", "get", schema, key],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().strip("'")
    except Exception:
        return ""


def apply_theme(theme=None, app=None, window=None):
    """Set the active theme and restyle everything that is already on screen."""
    global _current
    theme = (theme or DEFAULT_THEME).lower()
    if theme == "system":
        theme = detect_system_theme()
    if theme not in PALETTES:
        print(f"[theme] unknown theme '{theme}', using {DEFAULT_THEME}")
        theme = DEFAULT_THEME

    _current = theme
    gs.theme = theme
    gs.qss = f"qss/{theme}.qss"  # kept for code/configs that still read gs.qss

    from qtpy.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is not None:
        app.setStyleSheet(stylesheet(theme))

    if window is None and app is not None:
        for widget in app.topLevelWidgets():
            if hasattr(widget, "restyle_for_theme"):
                window = widget
                break
    if window is not None and hasattr(window, "restyle_for_theme"):
        window.restyle_for_theme(theme)

    _restyle_scenes(app)
    return theme


def _restyle_scenes(app):
    """Re-initialise the colours of every live canvas item."""
    if app is None:
        return
    from qtpy.QtWidgets import QGraphicsView

    seen = set()
    for widget in app.allWidgets():
        if not isinstance(widget, QGraphicsView):
            continue
        scene = widget.scene()
        if scene is None or id(scene) in seen:
            continue
        seen.add(id(scene))
        if hasattr(scene, "initAssets"):
            try:
                scene.initAssets()
                scene.setBackgroundBrush(scene._color_background)
            except Exception:
                pass
        for item in scene.items():
            if hasattr(item, "initAssets"):
                try:
                    item.initAssets()
                    if hasattr(item, "title_item") and hasattr(item, "_title_color"):
                        item.title_item.setDefaultTextColor(item._title_color)
                except Exception:
                    pass
            item.update()
        scene.update()
