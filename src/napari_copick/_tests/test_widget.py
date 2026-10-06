from types import SimpleNamespace

from napari_copick.widget import ANNOTATE_AVAILABLE, CLI_AVAILABLE, GALLERY_AVAILABLE, INFO_AVAILABLE, CopickPlugin


def _signal():
    return SimpleNamespace(connect=lambda callback: None)


def test_plugin_exposes_shared_ui_tabs(qtbot):
    viewer = SimpleNamespace(
        theme="dark",
        events=SimpleNamespace(theme=_signal()),
        layers=SimpleNamespace(
            selection=SimpleNamespace(active=None, events=SimpleNamespace(active=_signal())),
            events=SimpleNamespace(removed=_signal(), inserted=_signal()),
        ),
    )
    widget = CopickPlugin(viewer)
    qtbot.addWidget(widget)

    assert GALLERY_AVAILABLE is True
    assert INFO_AVAILABLE is True
    assert CLI_AVAILABLE is True
    assert ANNOTATE_AVAILABLE is True
    assert [widget.tab_widget.tabText(index) for index in range(widget.tab_widget.count())] == [
        "🌲 Tree View",
        "📸 Gallery View",
        "📋 Info View",
        "✏️ Annotate",
        "🔧 Tools",
    ]
    assert widget.annotate_widget is not None
    assert widget.gallery_widget is not None
    assert widget.info_widget is not None
    assert widget.cli_widget is not None
