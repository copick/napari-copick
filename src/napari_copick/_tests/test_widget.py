from types import SimpleNamespace

from napari_copick.widget import CLI_AVAILABLE, GALLERY_AVAILABLE, INFO_AVAILABLE, CopickPlugin


def test_plugin_exposes_shared_ui_tabs(qtbot):
    viewer = SimpleNamespace(
        theme="dark",
        events=SimpleNamespace(theme=SimpleNamespace(connect=lambda callback: None)),
    )
    widget = CopickPlugin(viewer)
    qtbot.addWidget(widget)

    assert GALLERY_AVAILABLE is True
    assert INFO_AVAILABLE is True
    assert CLI_AVAILABLE is True
    assert [widget.tab_widget.tabText(index) for index in range(widget.tab_widget.count())] == [
        "🌲 Tree View",
        "📸 Gallery View",
        "📋 Info View",
        "🔧 Tools",
    ]
    assert widget.gallery_widget is not None
    assert widget.info_widget is not None
    assert widget.cli_widget is not None
