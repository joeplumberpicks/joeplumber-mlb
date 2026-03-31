import ipywidgets as widgets
from IPython.display import display
from pathlib import Path

text_area = widgets.Textarea(
    value="",
    placeholder="Paste the full FanGraphs lineup text here...",
    description="Paste:",
    layout=widgets.Layout(width="100%", height="400px"),
)

button = widgets.Button(description="Save pasted text")
output = widgets.Output()

SAVE_PATH = Path("/content/fangraphs_pasted_lineups.txt")

def on_click(_):
    SAVE_PATH.write_text(text_area.value, encoding="utf-8")
    with output:
        output.clear_output()
        print(f"Saved pasted text to: {SAVE_PATH}")

button.on_click(on_click)

display(text_area, button, output)
