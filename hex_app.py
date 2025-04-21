import numpy as np
import base64
from io import BytesIO

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Range1d,
    Select, Toggle, Button, Div, FileInput
)
from bokeh.layouts import column, row

from catan_field import CatanFieldState, COLOR_CODES, FieldType

# --- Helpers -----------------------------------------------------

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

def get_color_for_state(state_str):
    if state_str is None:
        tup = COLOR_CODES["NONE"]
    else:
        tup = COLOR_CODES[FieldType(state_str)]
    return rgb_to_hex(tup)

# --- Grid setup -------------------------------------------------

NUM_ROWS = 7
NUM_COLS = 9

def row_length(r):
    return NUM_COLS if (r % 2) == 0 else max(0, NUM_COLS - 1)

data = {'row': [], 'col': [], 'q': [], 'r': [], 'state': [], 'color': []}
for r in range(NUM_ROWS):
    length = row_length(r)
    for c in range(length):
        data['row'].append(r)
        data['col'].append(c)
        data['q'].append(c + ((r % 2) - r) / 2)
        data['r'].append(r)
        data['state'].append(None)
        data['color'].append(get_color_for_state(None))

source = ColumnDataSource(data)

# --- Plot sizing -----------------------------------------------

hex_w = np.sqrt(3)
hex_h = 1.5
grid_w = hex_w * NUM_COLS
grid_h = hex_h * NUM_ROWS
plot_w = 800
plot_h = int(plot_w * (grid_h / grid_w))

p = figure(
    title="Interactive Catan Hex Grid",
    tools="tap,reset",
    match_aspect=True,
    background_fill_color="#fafafa",
    width=plot_w,
    height=plot_h,
)
p.hex_tile(
    q="q", r="r", size=1, orientation="pointytop",
    line_color="black", fill_color="color",
    source=source,
    nonselection_fill_alpha=1,
    nonselection_line_alpha=1,
    selection_line_color="black",
    selection_line_width=4,
)

max_r = NUM_ROWS - 1
p.y_range = Range1d(start=-(max_r * 1.5) - 1, end=1)
max_q = max(data['q'])
p.x_range = Range1d(start=-1, end=max_q * np.sqrt(3) + np.sqrt(3) + 1)

# --- Selection info + dropdown + brush toggle ------------------

selected_info = Div(text="Selected: None")

select_options = [("NONE", None)] + [(ft.name, ft.value) for ft in FieldType]
labels = [label for label, _ in select_options]

state_select = Select(title="Type:", value="NONE", options=labels)
brush_toggle = Toggle(label="Brush Mode", active=False)

def on_selection_change(attr, old, new_inds):
    if not new_inds:
        selected_info.text = "Selected: None"
        if not brush_toggle.active:
            state_select.value = "NONE"
        return

    i = new_inds[0]
    selected_info.text = f"Selected: row={data['row'][i]}, col={data['col'][i]}"

    if brush_toggle.active:
        label = state_select.value
        val = next(val for lbl, val in select_options if lbl == label)
        data['state'][i] = val
        source.patch({'state': [(i, val)], 'color': [(i, get_color_for_state(val))]})
    else:
        st = data['state'][i]
        state_select.value = FieldType(st).name if st is not None else "NONE"

source.selected.on_change('indices', on_selection_change)

def on_state_change(attr, old, new_label):
    if brush_toggle.active:
        return
    val = next(val for lbl, val in select_options if lbl == new_label)
    inds = source.selected.indices
    if not inds:
        return
    i = inds[0]
    data['state'][i] = val
    source.patch({'state': [(i, val)], 'color': [(i, get_color_for_state(val))]})

state_select.on_change('value', on_state_change)

# --- Export button ----------------------------------------------

export_button = Button(label="Export Grid State")
export_div = Div(text="")

def on_export():
    grid = np.full((NUM_ROWS, NUM_COLS), "", dtype='<U10')
    for i in range(len(data['row'])):
        r, c = data['row'][i], data['col'][i]
        v = data['state'][i]
        grid[r, c] = v if v is not None else ""
    buf = BytesIO()
    np.save(buf, grid); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    export_div.text = (
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="catan_grid_state.npy">Download Grid State</a>'
    )

export_button.on_click(on_export)

# --- File upload widget -----------------------------------------

file_input = FileInput(accept=".npy", multiple=False, title="Upload Grid State")

def on_upload(attr, old, new_b64):
    # new_b64 is the base64-encoded file contents
    decoded = base64.b64decode(new_b64)
    buf = BytesIO(decoded)
    grid = np.load(buf, allow_pickle=False)
    # update source.data arrays
    patches_state = []
    patches_color = []
    for i in range(len(data['row'])):
        r, c = data['row'][i], data['col'][i]
        val = grid[r, c] if grid[r, c] != "" else None
        if data['state'][i] != val:
            patches_state.append((i, val))
            patches_color.append((i, get_color_for_state(val)))
            data['state'][i] = val
    source.patch({'state': patches_state, 'color': patches_color})

file_input.on_change('value', on_upload)

# --- Layout -----------------------------------------------------

controls = row(state_select, brush_toggle, export_button, file_input)
curdoc().add_root(
    column(
        p,
        controls,
        selected_info,
        export_div,
        sizing_mode="stretch_width",
    )
)
