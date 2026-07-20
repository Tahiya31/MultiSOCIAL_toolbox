# MultiSOCIAL Toolbox — UI Design System

Canonical reference for GUI work in this project. Implementation lives in `src/gui_utils.py` (tokens) and `src/ui_components.py` (components).

## Palette

The app uses a refined **teal → forest green** gradient background with semi-transparent cards and owner-drawn controls.

| Token | Value | Usage |
|-------|-------|-------|
| `COLOR_BG_GRADIENT_START` | `#004D40` | Gradient top (deep teal) |
| `COLOR_BG_GRADIENT_END` | `#1B5E20` | Gradient bottom (deep forest) |
| `COLOR_BG_GLOW` | `(76, 175, 80, 35)` | Radial ambient highlight at top |
| `COLOR_SURFACE` | `(22, 32, 30, 215)` | Section card fill |
| `COLOR_SURFACE_ELEVATED` | `(28, 42, 38, 230)` | Elevated cards (footer, data) |
| `COLOR_INPUT_BG` | `(14, 22, 20)` | Native SpinCtrl / DirPicker on cards |
| `COLOR_GLASS_BORDER` / `COLOR_SURFACE_BORDER` | `(255, 255, 255, 28)` | Card borders |
| `COLOR_BORDER_HIGHLIGHT` | `(255, 255, 255, 55)` | Top-edge card highlight |
| `COLOR_PRIMARY` | `#4CAF50` | Primary action buttons |
| `COLOR_PRIMARY_HOVER` | `#66BB6A` | Primary hover |
| `COLOR_PRIMARY_PRESSED` | `#388E3C` | Primary pressed |
| `COLOR_SECONDARY` | `(255, 255, 255, 18)` | Ghost / outline buttons |
| `COLOR_SECONDARY_BORDER` | `(255, 255, 255, 45)` | Secondary outline |
| `COLOR_DANGER` | `#E53935` | Cancel button |
| `COLOR_TAB_TRACK` / `COLOR_TAB_ACTIVE` | dark inset / light pill | Segmented mode control |
| `COLOR_ACCENT_GREEN` | `#A5D6A7` | Logo ring, tab hover |
| `COLOR_PROGRESS_FILL` | `(102, 187, 106)` | Progress bar fill (gradient) |
| `COLOR_PROGRESS_TRACK` | `(0, 0, 0, 90)` | Progress bar track |
| `COLOR_TEXT_WHITE` | `#FFFFFF` | Labels on gradient / cards |
| `COLOR_TEXT_MUTED` | `#90A4AE` | Overlines, subtitles, de-emphasized |
| `COLOR_DISABLED` | `(120, 144, 156, 180)` | Disabled controls |
| `COLOR_TEXT_ON_DARK` | `#FFFFFF` | Button labels |
| `COLOR_TOOLTIP_BG` / `COLOR_TOOLTIP_FG` | `#1E1E1E` / `#F5F5F5` | Tooltips |

Legacy aliases (`COLOR_PROGRESS_GREEN`, `COLOR_TEXT_BLACK`, `COLOR_INFO_ICON_BG`) remain for backward compatibility.

## Typography

Use `Theme.get_font(size, bold=False)` everywhere.

A deliberate ramp (no two adjacent roles within 1 pt) keeps hierarchy legible.

| Token | Size (pt) | Usage |
|-------|-----------|-------|
| `FONT_DISPLAY` | 32 | App name ("MultiSOCIAL Toolbox") |
| `FONT_TITLE` | 17 | Section / screen titles |
| `FONT_BUTTON` | 16 | Action buttons |
| `FONT_HEADING` | 15 | Primary settings labels |
| `FONT_BODY` | 14 | Labels, spin controls, status |
| `FONT_SUBTITLE` | 13 | Header subtitle |
| `FONT_CAPTION` | 12 | Secondary / diarization status sublabels |
| `FONT_OVERLINE` | 11 | Uppercase card eyebrow headings |

**Platform fonts:** Segoe UI (Windows), SF Pro Text (macOS), Swiss (Linux).

**Resize scaling is capped at 1.25×** (`on_resize`) so typography stays consistent
across window sizes — large windows are filled by the responsive layout, not by
ballooning fonts (which also risked overflowing fixed-width cards).

Bold: headings, status line, card titles, tab labels.

## Spacing and radius

| Token | px | Usage |
|-------|-----|-------|
| `SPACE_XS` | 4 | Tight gaps |
| `SPACE_SM` | 8 | Inline control spacing |
| `SPACE_MD` | 12 | Card padding, button margins |
| `SPACE_LG` | 16 | Section gaps |
| `SPACE_XL` | 24 | Section margins |
| `SPACE_XXL` | 32 | Header breathing room |

| Token | px | Usage |
|-------|-----|-------|
| `RADIUS_BUTTON` | 10 | FlatButton corners |
| `RADIUS_CARD` | 14 | SectionCard |
| `RADIUS_TAB` | 12 | Segmented mode control |

Use `FromDIP()` for radii and min heights so scaling respects display density.

## Layout structure

```
GradientPanel (scrollable root)
├── Header — logo + title + subtitle (horizontal, full width)
├── ToggleTabBar — segmented Video | Audio          ┐
├── SectionCard "DATA FOLDER" — caption + DirPickerCtrl │ centered content column,
├── videoPanel / audioPanel (chrome-less grouping)      │ all blocks share one width
│   ├── SectionCard "SETTINGS"                           │ (see Responsive scaling)
│   └── SectionCard "ACTIONS"                            │
└── SectionCard "STATUS"                              ┘
    ├── statusLabel · CustomGauge (progress) · cancelBtn (danger, hidden when idle)
```

**Mode panels** are toggled (not `wx.Notebook`). Only one of `videoPanel` / `audioPanel` is visible.

**Responsive panel layout.** Inside each mode panel, `SETTINGS` and `ACTIONS`
stack vertically on narrow windows and sit **side-by-side** at/above the
breakpoint (`width ≥ FromDIP(860)`), where the content column also widens to
~90% of the frame. This removes the large side margins / vertical scrolling on
desktop-sized windows. Orientation is rebuilt (fresh `BoxSizer`, not
`SetOrientation`) only when the mode flips — see `_apply_panel_orientation`.

**Content alignment.** The **header bar**, `ToggleTabBar`, `DATA FOLDER`, the mode
panels, and `STATUS` are all sized to the same width and centered (one shared
content column), so their left/right edges line up at every window size. The
header logo+title is wrapped in a chrome-less `GlassPanel` for this reason — do
not add it full-width. All centered blocks use `wx.ALIGN_CENTER_HORIZONTAL` (never
`wx.EXPAND`, which left-aligns within the margins and breaks the column).

**Settings rows** use a 2-column `FlexGridSizer` (label | control, growable label
column) so numeric inputs share one aligned baseline; toggles are full-width and
left-aligned. Avoid `wx.ALIGN_CENTER` for settings — it produces ragged edges.

**Action hierarchy.** Exactly one `primary` (filled green) action per panel — the
core step (video: *Extract Pose Features*; audio: *Extract Audio Features*).
Prerequisite, QA, and optional steps (Convert, Embed, Verify, Transcripts, Align)
use `secondary` (ghost outline). One filled CTA + outline siblings = clear path.

**Native controls kept:** `wx.SpinCtrl`, `wx.DirPickerCtrl` — placed inside `SectionCard` on solid surfaces.

## Components

### FlatButton

Owner-drawn rounded button with drop shadow and top highlight. Variants: `primary`, `secondary` (ghost outline), `danger`.

| State | Behavior |
|-------|----------|
| Normal | Base variant color |
| Hover | `*_HOVER` token |
| Pressed | `*_PRESSED` token |
| Disabled | `COLOR_DISABLED`, no click |

Emits `wx.EVT_BUTTON`. API: `SetLabel`, `GetLabel`, `Enable`, `SetFont`, `SetMinSize`.

### ToggleTabBar

Segmented pill control (Video | Audio). Active segment uses elevated fill; hover highlights inactive label. `set_selected('video'|'audio')`, `EnableTabs(bool)`. Font scaling via `video_tab` / `audio_tab` proxies.

### CustomCheckBox

Owner-drawn checkbox + label. `GetValue` / `SetValue`, `SetLabel`, emits `wx.EVT_CHECKBOX`.

### SectionCard

`GlassPanel` subclass with uppercase overline heading, divider, and `SPACE_LG` padding. Uses `COLOR_SURFACE_ELEVATED`. `content_sizer` for children.

Native inputs on cards: call `gui_utils.style_native_input()` for `SpinCtrl` / `DirPickerCtrl`.

### CustomGauge

Rounded progress track with gradient fill and top shine. Default height 32 DIP.

`SetValue` no-ops when the clamped value is unchanged (avoids needless `Refresh` during dense worker progress callbacks). Prefer `Refresh()` over `Update()` when the value does change — do not force synchronous paints from the worker-driven UI path.

### TooltipButton / InfoIcon

Horizontal row: expanding `FlatButton` + `InfoIcon`.

Action tooltips use compact, scannable sections (typically `CREATES` / `REQUIRES` / `SAVES TO` / `USE NEXT`) rather than long prose. Keep lists short; avoid packing every edge case into the tip.

Settings modifiers that affect an action (for example **Add captions to pose-embedded video**) should use the **same adjacent InfoIcon pattern** as action buttons — put details on `InfoIcon`, not native `SetToolTip` on the checkbox. That keeps hover look and wrap behavior consistent with the rest of the panel.

`InfoIcon` / `CustomTooltip` wrap width is intentionally a bit wider (~380 DIP) so short section lists stay readable. `InfoIcon` constructs **one** reusable `wx.Timer` in `__init__` and rebinds it for leave/re-entry hide cycles. Do not allocate a new timer inside `_schedule_hide`.

### ElevatedLogoPanel

Caches circular-masked bitmaps by diameter (`_circular_bitmap_cache`) so normal ↔ hover size changes do not rebuild the alpha mask every time.

### Status line (app.py)

Worker threads may emit hundreds of status updates per second. `set_status_message` keeps only the latest pending string and schedules a single `_flush_status_message` on the GUI event loop. Wrap/center work is similarly coalesced; parent `Refresh()` is preferred over full-frame `Layout()` when only the status text changed.

### ResponsiveText

Entry point: `gui_utils.create_transparent_text()` — `TransparentStaticText` on Windows, `wx.StaticText` on macOS.

## Cross-platform rules

1. **Owner-draw interactive controls** — buttons and checkboxes must not use native `wx.Button` / `wx.CheckBox` on glass backgrounds.
2. **Never rely on native transparency** for interactive widgets; use `BG_STYLE_PAINT` + `GraphicsContext`.
3. **Fill the control background on Windows.** `BG_STYLE_PAINT` gives no real transparency on Windows, so an owner-drawn `wx.Window`'s rounded corners / shadow band would show an uninitialised background. Every owner-drawn control calls `_fill_windows_background(self, dc)` first (no-op on macOS), which fills with `gui_utils.composited_background_colour()` — the gradient sampled at the control's position, composited with any chrome-painting `GlassPanel` ancestor. This is why `FlatButton` / `CustomCheckBox` / `ToggleTabBar` / `CustomGauge` look identical on both platforms. **Keep effects (shadow, shine, glow, gradients) — they render consistently via GDI+; it's the *background* that diverges.**
4. **Single font-scaling registry** — register `(widget, base_size, bold)` at creation; scale in one `on_resize` loop (max 2× baseline). The loop is per-widget defensive (proxies without `IsShown` are allowed; one failure never aborts the pass).
5. **Windows gradient under cards** — keep `GlassPanel._paint_gradient_background_windows`; owner-drawn children sit on top.
6. **Adding a new control** — if it is clickable, owner-draw it **and call `_fill_windows_background` at the top of its paint**; if native (picker/spin), put it in a `SectionCard` + `style_native_input`; register fonts in `_scalable_widgets`.

## Cancel UX (#17)

| Phase | UI |
|-------|-----|
| Idle | Cancel hidden and disabled |
| Job started | `_begin_process()` — disable actions, tabs, folder picker; show Cancel |
| User clicks Cancel | Status: "Cancelling…"; Cancel disabled (no double-click) |
| Job ends (cancelled) | Status: "Processing cancelled."; no success dialog |
| Job ends (success) | Success `MessageBox`; status cleared on next action |

**Cancel granularity:**

- **Pose extract / embed:** cooperative check each frame (`cancel_check` in `pose.py` loops).
- **All other batch jobs:** stop at next file boundary (convert, verify, audio features, transcripts, align prep).

Diarization pip install is out of scope for Cancel.

## Responsive scaling

- Baseline size captured at first layout (`_baseline_size`).
- Scale factor: `min(width_ratio, height_ratio, 2.0)`, never below 1.0.
- **Content-column width** (`_update_panel_sizes`):
  - Two-column (`width ≥ FromDIP(860)`): `min(90% of frame, FromDIP(1180))`, floor `FromDIP(720)`.
  - Single-column: `min(70% of frame, FromDIP(640))`, floor `FromDIP(360)`.
- Tabs, folder card, mode panels, and status footer all use this width (centered).
- Progress bar height fixed at `FromDIP(34)`.
