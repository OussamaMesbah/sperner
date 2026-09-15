"""An interactive figure: an SVG drawing with play controls and clickable parts.

The drawing is made in Python, like every other figure on the site. Parts of it can be
marked so that the figure animates and reacts in the browser:

* ``data-from="i"``: shown from step ``i`` on, such as the part of a walk already done;
* ``data-at="i"``: shown at step ``i`` only, such as the current cell of a walk;
* ``data-click="id"``: clickable; a click sends ``id`` back to Python.

With ``steps > 0`` the figure gets buttons to play, pause and step, and a scrubber, and
shows ``captions[i]`` under the drawing at step ``i``. Playing runs in the browser, so it
is smooth and costs no reruns.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st
from streamlit.runtime import Runtime

_CSS = """
.figure { font-family: var(--st-font, sans-serif); color: #222; }
.drawing svg { width: 100%; height: auto; display: block; }
.drawing [data-click] { cursor: pointer; }
.drawing [data-click]:hover { filter: brightness(1.18) drop-shadow(0 0 2px #111); }
.drawing [data-click]:focus {
  outline: none; filter: drop-shadow(0 0 3px #ff4b4b) drop-shadow(0 0 1px #ff4b4b);
}
.controls { display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.controls button {
  border: 1px solid #c8cdd3; background: #fff; border-radius: 8px; min-width: 38px;
  height: 34px; font-size: 15px; cursor: pointer; color: #222;
}
.controls button:hover { border-color: #ff4b4b; color: #ff4b4b; }
.controls button.play { background: #ff4b4b; border-color: #ff4b4b; color: #fff; min-width: 56px; }
.controls input[type=range] { flex: 1; min-width: 120px; accent-color: #ff4b4b; }
.controls .count { font-size: 13px; color: #555; min-width: 70px; text-align: right; }
.caption { font-size: 14px; color: #333; margin-top: 4px; min-height: 1.4em; }
.hint { font-size: 12.5px; color: #666; margin-top: 2px; }
"""

_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  // Streamlit calls this again when the data changes; stop the old figure's timer first.
  if (parentElement.__stopFigure) parentElement.__stopFigure();
  for (const old of parentElement.querySelectorAll('.figure')) old.remove();
  const figure = document.createElement('div');
  figure.className = 'figure';
  figure.setAttribute('role', 'group');
  figure.setAttribute('aria-label', data.description || 'figure');
  parentElement.appendChild(figure);

  const drawing = document.createElement('div');
  drawing.className = 'drawing';
  drawing.innerHTML = data.svg;
  figure.appendChild(drawing);

  const keyboard = data.keyboard !== false;
  const clickable = drawing.querySelectorAll('[data-click]');
  for (const part of clickable) {
    const send = () => setTriggerValue('click', part.getAttribute('data-click'));
    part.addEventListener('click', send);
    if (keyboard) {
      part.setAttribute('tabindex', '0');
      part.setAttribute('role', 'button');
      if (!part.getAttribute('aria-label')) {
        part.setAttribute('aria-label', part.getAttribute('data-click'));
      }
      part.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); send(); }
      });
    }
  }
  // A drawing with buttons in it is not a single image to screen readers.
  const root = drawing.querySelector('svg');
  if (root && keyboard && clickable.length) root.removeAttribute('role');
  if (data.hint) {
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = data.hint;
    figure.appendChild(hint);
  }

  const steps = data.steps || 0;
  const caption = document.createElement('div');
  caption.className = 'caption';
  caption.setAttribute('aria-live', 'polite');
  let timer = null;
  let step = Math.min(Math.max(data.start ?? steps, 0), steps);
  const timed = Array.from(drawing.querySelectorAll('[data-from], [data-at]'));

  function show() {
    for (const part of timed) {
      const from = part.getAttribute('data-from');
      const at = part.getAttribute('data-at');
      const visible = (from === null || step >= +from) && (at === null || step === +at);
      part.style.display = visible ? '' : 'none';
    }
    if (steps > 0) {
      slider.value = step;
      count.textContent = `${step} / ${steps}`;
      caption.textContent = (data.captions || [])[step] || '';
    }
  }

  const controls = document.createElement('div');
  controls.className = 'controls';
  const slider = document.createElement('input');
  const count = document.createElement('span');
  const play = document.createElement('button');

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    play.textContent = '▶';
    play.setAttribute('aria-label', 'Play');
  }
  function go(to) { step = Math.min(Math.max(to, 0), steps); show(); }
  function button(text, label, action) {
    const b = document.createElement('button');
    b.textContent = text;
    b.setAttribute('aria-label', label);
    b.addEventListener('click', () => { stop(); action(); });
    controls.appendChild(b);
    return b;
  }

  if (steps > 0) {
    button('⏮', 'First step', () => go(0));
    button('◂', 'Step back', () => go(step - 1));
    play.className = 'play';
    play.addEventListener('click', () => {
      if (timer) { stop(); return; }
      if (step >= steps) go(0);
      play.textContent = '⏸';
      play.setAttribute('aria-label', 'Pause');
      const tick = () => { if (step >= steps) stop(); else go(step + 1); };
      timer = setInterval(tick, data.interval || 350);
    });
    controls.appendChild(play);
    button('▸', 'Step forward', () => go(step + 1));
    button('⏭', 'Last step', () => go(steps));
    slider.type = 'range';
    slider.min = 0;
    slider.max = steps;
    slider.setAttribute('aria-label', 'Step');
    slider.addEventListener('input', () => { stop(); go(+slider.value); });
    controls.appendChild(slider);
    count.className = 'count';
    controls.appendChild(count);
    figure.appendChild(controls);
    figure.appendChild(caption);
    stop();
  }
  show();
  if (data.autoplay && steps > 0) play.click();
  parentElement.__stopFigure = () => { if (timer) clearInterval(timer); timer = null; };
  return parentElement.__stopFigure;
}
"""

_registered: dict[str, object] = {}


def _component():
    """The mounting command, registered with the current Streamlit runtime.

    Components are registered per runtime, and a module is imported only once per
    process, so the component is registered again whenever the runtime changes, as it
    does for every test app.
    """
    runtime = Runtime.instance() if Runtime.exists() else None
    if _registered.get("runtime") is not runtime or "mount" not in _registered:
        _registered["runtime"] = runtime
        _registered["mount"] = st.components.v2.component("sperner_figure", css=_CSS, js=_JS)
    return _registered["mount"]


def figure(
    svg: str,
    *,
    key: str,
    description: str,
    steps: int = 0,
    captions: Sequence[str] = (),
    start: int | None = None,
    hint: str = "",
    interval: int = 350,
    autoplay: bool = False,
    keyboard: bool = True,
) -> str | None:
    """Show an interactive SVG drawing; return the ``data-click`` id just clicked, if any.

    Args:
        svg: The drawing, with parts marked by ``data-from``, ``data-at`` and
            ``data-click`` (see the module docstring).
        key: A key that is unique on the page.
        description: What the figure shows, for screen readers.
        steps: The last step of the animation; 0 for a figure without one.
        captions: A caption for each step, from 0 to ``steps``.
        start: The step shown first; the last one if ``None``.
        hint: A line under the drawing, such as what clicking does.
        interval: Milliseconds per step when playing.
        autoplay: Whether to start playing at once.
        keyboard: Whether the clickable parts can also be reached with the keyboard and
            are announced as buttons. Turn it off for drawings with very many parts, and
            offer another way to choose.
    """
    result = _component()(
        key=key,
        data={
            "svg": svg,
            "description": description,
            "steps": steps,
            "captions": list(captions),
            "start": steps if start is None else start,
            "hint": hint,
            "interval": interval,
            "autoplay": autoplay,
            "keyboard": keyboard,
        },
        on_click_change=lambda: None,
    )
    return getattr(result, "click", None)
