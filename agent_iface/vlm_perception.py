"""Perception from images: a vision-language model reads a camera render.

Drop-in for PerceptionModel.  Nothing is injected here: `perceive` renders the
scene from the asked camera, the VLM says where each cube is, and that answer
becomes the scene report.  Its errors are the model's own, and they depend on
the viewpoint for physical reasons (the arm hides things; from the side the
three pads line up in depth).

What the camera system is given, and what it is not:
  * the pads carry a letter (A, B, C) drawn at their known position -- a
    calibrated camera knows where the fixed pads are;
  * whether the gripper holds a block comes from the fingers, as in the
    injected-error study, so a held block is never misreported;
  * before a picture is taken the empty hand moves up and back, out of the
    cameras' way (not while it holds a block);
  * everything else -- which cube is on which pad or on which cube -- is the VLM's.
"""
import base64
import copy
import hashlib
import io
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.tabletop import OBJECT_SIZE, ZONE_HALF, ZONES

from .perception import VIEWS, _reclear, _report

CAMS = {"front": dict(yaw=90, pitch=-25), "top": dict(yaw=90, pitch=-89), "side": dict(yaw=0, pitch=-25)}
W, H, DIST, FOV, TARGET = 896, 672, 0.55, 60.0, (0.0, 0.0, 0.0)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PROMPT = """This image is from the {view} camera of a robot workcell. Inside the dark frame on the table there are:
- three grey square pads. Each has a white letter drawn next to it: A, B and C. They are called zone_A, zone_B and zone_C.
- four small cubes: one red, one green, one blue, one yellow.
The grey robot arm may hide part of the scene.

For every cube, say where it is. Use exactly one of:
  "zone_A", "zone_B", "zone_C"   the cube rests on that pad
  "on_red_block", "on_green_block", "on_blue_block", "on_yellow_block"   the cube sits on top of that other cube
  "table"         the cube is on the table, on no pad and on no other cube
  "not_visible"   you cannot see this cube in the image
Look at each cube separately. A cube that is only near a pad, not on it, is "table"."""

PLACES = ["zone_A", "zone_B", "zone_C", "on_red_block", "on_green_block", "on_blue_block", "on_yellow_block",
          "table", "not_visible"]

# one question per cube instead of one for the whole scene
PROMPT_CUBE = """This image is from the {view} camera of a robot workcell. Inside the dark frame on the table there are:
- three grey square pads. Each has a white letter drawn next to it: A, B and C. They are called zone_A, zone_B and zone_C.
- four small cubes: one red, one green, one blue, one yellow.
The grey robot arm may hide part of the scene.

Where is the {colour} cube? Use exactly one of:
  "zone_A", "zone_B", "zone_C"   it rests on that grey pad
  {others}   it sits on top of that other cube
  "table"         it is on the light table surface, on no pad and on no other cube
  "not_visible"   you cannot see the {colour} cube in the image
A cube that is only near a pad, not on it, is "table". Most of the table is not a pad."""
NOTES = {"scene": "one short sentence per cube: what it is standing on, and the letter of the nearest pad",
         "cube": "one short sentence: what the cube is standing on, and the letter of the nearest pad"}
STYLES = ("scene", "scene_notes", "cube", "cube_notes", "detect")


def camera(pc, view: str, w: int, h: int):
    """PyBullet view and projection matrices of `view`, and the same as numpy 4x4."""
    vm = pc.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=TARGET, distance=DIST, roll=0,
                                              upAxisIndex=2, **CAMS[view])
    pm = pc.computeProjectionMatrixFOV(fov=FOV, aspect=w / h, nearVal=0.1, farVal=100.0)
    return vm, pm, np.array(vm).reshape(4, 4).T, np.array(pm).reshape(4, 4).T


def render(env, view: str, labels: bool = True, size: Tuple[int, int] = (W, H)) -> Image.Image:
    """The camera image of `view`, optionally with the pad letters drawn in."""
    pc, (w, h) = env.sim.physics_client, size
    vm, pm, V, P = camera(pc, view, w, h)
    rgba = pc.getCameraImage(w, h, vm, pm, shadow=1, renderer=pc.ER_TINY_RENDERER)[2]
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[..., :3])
    if labels:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(FONT, 26)
        except OSError:
            font = ImageFont.load_default()
        for z, xy in ZONES.items():
            # just outside the pad, on the side away from the robot
            c = P @ V @ np.array([xy[0] + ZONE_HALF + 0.022, xy[1], 0.001, 1.0])
            px, py = (c[0] / c[3] + 1) / 2 * w, (1 - c[1] / c[3]) / 2 * h
            draw.text((px, py), z[-1], fill=(255, 255, 255), font=font, anchor="mm",
                      stroke_width=3, stroke_fill=(0, 0, 0))
    return img


# ---- style "detect": the VLM only finds the cubes; calibrated geometry does the rest
DETECT_SIZE = (1036, 784)          # qwen2.5vl returns boxes in this pixel grid without rescaling (measured)
DETECT_PROMPT = ("Detect the red cube, the green cube, the blue cube and the yellow cube in the image. "
                 "Output a JSON list with one entry per cube you can see: "
                 "{\"label\": colour, \"bbox_2d\": [x1, y1, x2, y2]}. Leave out cubes you cannot see.")
BOX = re.compile(r'"label":\s*"(\w+)[^\[]*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]')
VIEW_POSE = (-0.30, 0.0, 0.32)     # where the empty hand waits while a picture is taken (m, table frame)
STACK_TOL = 0.035                  # m: how close a cube's stacked-hypothesis position must be to another cube;
                                   # set on development seeds 7000+ (real stacks: 21 of 24 within it from the
                                   # front and top cameras, 19 of 24 from the side; unstacked pairs: 0, 0, 2 of 22)


def backproject(V, P, w: int, h: int, u: float, v: float, z: float) -> np.ndarray:
    """World point at height z that pixel (u, v) looks at."""
    M = np.linalg.inv(P @ V)
    x, y = 2 * u / w - 1, 1 - 2 * v / h
    a, b = M @ np.array([x, y, -1.0, 1.0]), M @ np.array([x, y, 1.0, 1.0])
    a, b = a[:3] / a[3], b[:3] / b[3]
    return a + (z - a[2]) / (b[2] - a[2]) * (b - a)


def places_from_boxes(boxes: Dict[str, List[float]], names: List[str], V, P, w: int, h: int,
                      view: str = "front", xy: Optional[Dict[str, List[float]]] = None) -> Dict[str, str]:
    """Turn cube boxes into places with the known camera and pad geometry.

    A cube's box centre is read as the cube centre.  It is either on the table
    (centre at half a cube) or on another cube (centre at one and a half): it
    counts as stacked when the second reading puts it over another detected cube.
    From one camera the two readings can be confused -- that is a real ambiguity
    of a single view, and it is left in."""
    table, stack = {}, {}
    for n, (x1, y1, x2, y2) in boxes.items():
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        table[n] = backproject(V, P, w, h, u, v, 0.5 * OBJECT_SIZE)[:2]
        stack[n] = backproject(V, P, w, h, u, v, 1.5 * OBJECT_SIZE)[:2]
    out: Dict[str, str] = {}
    for n in names:
        if n not in boxes:
            out[n] = "not_visible"
            continue
        below = [(float(np.linalg.norm(stack[n] - table[m])), m) for m in boxes if m != n]
        # from the front and the side a stacked cube is also drawn higher up than the one under it
        below = [(d, m) for d, m in below if d <= STACK_TOL and (view == "top" or boxes[n][1] < boxes[m][1])]
        if xy is not None:                                   # where this camera puts the cube (m)
            xy[n] = [float(v) for v in (stack[n] if below else table[n])]
        if below:
            out[n] = "on_" + min(below)[1]
            continue
        zone = next((z for z, xy in ZONES.items()
                     if abs(table[n][0] - xy[0]) <= ZONE_HALF and abs(table[n][1] - xy[1]) <= ZONE_HALF), None)
        out[n] = zone or "table"
    return out


class VLMPerception:
    mode = "vlm"

    def __init__(self, env, spec: Dict[str, Any], seed: int, model: str = "qwen2.5vl:7b",
                 host: str = "http://127.0.0.1:11435", cache_path: Optional[str] = None,
                 save_dir: Optional[str] = None, timeout: float = 300.0, style: str = "scene",
                 retract_first: bool = True) -> None:
        assert style in STYLES
        self.style, self.retract_first = style, retract_first
        self.env, self.spec, self.seed, self.model, self.host = env, spec, seed, model, host
        self.cache_path, self.save_dir, self.timeout = cache_path, save_dir, timeout
        self.goals = [g for g in spec.get("goals", []) if g["type"] in ("in_zone", "on_top_of")]
        self.calls, self.seconds, self.hits, self.failed = 0, 0.0, 0, 0
        self._cache: Optional[Dict[str, Any]] = None
        names = list(env.object_names)
        self.schema = {"type": "object", "required": names,
                       "properties": {n: {"type": "string", "enum": [p for p in PLACES if p != f"on_{n}"]}
                                      for n in names}}

    # ------------------------------------------------------------------ the model
    def _load(self) -> Dict[str, Any]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    for line in f:
                        if line.strip():
                            e = json.loads(line)
                            self._cache[e["key"]] = e["out"]
        return self._cache

    def _chat(self, png: bytes, prompt: str, schema: Dict[str, Any], view: str) -> Optional[Dict[str, Any]]:
        key = hashlib.sha256(self.model.encode() + prompt.encode() + json.dumps(schema, sort_keys=True).encode()
                             + png).hexdigest()
        cache = self._load()
        if key in cache:
            self.hits += 1
            return cache[key]
        body = {"model": self.model, "stream": False, "format": schema,
                "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(png).decode()]}],
                "options": {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 200}}
        req = urllib.request.Request(self.host.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0, out = time.time(), None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads((json.load(r).get("message") or {}).get("content", ""))
            if not isinstance(out, dict):
                out = None
        except Exception:                                     # no answer is an answer: nothing detected
            out = None
        self.calls += 1
        self.seconds += time.time() - t0
        if out is None:
            self.failed += 1
            return None
        cache[key] = out
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "out": out, "view": view}) + "\n")
        return out

    def boxes(self, img: Image.Image, view: str) -> Dict[str, List[float]]:
        """Cube boxes the VLM reports, by block name (first box per colour)."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        key = hashlib.sha256(self.model.encode() + DETECT_PROMPT.encode() + png).hexdigest()
        cache = self._load()
        if key in cache:
            self.hits += 1
            text = cache[key]
        else:
            body = {"model": self.model, "stream": False,
                    "messages": [{"role": "user", "content": DETECT_PROMPT,
                                  "images": [base64.b64encode(png).decode()]}],
                    "options": {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 300}}
            req = urllib.request.Request(self.host.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0, text = time.time(), None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = (json.load(r).get("message") or {}).get("content", "")
            except Exception:
                self.failed += 1
            self.calls += 1
            self.seconds += time.time() - t0
            if text is None:
                return {}
            cache[key] = text
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "out": text, "view": view}) + "\n")
        found: Dict[str, List[float]] = {}
        for m in BOX.finditer(text):
            name = m.group(1).lower() + "_block"
            if name in self.env.object_names and name not in found:
                found[name] = [float(x) for x in m.groups()[1:]]
        return found

    def retract(self, max_steps: int = 60) -> None:
        """Move the empty hand up and back out of the cameras' way before a picture
        is taken: rise first, then translate.  Scripted, no randomness."""
        robot, sim = self.env.robot, self.env.sim
        goal = np.array(VIEW_POSE)
        for _ in range(max_steps):
            ee = np.asarray(robot.get_ee_position(), dtype=float)
            target = goal.copy()
            if ee[2] < goal[2] - 0.03:
                target[:2] = ee[:2]
            if np.linalg.norm(ee - goal) < 0.015:
                break
            d = np.clip((target - ee) / 0.05, -1.0, 1.0)
            robot.set_action(np.array([d[0], d[1], d[2], 1.0], dtype=np.float32))
            sim.step()

    def look(self, view: str, skip: Optional[str] = None) -> Tuple[Image.Image, Optional[Dict[str, str]]]:
        """Render `view` and return (image, where the perception system says each cube is).
        `skip` is the block in the gripper, if any; with an empty hand the arm
        first moves out of the way (a held block would be dropped by that)."""
        if self.retract_first and skip is None:
            self.retract()
        if self.style == "detect":
            w, h = DETECT_SIZE
            img = render(self.env, view, labels=False, size=DETECT_SIZE)
            _, _, V, P = camera(self.env.sim.physics_client, view, w, h)
            found = self.boxes(img, view)
            self.last_boxes, self.last_xy = found, {}
            return img, places_from_boxes(found, list(self.env.object_names), V, P, w, h, view, self.last_xy)
        img = render(self.env, view)
        return img, self.ask(img, view, skip=skip)

    def ask(self, img: Image.Image, view: str, skip: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Where the VLM says each cube is, or None if it gave no usable answer.
        `skip` names a cube that need not be asked about (the one in the gripper)."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png, names = buf.getvalue(), list(self.env.object_names)
        notes = self.style.endswith("_notes")
        if self.style.startswith("scene"):
            schema = copy.deepcopy(self.schema)
            if notes:
                schema["properties"] = {"notes": {"type": "string"}, **schema["properties"]}
                schema["required"] = ["notes"] + schema["required"]
            prompt = PROMPT.format(view=view) + (f"\nFirst write \"notes\": {NOTES['scene']}." if notes else "")
            return self._chat(png, prompt, schema, view)
        said: Dict[str, str] = {}
        for n in names:
            if n == skip:
                continue
            opts = [p for p in PLACES if p != f"on_{n}"]
            props: Dict[str, Any] = {"place": {"type": "string", "enum": opts}}
            if notes:
                props = {"notes": {"type": "string"}, **props}
            others = ", ".join(f'"{p}"' for p in opts if p.startswith("on_"))
            prompt = PROMPT_CUBE.format(view=view, colour=n.split("_")[0], others=others) \
                + (f"\nFirst write \"notes\": {NOTES['cube']}." if notes else "")
            out = self._chat(png, prompt, {"type": "object", "required": list(props), "properties": props}, view)
            said[n] = (out or {}).get("place", "not_visible")
        return said

    # ------------------------------------------------------------------ the report
    @staticmethod
    def to_graph(graph: Dict[str, Any], said: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """The scene graph the VLM's answer describes.  Blocks that are off the
        table, and the block in the gripper, keep their true entries."""
        out = copy.deepcopy(graph)
        said = said or {}
        free = [n for n in graph["in_zone"] if n != graph.get("held")]
        place = {n: said.get(n, "not_visible") for n in free}
        for n in free:
            p = place[n]
            out["on_top_of"][n] = None
            if p.startswith("zone_"):
                out["in_zone"][n] = p
            elif p == "table":
                out["in_zone"][n] = None
        for n in free:                                       # stacked: on that cube, and in its zone
            p = place[n]
            if p.startswith("on_"):
                below = p[3:]
                ok = below in out["in_zone"] and below != n and place.get(below) != "not_visible"
                out["on_top_of"][n] = below if ok else None
                q = place.get(below, "")
                out["in_zone"][n] = q if q.startswith("zone_") else (graph["in_zone"].get(below)
                                                                     if below == graph.get("held") else None)
        for n in free:
            if place[n] == "not_visible":
                for k in ("in_zone", "clear", "on_top_of"):
                    out[k].pop(n, None)
                out["on_top_of"] = {k: (None if v == n else v) for k, v in out["on_top_of"].items()}
                out.setdefault("hidden", []).append(n)
        _reclear(out)
        return out

    @staticmethod
    def _met(g, graph) -> bool:
        if g["type"] == "in_zone":
            return graph["in_zone"].get(g["obj"]) == g["zone"]
        return graph["on_top_of"].get(g["top"]) == g["bottom"]

    def perceive(self, graph: Dict[str, Any], view: str = "front"
                 ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        img, said = self.look(view, skip=graph.get("held"))
        out = self.to_graph(graph, said)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            k = len(os.listdir(self.save_dir))
            img.save(os.path.join(self.save_dir, f"{self.seed}_{k:04d}_{view}.png"))
        est = getattr(self, "last_xy", None)
        if est is not None:                                  # positions as this camera estimated them
            est = {n: [x, y, graph["positions"][n][2]] for n, (x, y) in est.items() if n in out["in_zone"]}
            held = graph.get("held")
            if held is not None:
                est[held] = graph["positions"][held]
        if _report(out) == _report(graph):
            return (graph if est is None else dict(graph, positions=est)), None
        if est is not None:
            out["positions"] = est
        errors: List[Dict[str, Any]] = []
        for g in self.goals:
            was, now = self._met(g, graph), self._met(g, out)
            if was != now:
                errors.append({"goal": g, "direction": "false_success" if now else "false_failure", "type": "vlm"})
        return out, {"type": "vlm", "direction": errors[0]["direction"] if errors else "other",
                     "errors": errors, "view": view, "mode": "vlm", "said": said}


assert set(CAMS) == set(VIEWS)
