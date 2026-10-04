# Packaging a dpg_system patch as a deployable application

Working notes from a 2026-09-10 discussion. Nothing here has been implemented yet;
this is an assessment of the options and of what in the code would have to change first.

## Short answer

Yes, but the realistic form for this codebase is **a relocatable environment plus a
launcher, in a folder**, not a single frozen executable. The dependency set (torch,
opencv, moderngl, pybullet, spacy, llama-cpp with Metal, pedalboard/JUCE, NDI,
dearpygui) is close to the worst case for PyInstaller-style freezing, and the folder
approach gives the same double-click experience once wrapped in a `.app`.

## What the code already provides

The engine is already driven by two files in the working directory:

- `dpg_system_config.json` selects which node families are imported. Families set to
  false are never imported, so this file also decides which heavy dependencies are needed.
- `dpg_app_default_patcher.json` names a patch to open automatically at launch.

So "application = engine + patch" is mostly: a folder containing those two files set
appropriately, a slimmed environment, and something double-clickable that runs
`dpg_system_main.py`. A `load_action` node in the patch can drive startup behaviour
without GUI clicks (see the headless-drive notes in memory).

## Option 1 (recommended): packed conda env + launcher folder

Use `conda-pack` on a slimmed clone of the working env to produce a relocatable tarball.
Ship it beside the repo checkout:

```
MyPiece/
  MyPiece.app            (or run.command)
                          -> cd payload && env/bin/python -u dpg_system_main.py
  payload/
    env/                 conda-pack output; run conda-unpack once on first launch
    dpg_system/          the package, minus models/ that the patch does not use
    patches/my_piece.json
    dpg_system_config.json         only the node families the patch needs set true
    dpg_app_default_patcher.json   {"path": "patches/my_piece.json"}
```

- The `.app` can be a hand-made bundle (Info.plist + shell script as the executable)
  or a Platypus wrapper.
- A launchd LaunchAgent with `KeepAlive` gives auto-start and restart-on-crash for an
  installation machine.
- Windows: same recipe with `environment_windows.yml`, conda-pack, and a `.bat`.
- Size: the current env is 3.2 GB and `dpg_system/models` is 5.5 GB. A patch that does
  not touch torch, spacy, whisper, or llama drops most of both. Trim the config first,
  then build the env from what is actually imported.
- `install.sh` steps outside the environment file (chumpy from git, the spacy
  `en_core_web_lg` download) must be baked into the packed env.

## Option 2: PyInstaller / Nuitka single bundle

Possible in principle, but expect a long fight:

- Node modules load via `import_module` in `dpg_app.py`, so every `*_nodes` module
  needs a hidden-import entry.
- torch, opencv, moderngl, pybullet, spacy, llama-cpp (Metal), pedalboard (JUCE),
  NDI and dearpygui each bring native libraries, data files, or dlopen paths needing hooks.
- Output is still multi-gigabyte, startup slows, and each new node family reopens the work.
- The `objc[` GLFW duplicate-class filter in `dpg_system_main.py` and the
  `os._exit` shutdown path (pedalboard JUCE timer thread) both still apply.

Only worth it if the deliverable must be a self-contained `.app` handed to people who
will not tolerate a folder. Even then, option 1 wrapped in a `.app` looks identical.

Docker is not an option: Metal, CoreAudio, MIDI and the GUI do not survive it on macOS.

## Code changes needed regardless of option

1. **Portable paths in patches (biggest blocker).** Every patch in `patches/` stores its
   own absolute path (`/Users/drokeby/dpg_system/patches/...`), and file-referencing
   nodes (movie, sampler, shader, SMPL model paths, etc.) will store absolute asset
   paths the same way. Deployment needs a convention: paths stored relative to the
   patch file, or a token such as `$PATCH_DIR` / `$DPG_HOME` resolved on load.
2. **cwd-independence of config files.** `dpg_app.py` opens `dpg_system_config.json`
   with a relative path at import time (line ~28, again ~107 and ~317), and the
   `<project>_recent_patchers.json` file likewise. Either the launcher must chdir to the
   payload root, or these should resolve against an env var such as `DPG_SYSTEM_HOME`.
3. **Kiosk mode.** Does not exist yet. There is an `easy` flag in the config but no
   fullscreen / undecorated viewport, hidden editor, or locked editing. A performance
   deployment wants all three.
4. **Gatekeeper.** An unsigned app copied to another Mac is quarantined. For a machine
   you set up yourself: `xattr -dr com.apple.quarantine`. Wider distribution would need
   signing and notarization, which is a separate project.

## Longer-term structure

If deployments become routine: make dpg_system a versioned installable package
(`pyproject.toml`), and give each piece its own small repo holding the patch, assets,
config and launcher, pinned to an engine version. That separates "the engine changed"
from "the piece changed" and makes rebuilding an installation machine reproducible.

## Suggested order of work

1. Portable path convention for patches and file-referencing nodes.
2. cwd-independence of the config / recent-patchers files.
3. Kiosk flag (fullscreen, hidden editor, editing locked).
4. conda-pack recipe + `.app` launcher + launchd plist.

Steps 1 and 2 are what actually make a patch run on another machine; 3 and 4 are
polish on top of that.
