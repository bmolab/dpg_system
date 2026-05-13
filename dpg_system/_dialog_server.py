#!/usr/bin/env python3
"""Persistent GTK file-dialog server for dpg_system.

Started once per session by the main app (via _LinuxDialogServer in node.py).
GTK is initialised on startup so subsequent dialogs appear with no delay.
Communicates over stdin/stdout with newline-delimited JSON.

Request:  {"action": "open"|"save", "dir": "<path>", "name": "<filename>"}
Response: {"path": "<selected path>"}  (empty string if cancelled)
"""
import sys
import json
import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


def show_dialog(action, target_dir, default_name=""):
    if action == "save":
        dlg = Gtk.FileChooserDialog(
            title="Save As",
            action=Gtk.FileChooserAction.SAVE,
        )
        dlg.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE,   Gtk.ResponseType.OK,
        )
        dlg.set_do_overwrite_confirmation(True)
        if os.path.isdir(target_dir):
            dlg.set_current_folder(target_dir)
        if default_name:
            dlg.set_current_name(default_name)
    else:
        dlg = Gtk.FileChooserDialog(
            title="Open",
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,   Gtk.ResponseType.OK,
        )
        if os.path.isdir(target_dir):
            dlg.set_current_folder(target_dir)

    response = dlg.run()
    path = dlg.get_filename() if response == Gtk.ResponseType.OK else ""
    dlg.destroy()
    while Gtk.events_pending():
        Gtk.main_iteration_do(False)
    return path or ""


sys.stdout.write("READY\n")
sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line.strip())
        path = show_dialog(
            req.get("action", "open"),
            req.get("dir", os.getcwd()),
            req.get("name", ""),
        )
        sys.stdout.write(json.dumps({"path": path}) + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"path": "", "error": str(e)}) + "\n")
        sys.stdout.flush()