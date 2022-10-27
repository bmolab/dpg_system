from dpg_system.dpg_app import App
import threading

dpg_app = None


def run_dpg():
    global dpg_app
    dpg_app = App()
    dpg_app.start()
    dpg_app.run_loop()


dpg_thread = threading.Thread(target=run_dpg)
dpg_thread.run()

