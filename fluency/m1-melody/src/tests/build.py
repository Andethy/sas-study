import os
import shutil

path_to_build_au = "../../plugin/Automator/Builds/MacOSX/build/Debug/Automator.component"
path_to_plugin_au = '/Library/Audio/Plug-Ins/Components/Automator.component'

path_to_build_vst3 = "../../plugin/Automator/Builds/MacOSX/build/Debug/Automator.vst3"
path_to_plugin_vst3 = '/Library/Audio/Plug-Ins/VST3/Automator.vst3'

if __name__ == '__main__':
    try:
        if os.path.exists(path_to_plugin_au):
            shutil.rmtree(path_to_plugin_au)
        shutil.copytree(path_to_build_au, path_to_plugin_au)
        print(f"Plugin copied to {path_to_plugin_au}")
    except Exception as e:
        print(f"OOPs: {e}")

    try:
        # if os.path.exists(path_to_plugin_vst3):
        #     shutil.rm(path_to_plugin_vst3)
        shutil.copy(path_to_build_vst3, path_to_plugin_vst3)
        print(f"Plugin copied to {path_to_plugin_vst3}")
    except Exception as e:
        print(f"OOPs: {e}")