from camera_set import CameraSet

class DrrSet(CameraSet):
    def __init__(self, raybox):
        super().__init__()
        self.raybox = raybox

    def set_cams(self, cam1, cam2):
        super().set_cams(cam1, cam2)
        self.raybox.set_cams(cam1, cam2)
