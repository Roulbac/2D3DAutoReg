from camera_set import CameraSet

class DrrSet(CameraSet):
    def __init__(self, cam1, cam2, raybox):
        super().__init__(cam1, cam2)
        self.raybox = raybox
        self.raybox.set_cams(cam1, cam2)
