from camera_set import CameraSet

class DrrSet(CameraSet):
    def __init__(self, raybox):
        super().__init__()
        self.raybox = raybox

    def set_cams(self, *cams):
        super().set_cams(*cams)
        self.raybox.set_cams(*cams)
