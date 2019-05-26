import sys
import re
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
from raybox import RayBox
from camera import Camera
from drr_set import DrrSet
from drr_registration import DrrRegistration
from utils import str_to_mat, read_rho, read_image_as_np


class RecenterWidget(QtWidgets.QWidget):
    new_center = QtCore.Signal(list)

    def __init__(self, parent):
        super().__init__(parent)
        self.input_box = QtWidgets.QLineEdit(self)
        self.input_box.setPlaceholderText('X,Y,Z')
        self.input_box.setAlignment(QtCore.Qt.AlignCenter)
        rx = QtCore.QRegExp('-?\\d+(.?\\d*)?,-?\\d+(.?\\d*)?,-?\\d+(.?\\d*)?')
        self.input_box.setValidator(QtGui.QRegExpValidator(rx, self))
        self.button = QtWidgets.QPushButton('Recenter at', self)
        self.button.clicked.connect(self.on_recenter_but)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.input_box, QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    @QtCore.Slot()
    def on_recenter_but(self):
        cords_str = self.input_box.text()
        center = list(map(lambda x: float(x), cords_str.split(',')))
        self.new_center.emit(center)

class ThresholdWidget(QtWidgets.QWidget):
    new_threshold = QtCore.Signal(float)

    def __init__(self, parent):
        super().__init__(parent)
        self.input_box = QtWidgets.QLineEdit(self)
        self.input_box.setPlaceholderText('0')
        self.input_box.setAlignment(QtCore.Qt.AlignCenter)
        self.input_box.setValidator(QtGui.QDoubleValidator())
        self.button = QtWidgets.QPushButton('Set Threshold', self)
        self.button.clicked.connect(self.on_clicked)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.input_box, QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    @QtCore.Slot()
    def on_clicked(self):
        threshold = float(self.input_box.text())
        self.new_threshold.emit(threshold)


class ImageWidget(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.base = None
        self.drr = None
        self.alpha = 0.5
        self.setScaledContents(True)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

    @staticmethod
    def np_to_qrgb_pixmap(arr, color, alpha=0.5):
        h, w = arr.shape
        arr = (255*arr).astype(np.uint8).flatten(order='C')
        qrgb_dict = {'r': lambda x: QtGui.qRgba(x, 0, 0, int(255*alpha)),
                     'g': lambda x: QtGui.qRgba(0, x, 0, int(255*alpha)),
                     'b': lambda x: QtGui.qRgba(0, 0, x, int(255*alpha))}
        colortable = [qrgb_dict[color](i) for i in range(256)]
        img = QtGui.QImage(
            arr,
            w, h,
            w*np.nbytes[np.uint8],
            QtGui.QImage.Format_Indexed8)
        img.setColorTable(colortable)
        return QtGui.QPixmap.fromImage(img)

    def blend_with_base(self, overlay):
        pm = QtGui.QPixmap(overlay.size())
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter()
        painter.begin(pm)
        # Draw overlay
        painter.drawPixmap(0, 0, self.base)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        self.setPixmap(pm)

    @QtCore.Slot(QtGui.QPixmap)
    def on_base_pixmap(self, base):
        self.base = base
        self.setPixmap(base)

    @QtCore.Slot(float)
    def on_alpha(self, alpha):
        if self.drr is not None:
            self.alpha = alpha
            overlay = ImageWidget.np_to_qrgb_pixmap(self.drr, 'r', alpha)
            overlay = overlay.scaled(self.base.size(), QtCore.Qt.IgnoreAspectRatio)
            self.setPixmap(self.base)
            self.blend_with_base(overlay)

    @QtCore.Slot(np.ndarray)
    def on_drr(self, drr):
        self.drr = drr
        overlay = self.np_to_qrgb_pixmap(drr, 'r', self.alpha)
        overlay = overlay.scaled(self.base.size(), QtCore.Qt.IgnoreAspectRatio)
        self.blend_with_base(overlay)

class ParametersWidget(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setTickInterval(1)
        self.alpha_label = QtWidgets.QLabel('Alpha', self)
        self.tx_lab = QtWidgets.QLabel('Tx', self)
        self.ty_lab = QtWidgets.QLabel('Ty', self)
        self.tz_lab = QtWidgets.QLabel('Tz', self)
        self.rx_lab = QtWidgets.QLabel('Rx', self)
        self.ry_lab = QtWidgets.QLabel('Ry', self)
        self.rz_lab = QtWidgets.QLabel('Rz', self)
        self.tx_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ty_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tz_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.rx_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ry_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.rz_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tx_widg = QtWidgets.QDoubleSpinBox(self)
        self.ty_widg = QtWidgets.QDoubleSpinBox(self)
        self.tz_widg = QtWidgets.QDoubleSpinBox(self)
        self.tx_widg.setRange(-3E10, 3E10)
        self.ty_widg.setRange(-3E10, 3E10)
        self.tz_widg.setRange(-3E10, 3E10)
        self.tx_widg.setSingleStep(5)
        self.ty_widg.setSingleStep(5)
        self.tz_widg.setSingleStep(5)
        self.tx_widg.setDecimals(2)
        self.ty_widg.setDecimals(2)
        self.tz_widg.setDecimals(2)
        self.rx_widg = QtWidgets.QDoubleSpinBox(self)
        self.ry_widg = QtWidgets.QDoubleSpinBox(self)
        self.rz_widg = QtWidgets.QDoubleSpinBox(self)
        self.rx_widg.setRange(-180, 180)
        self.ry_widg.setRange(-180, 180)
        self.rz_widg.setRange(-180, 180)
        self.ry_widg.setSingleStep(5)
        self.rz_widg.setSingleStep(5)
        self.rx_widg.setSingleStep(5)
        self.ry_widg.setDecimals(2)
        self.rz_widg.setDecimals(2)
        self.rx_widg.setDecimals(2)
        # Layout
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.alpha_slider, 0, 0, 1, 2)
        self.layout.addWidget(self.alpha_label, 0, 2, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.tx_lab, 1, 0, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.ty_lab, 1, 1, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.tz_lab, 1, 2, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.tx_widg, 2, 0)
        self.layout.addWidget(self.ty_widg, 2, 1)
        self.layout.addWidget(self.tz_widg, 2, 2)
        self.layout.addWidget(self.rx_lab, 4, 0, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.ry_lab, 4, 1, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.rz_lab, 4, 2, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.rx_widg, 3, 0)
        self.layout.addWidget(self.ry_widg, 3, 1)
        self.layout.addWidget(self.rz_widg, 3, 2)
        self.setLayout(self.layout)

    def get_params(self):
        params = [self.tx_widg.value(),
                  self.ty_widg.value(),
                  self.tz_widg.value(),
                  self.rx_widg.value(),
                  self.ry_widg.value(),
                  self.rz_widg.value()]
        return params

    def set_params(self, *params):
        tx, ty, tz, rx, ry, rz = params
        self.tx_widg.setValue(tx)
        self.ty_widg.setValue(ty)
        self.tz_widg.setValue(tz)
        self.rx_widg.setValue(rx)
        self.ry_widg.setValue(ry)
        self.rz_widg.setValue(rz)

class MainWindow(QtWidgets.QMainWindow):
    new_ct = QtCore.Signal(list)
    input_cams_sig = QtCore.Signal(list)
    base_pixmap_1 = QtCore.Signal(QtGui.QPixmap)
    base_pixmap_2 = QtCore.Signal(QtGui.QPixmap)
    drr1 = QtCore.Signal(np.ndarray)
    drr2 = QtCore.Signal(np.ndarray)
    alpha = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('DRR Viewer')
        self.setMinimumSize(QtCore.QSize(1280, 720))
        self.central_widg = QtWidgets.QWidget(self)
        # File dialog
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_dialog.setViewMode(QtWidgets.QFileDialog.List)
        self.file_dialog_out = ''
        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu('File')
        self.fd_in_out = {
            'CT': [self.on_ct_menu, ''],
            'Camera Files': [self.on_cam_menu, ''],
            'X-Ray Files': [self.on_xray_menu, '']
        }
        for entry in ['CT', 'Camera Files', 'X-Ray Files']:
            action = QtWidgets.QAction('Open {}'.format(entry), self)
            action.triggered.connect(self.fd_in_out[entry][0])
            self.file_menu.addAction(action)
        self.edit_menu = self.menu.addMenu('Edit')
        # Status bar
        self.status_bar = self.statusBar()
        # Images
        self.img1_widg = ImageWidget(self.central_widg)
        self.img2_widg = ImageWidget(self.central_widg)
        # Parameter widgs
        self.params_widg = ParametersWidget(self.central_widg)
        # Threshold widg
        self.threshold_widg = ThresholdWidget(self.central_widg)
        # Refresh button
        self.refresh_butn = QtWidgets.QPushButton('Refresh', self)
        # Recenter button
        self.recenter_widg = RecenterWidget(self)
        # Layout
        refr_thr_layout = QtWidgets.QVBoxLayout()
        refr_thr_layout.addWidget(self.refresh_butn)
        refr_thr_layout.addWidget(self.recenter_widg)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.img1_widg, 0, 0)
        layout.addWidget(self.img2_widg, 0, 2)
        layout.addWidget(self.threshold_widg, 0, 1)
        layout.addWidget(self.params_widg, 1, 0, 1, 2)
        layout.addLayout(refr_thr_layout, 1, 2)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 1)
        self.central_widg.setLayout(layout)
        self.setCentralWidget(self.central_widg)
        # Focus
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        # Connect signals and slots
        self.base_pixmap_1.connect(self.img1_widg.on_base_pixmap)
        self.base_pixmap_2.connect(self.img2_widg.on_base_pixmap)
        self.params_widg.alpha_slider.valueChanged.connect(self.on_alphaslider_update)
        self.refresh_butn.released.connect(self.on_refresh_butn)
        self.threshold_widg.new_threshold.connect(self.on_new_threshold)
        self.recenter_widg.new_center.connect(self.on_new_center)
        self.alpha.connect(self.img1_widg.on_alpha)
        self.alpha.connect(self.img2_widg.on_alpha)
        self.drr1.connect(self.img1_widg.on_drr)
        self.drr2.connect(self.img2_widg.on_drr)
        # Logic
        self.raybox = RayBox()
        self.drr_set = DrrSet(self.raybox)
        self.drr_registration = DrrRegistration(self.drr_set)
        runoptim_action = QtWidgets.QAction('Run Optimizer', self)
        runoptim_action.triggered.connect(self.on_runoptim_action)
        self.edit_menu.addAction(runoptim_action)
        popup_3d_action = QtWidgets.QAction('Pop up 3D visualizer', self)
        self.edit_menu.addAction(popup_3d_action)
        popup_3d_action.triggered.connect(self.plot_set)
        gpu_mode_action = QtWidgets.QAction('GPU mode', self)
        gpu_mode_action.setCheckable(True)
        gpu_mode_action.toggled.connect(self.on_toggled_gpu_mode)
        self.edit_menu.addAction(gpu_mode_action)

    @QtCore.Slot()
    def on_runoptim_action(self):
        params = np.array(self.params_widg.get_params())
        res = self.drr_registration.register(params)
        print(res)
        new_params = res.x.tolist()
        self.params_widg.set_params(*new_params)
        self.drr_set.set_tfm_params(*new_params)
        self.draw_drrs()

    @QtCore.Slot(bool)
    def on_toggled_gpu_mode(self, checked):
        if checked:
            self.raybox.mode = 'gpu'
        else:
            self.raybox.mode = 'cpu'

    @QtCore.Slot()
    def plot_set(self):
        self.drr_set.plot_camera_set()

    @QtCore.Slot(list)
    def on_new_center(self, center):
        self.drr_set.move_to(np.array(center))
        self.params_widg.set_params(*(self.drr_set.params))

    @QtCore.Slot(int)
    def on_alphaslider_update(self, val):
        alpha = val / 100
        print('alpha', alpha)
        self.alpha.emit(alpha)

    @QtCore.Slot(float)
    def on_new_threshold(self, val):
        self.raybox.set_threshold(val)

    @QtCore.Slot()
    def on_ct_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['CT'][1] = fpaths
            self.set_rho(fpaths)

    @QtCore.Slot()
    def on_cam_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['Camera Files'][1] = fpaths
            print(fpaths)
            self.init_cams_from_path(fpaths)

    @QtCore.Slot()
    def on_xray_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['X-Ray Files'][1] = fpaths
            xray1 = read_image_as_np(fpaths[0])
            xray2 = read_image_as_np(fpaths[1])
            self.drr_registration.set_xrays(xray1, xray2)
            self.base_pixmap_1.emit(QtGui.QPixmap(fpaths[0]))
            self.base_pixmap_2.emit(QtGui.QPixmap(fpaths[1]))

    @QtCore.Slot()
    def on_refresh_butn(self):
        params = self.params_widg.get_params()
        self.drr_set.set_tfm_params(*params)
        self.draw_drrs()

    def draw_drrs(self):
        drr1, drr2 = self.raybox.trace_rays()
        drr1 = (1-drr1)
        drr2 = (1-drr2)
        print('DRR')
        self.drr1.emit(drr1)
        self.drr2.emit(drr2)

    def set_rho(self, fpaths):
        rho, sp = read_rho(fpaths[0])
        self.raybox.set_rho(rho, sp)
        print('Set Rho')

    def init_cams_from_path(self, fpaths):
        with open(fpaths[0]) as f:
            s1 = f.read()
        with open(fpaths[1]) as f:
            s2 = f.read()
        m1 = str_to_mat(re.search('M = \[(.*)\]', s1).group(1))
        k1 = str_to_mat(re.search('K = \[(.*)\]', s1).group(1))
        m2 = str_to_mat(re.search('M = \[(.*)\]', s2).group(1))
        k2 = str_to_mat(re.search('K = \[(.*)\]', s2).group(1))
        cam1 = Camera(m=m1, k=k1, h=768, w=768)
        cam2 = Camera(m=m2, k=k2, h=768, w=768)
        self.drr_set.set_cams(cam1, cam2)
        print('Set cams')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
