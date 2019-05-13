import time
import sys
import re
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
from raybox import RayBox
from camera import Camera
from utils import str_to_mat, recons_DLT, read_rho


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
        h, w = arr.shape[0], arr.shape[1]
        arr = (255*arr).astype(np.uint8).flatten()
        qrgb_dict = {'r': lambda x: QtGui.qRgba(x, 0, 0, int(255*alpha)),
                     'g': lambda x: QtGui.qRgba(0, x, 0, int(255*alpha)),
                     'b': lambda x: QtGui.qRgba(0, 0, x, int(255*alpha))}
        colortable = [qrgb_dict[color](i) for i in range(256)]
        img = QtGui.QImage(arr, w, h, w, QtGui.QImage.Format_Indexed8)
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
    refresh_params = QtCore.Signal(list)

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
        self.phi_lab = QtWidgets.QLabel('Phi', self)
        self.theta_lab = QtWidgets.QLabel('Theta', self)
        self.psi_lab = QtWidgets.QLabel('Psi', self)
        self.tx_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ty_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tz_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.phi_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.theta_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.psi_lab.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tx_widg = QtWidgets.QDoubleSpinBox(self)
        self.ty_widg = QtWidgets.QDoubleSpinBox(self)
        self.tz_widg = QtWidgets.QDoubleSpinBox(self)
        self.tx_widg.setKeyboardTracking(False)
        self.ty_widg.setKeyboardTracking(False)
        self.tz_widg.setKeyboardTracking(False)
        self.tx_widg.setRange(-3E10, 3E10)
        self.ty_widg.setRange(-3E10, 3E10)
        self.tz_widg.setRange(-3E10, 3E10)
        self.tx_widg.setSingleStep(0.1)
        self.ty_widg.setSingleStep(0.1)
        self.tz_widg.setSingleStep(0.1)
        self.tx_widg.valueChanged.connect(self.on_refresh_call)
        self.ty_widg.valueChanged.connect(self.on_refresh_call)
        self.tz_widg.valueChanged.connect(self.on_refresh_call)
        self.phi_widg = QtWidgets.QDoubleSpinBox(self)
        self.theta_widg = QtWidgets.QDoubleSpinBox(self)
        self.psi_widg = QtWidgets.QDoubleSpinBox(self)
        self.phi_widg.setKeyboardTracking(False)
        self.theta_widg.setKeyboardTracking(False)
        self.psi_widg.setKeyboardTracking(False)
        self.phi_widg.setRange(-180, 180)
        self.theta_widg.setRange(-180, 180)
        self.psi_widg.setRange(-180, 180)
        self.theta_widg.setSingleStep(0.5)
        self.psi_widg.setSingleStep(0.5)
        self.phi_widg.setSingleStep(0.5)
        self.phi_widg.valueChanged.connect(self.on_refresh_call)
        self.theta_widg.valueChanged.connect(self.on_refresh_call)
        self.psi_widg.valueChanged.connect(self.on_refresh_call)
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
        self.layout.addWidget(self.phi_lab, 4, 0, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.theta_lab, 4, 1, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.psi_lab, 4, 2, QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.phi_widg, 3, 0)
        self.layout.addWidget(self.theta_widg, 3, 1)
        self.layout.addWidget(self.psi_widg, 3, 2)
        self.setLayout(self.layout)

    @QtCore.Slot()
    def on_refresh_call(self):
        params = self.get_params()
        self.refresh_params.emit(params)

    def get_params(self):
        params = [self.tx_widg.value(),
                  self.ty_widg.value(),
                  self.tz_widg.value(),
                  self.phi_widg.value(),
                  self.theta_widg.value(),
                  self.psi_widg.value()]
        return params


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
        self.setMinimumSize(QtCore.QSize(1280, 600))
        self.central_widg = QtWidgets.QWidget(self)
        # File dialog
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_dialog.setViewMode(QtWidgets.QFileDialog.List)
        self.file_dialog_out = ''
        # Menu
        self.menu = QtWidgets.QMenuBar()
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
        runoptim_action = QtWidgets.QAction('Run Optimizer', self)
        self.edit_menu.addAction(runoptim_action)
        popup_3d_action = QtWidgets.QAction('Pop up 3D visualizer', self)
        self.edit_menu.addAction(popup_3d_action)
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
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.img1_widg, 0, 0)
        layout.addWidget(self.img2_widg, 0, 2)
        layout.addWidget(self.threshold_widg, 0, 1)
        layout.addWidget(self.params_widg, 1, 0, 1, 2)
        layout.addWidget(self.refresh_butn, 1, 2)
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
        self.params_widg.refresh_params.connect(self.on_new_params)
        self.params_widg.alpha_slider.valueChanged.connect(self.on_alphaslider_update)
        self.refresh_butn.released.connect(self.params_widg.on_refresh_call)
        self.threshold_widg.new_threshold.connect(self.on_new_threshold)
        self.alpha.connect(self.img1_widg.on_alpha)
        self.alpha.connect(self.img2_widg.on_alpha)
        self.drr1.connect(self.img1_widg.on_drr)
        self.drr2.connect(self.img2_widg.on_drr)
        # Logic
        self.raybox = RayBox('cpu')
        # Debug stuff
        # b = np.array([-3, -2, 0], dtype=np.float32)
        # n = np.array([3, 3, 3], dtype=np.int32)
        # sp = np.array([1, 1, 1], dtype=np.float32)
        # rho = np.ones((n - 1).tolist(), dtype=np.float32)
        # self.raybox.set_rho(rho, b, n, sp)
        # h, w = 768, 768
        # k = np.array([[2 * (h / 2), 0, 1 * (h / 2), 0],
        #               [0, 2 * (w / 2), 1 * (w / 2), 0],
        #               [0, 0, 1, 0]])
        # m1 = np.array([[1, 0, 0, 2],
        #                [0, 0, -1, 1],
        #                [0, 1, 0, -4],
        #                [0, 0, 0, 1]])
        # m2 = np.array([[0, -1, 0, -1],
        #                [0, 0, -1, 1],
        #                [1, 0, 0, -3],
        #                [0, 0, 0, 1]])
        # cam1 = Camera(m=m1, k=k, h=h, w=w)
        # cam2 = Camera(m=m2, k=k, h=h, w=w)
        # self.raybox.set_cams(cam1, cam2)
        # pm1 = QtGui.QPixmap('/Users/reda/Desktop/Work/MSc/Projects/drr/L4L5_0.BMP')
        # pm2 = QtGui.QPixmap('/Users/reda/Desktop/Work/MSc/Projects/drr/drr_AP.bmp')
        # self.img1_widg.base = pm1
        # self.img2_widg.base = pm2
        # self.img1_widg.setPixmap(pm1)
        # self.img2_widg.setPixmap(pm2)


    @QtCore.Slot(list)
    def on_new_params(self, params):
        new_tfm = Camera.m_from_params(params)
        cams = self.raybox.get_cams()
        cams[0].tfm = new_tfm
        m = cams[0].m
        m_prime = cams[1].m
        cams[1].tfm = np.linalg.multi_dot(
            [m_prime,
             np.linalg.inv(m),
             new_tfm,
             m,
             np.linalg.inv(m_prime)]
        )
        self.raybox.set_cams(*cams)
        drr1, drr2 = self.raybox.trace_rays()
        drr1 = (1-drr1)
        drr2 = (1-drr2)
        # drr1, drr2  = np.ones((768, 768)), np.ones((768, 768))
        # drr1, drr2 = 0.5*drr1, 0.5*drr2
        print('DRR')
        self.drr1.emit(drr1)
        self.drr2.emit(drr2)

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
            self.init_cams_from_path(fpaths)

    @QtCore.Slot()
    def on_xray_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['X-Ray Files'][1] = fpaths
            self.base_pixmap_1.emit(QtGui.QPixmap(fpaths[0]))
            self.base_pixmap_2.emit(QtGui.QPixmap(fpaths[1]))

    def set_rho(self, fpaths):
        rho, b, n, sp = self.raybox.get_rho_params(fpaths[0])
        self.raybox.set_rho(rho, b, n, sp)
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
        self.raybox.set_cams(cam1, cam2)
        print('Set cams')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
