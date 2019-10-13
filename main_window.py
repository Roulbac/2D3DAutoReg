import sys
import re
import numpy as np
import matplotlib.pyplot as plt
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
        button_font = self.button.font()
        button_font.setPointSize(10)
        self.button.setFont(button_font)
        # self.button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.input_box.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
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
    clicked = QtCore.Signal(float, float)
    moved = QtCore.Signal(float, float)
    released = QtCore.Signal()
    roi_select = QtCore.Signal(float, float, float, float)
    roi_finalize = QtCore.Signal(float, float, float, float)

    def __init__(self, parent):
        super().__init__(parent)
        self.base = QtGui.QPixmap()
        self.overlay = QtGui.QPixmap()
        self.roi = QtGui.QPixmap()
        self.setPixmap(self.base)
        self.drr = None
        self.alpha = 0.5
        self.setScaledContents(True)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.roi_enabled = False
        self.is_selecting_roi = False
        self.a, self.b, self.c, self.d = -1, -1, -1, -1
        self.roi_select.connect(self.on_roi_select)
        self.clicked.connect(self.on_clicked)
        self.moved.connect(self.on_moved)
        self.released.connect(self.on_released)

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

    def set_base(self, base):
        self.base = base
        pm = QtGui.QPixmap(base.size())
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter()
        painter.begin(pm)
        painter.drawPixmap(0, 0, base)
        if not self.overlay.isNull():
            painter.drawPixmap(0, 0, self.overlay)
        if not self.roi.isNull():
            painter.drawPixmap(0, 0, self.roi)
        painter.end()
        self.setPixmap(pm)

    def set_overlay(self, overlay):
        self.overlay = overlay
        if not self.base.isNull():
            pm = QtGui.QPixmap(self.base.size())
        else:
            pm = QtGui.QPixmap(overlay.size())
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter()
        painter.begin(pm)
        # Draw overlay
        if not self.base.isNull():
            painter.drawPixmap(0, 0, self.base)
        if not self.roi.isNull():
            painter.drawPixmap(0, 0, self.roi)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        self.setPixmap(pm)

    def set_roi(self, roi):
        self.roi = roi
        pm = QtGui.QPixmap(self.base.size())
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter()
        painter.begin(pm)
        if not self.base.isNull():
            painter.drawPixmap(0, 0, self.base)
        painter.drawPixmap(0, 0, roi)
        if not self.overlay.isNull():
            painter.drawPixmap(0, 0, self.overlay)
        painter.end()
        self.setPixmap(pm)

    @QtCore.Slot(QtGui.QPixmap)
    def on_base_pixmap(self, base):
        self.set_base(base)

    @QtCore.Slot(float)
    def on_alpha(self, alpha):
        if self.drr is not None:
            self.alpha = alpha
            overlay = ImageWidget.np_to_qrgb_pixmap(1 - self.drr, 'r', alpha)
            self.set_overlay(overlay)

    @QtCore.Slot(np.ndarray)
    def on_drr(self, drr):
        self.drr = drr
        overlay = self.np_to_qrgb_pixmap(1 - drr, 'r', self.alpha)
        self.set_overlay(overlay)

    @QtCore.Slot(float, float, float, float)
    def on_roi_select(self, a, b, c, d):
        """on_roi_select

        :param a: Top left x
        :param b: Top left y
        :param c: Bottom right x
        :param d: Bottoom right y
        """

        w, h = self.base.size().toTuple()
        roi_arr = np.ones((h, w))
        start_dim1, end_dim1 = min(int(b*h), int(d*h)), max(int(b*h), int(d*h))
        start_dim2, end_dim2 = min(int(a*w), int(c*w)), max(int(a*w), int(c*w))
        roi_arr[start_dim1:end_dim1, start_dim2:end_dim2] = 0
        roi_arr = roi_arr.astype(np.uint8).flatten(order='C')
        colortable = [QtGui.qRgba(0, 0, 0, int(255*x)) for x in range(2)]
        img = QtGui.QImage(
            roi_arr,
            w, h,
            w*np.nbytes[np.uint8],
            QtGui.QImage.Format_Indexed8
        )
        img.setColorTable(colortable)
        roi = QtGui.QPixmap.fromImage(img)
        self.set_roi(roi)

    def mousePressEvent(self, event):
        w, h = self.frameSize().toTuple()
        x, y = event.x(), event.y()
        self.clicked.emit(x/w, y/h)
        QtWidgets.QLabel.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.released.emit()
        QtWidgets.QLabel.mouseReleaseEvent(self, event)

    def mouseMoveEvent(self, event):
        w, h = self.frameSize().toTuple()
        x, y = event.x(), event.y()
        self.moved.emit(min(max(x/w, 0.0), 1.0), min(max(y/h, 0.0), 1.0))
        QtWidgets.QLabel.mouseMoveEvent(self, event)

    @QtCore.Slot(float, float)
    def on_clicked(self, x, y):
        if 0 <= x <= 1 and 0 <= y <= 1 and self.roi_enabled:
            self.a, self.b = x, y
            self.is_selecting_roi = True

    @QtCore.Slot(float, float)
    def on_moved(self, x, y):
        if self.roi_enabled and self.is_selecting_roi and abs(x-self.a) > 0.05 and abs(y - self.b) > 0.05:
            self.c, self.d = x, y
            self.roi_select.emit(self.a, self.b, self.c, self.d)

    @QtCore.Slot()
    def on_released(self):
        if self.roi_enabled and self.is_selecting_roi:
            self.is_selecting_roi = False
            self.roi_finalize.emit(self.a, self.b, self.c, self.d)

    @QtCore.Slot(bool)
    def on_enable_roi(self, flag):
        self.roi_enabled = flag


class ParametersWidget(QtWidgets.QWidget):
    params_edited = QtCore.Signal()

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
        # Connect change signal
        self.tx_widg.editingFinished.connect(self.on_editingfinish)
        self.ty_widg.editingFinished.connect(self.on_editingfinish)
        self.tz_widg.editingFinished.connect(self.on_editingfinish)
        self.rx_widg.editingFinished.connect(self.on_editingfinish)
        self.ry_widg.editingFinished.connect(self.on_editingfinish)
        self.rz_widg.editingFinished.connect(self.on_editingfinish)
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

    @QtCore.Slot(float)
    def on_editingfinish(self):
        self.params_edited.emit()

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
    base_pixmap_3 = QtCore.Signal(QtGui.QPixmap)
    base_pixmap_4 = QtCore.Signal(QtGui.QPixmap)
    drr1 = QtCore.Signal(np.ndarray)
    drr2 = QtCore.Signal(np.ndarray)
    drr3 = QtCore.Signal(np.ndarray)
    drr4 = QtCore.Signal(np.ndarray)
    alpha = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('DRR Viewer')
        self.setMinimumSize(QtCore.QSize(800, 720))
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        self.central_widg = QtWidgets.QWidget(self)
        # File dialog
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setViewMode(QtWidgets.QFileDialog.List)
        self.file_dialog_out = ''
        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu('File')
        self.fd_in_out = {
            'CT': [self.on_ct_menu, ''],
            'Camera Files': [self.on_cam_menu, ''],
            'C-Arm Images': [self.on_carm_menu, '']
        }
        for entry in ['CT', 'Camera Files', 'C-Arm Images']:
            action = QtWidgets.QAction('Open {}'.format(entry), self)
            action.triggered.connect(self.fd_in_out[entry][0])
            self.file_menu.addAction(action)
        save_drr1_action = QtWidgets.QAction('Save first DRR as ...', self)
        save_drr1_action.triggered.connect(self.on_save_drr1)
        self.file_menu.addAction(save_drr1_action)
        save_drr2_action = QtWidgets.QAction('Save second DRR as ...', self)
        save_drr2_action.triggered.connect(self.on_save_drr2)
        self.file_menu.addAction(save_drr2_action)
        save_params_action = QtWidgets.QAction('Save parameters as ...', self)
        save_params_action.triggered.connect(self.on_save_params)
        self.file_menu.addAction(save_params_action)
        load_params_action = QtWidgets.QAction('Load parameters from ...', self)
        load_params_action.triggered.connect(self.on_load_params)
        self.file_menu.addAction(load_params_action)
        save_setup_action = QtWidgets.QAction('Save current setup as ...', self)
        save_setup_action.triggered.connect(self.on_save_setup)
        self.file_menu.addAction(save_setup_action)
        self.edit_menu = self.menu.addMenu('Edit')
        # Status bar
        self.status_bar = self.statusBar()
        # Images
        self.img1_widg = ImageWidget(self.central_widg)
        self.img2_widg = ImageWidget(self.central_widg)
        self.img3_widg = ImageWidget(self.central_widg)
        self.img4_widg = ImageWidget(self.central_widg)
        # Parameter widgs
        self.params_widg = ParametersWidget(self.central_widg)
        self.params_widg.params_edited.connect(self.on_refresh)
        # Threshold widg
        self.threshold_widg = ThresholdWidget(self.central_widg)
        # Refresh button
        self.refresh_butn = QtWidgets.QPushButton('Refresh', self)
        # Recenter button
        self.recenter_widg = RecenterWidget(self)
        # Layout
        refresh_layout = QtWidgets.QVBoxLayout()
        refresh_layout.addWidget(self.refresh_butn)
        refresh_layout.addWidget(self.recenter_widg)
        # TODO: Add Labels (AP, LAT, left, right)
        # TODO: Require at least AP and LAT
        # TODO: 4x4 grid layout
        left_imgs_layout = QtWidgets.QVBoxLayout()
        left_imgs_layout.addWidget(self.img1_widg)
        left_imgs_layout.addWidget(self.img3_widg)
        right_imgs_layout = QtWidgets.QVBoxLayout()
        right_imgs_layout.addWidget(self.img2_widg)
        right_imgs_layout.addWidget(self.img4_widg)
        # layout = QtWidgets.QGridLayout()
        # layout.addLayout(left_imgs_layout, 0, 0, 1, 2)
        # layout.addLayout(right_imgs_layout, 0, 3, 1, 2)
        # layout.addWidget(self.threshold_widg, 0, 2, 1, 1)
        # layout.addWidget(self.params_widg, 1, 0, 1, 3)
        # layout.addLayout(refresh_layout, 1, 3, 1, 2)
        # layout.setRowStretch(0, 1)
        layout = QtWidgets.QVBoxLayout()
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addLayout(left_imgs_layout, 1)
        top_layout.addWidget(self.threshold_widg, 0)
        top_layout.addLayout(right_imgs_layout, 1)
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.params_widg, 1)
        bottom_layout.addLayout(refresh_layout, 0)
        layout.addLayout(top_layout, 2)
        layout.addLayout(bottom_layout, 0)
        self.central_widg.setLayout(layout)
        self.setCentralWidget(self.central_widg)
        # Focus
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        # Connect signals and slots
        self.base_pixmap_1.connect(self.img1_widg.on_base_pixmap)
        self.base_pixmap_2.connect(self.img2_widg.on_base_pixmap)
        self.base_pixmap_3.connect(self.img3_widg.on_base_pixmap)
        self.base_pixmap_4.connect(self.img4_widg.on_base_pixmap)
        self.params_widg.alpha_slider.valueChanged.connect(self.on_alphaslider_update)
        self.refresh_butn.released.connect(self.on_refresh_btn)
        self.threshold_widg.new_threshold.connect(self.on_new_threshold)
        self.recenter_widg.new_center.connect(self.on_new_center)
        self.alpha.connect(self.img1_widg.on_alpha)
        self.alpha.connect(self.img2_widg.on_alpha)
        self.alpha.connect(self.img3_widg.on_alpha)
        self.alpha.connect(self.img4_widg.on_alpha)
        self.drr1.connect(self.img1_widg.on_drr)
        self.drr2.connect(self.img2_widg.on_drr)
        self.drr3.connect(self.img3_widg.on_drr)
        self.drr4.connect(self.img4_widg.on_drr)
        self.img1_widg.roi_finalize.connect(self.on_roi_finalize_1)
        self.img2_widg.roi_finalize.connect(self.on_roi_finalize_2)
        self.img3_widg.roi_finalize.connect(self.on_roi_finalize_3)
        self.img4_widg.roi_finalize.connect(self.on_roi_finalize_4)
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
        self.autorefresh = False
        autorefresh_action = QtWidgets.QAction('Auto refresh', self)
        autorefresh_action.setCheckable(True)
        autorefresh_action.toggled.connect(self.on_autorefresh_toggle)
        self.edit_menu.addAction(autorefresh_action)
        roiselection_action = QtWidgets.QAction('Enable ROI selection', self)
        roiselection_action.setCheckable(True)
        roiselection_action.toggled.connect(self.img1_widg.on_enable_roi)
        roiselection_action.toggled.connect(self.img2_widg.on_enable_roi)
        roiselection_action.toggled.connect(self.img3_widg.on_enable_roi)
        roiselection_action.toggled.connect(self.img4_widg.on_enable_roi)
        self.edit_menu.addAction(roiselection_action)

    @QtCore.Slot(bool)
    def on_autorefresh_toggle(self, checked):
        if checked:
            self.autorefresh = True
        else:
            self.autorefresh = False

    @QtCore.Slot()
    def on_save_setup(self):
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.file_dialog.setNameFilter('Any files (*)')
        if self.file_dialog.exec_():
            fpath = self.file_dialog.selectedFiles()[0]
            fpath_params = '{}.txt'.format(fpath)
            fpath_drr1 = '{}_drr1.png'.format(fpath)
            fpath_drr2 = '{}_drr2.png'.format(fpath)
            fpath_drr3 = '{}_drr3.png'.format(fpath)
            fpath_drr4 = '{}_drr4.png'.format(fpath)
            with open(fpath_params, 'w') as f:
                params_str = 'Tx = {:.4f}\nTy = {:.4f}\nTz = {:.4f}\nRx = {:.4f}\nRy = {:.4f}\nRz = {:.4f}'.format(
                    self.drr_set.params[0],
                    self.drr_set.params[1],
                    self.drr_set.params[2],
                    self.drr_set.params[3],
                    self.drr_set.params[4],
                    self.drr_set.params[5]
                )
                f.write(params_str)
            plt.imsave(fpath_drr1, self.img1_widg.drr, cmap='gray', vmin=0, vmax=1)
            plt.imsave(fpath_drr2, self.img2_widg.drr, cmap='gray', vmin=0, vmax=1)
            if self.img3_widg.drr is not None:
                plt.imsave(fpath_drr3, self.img3_widg.drr, cmap='gray', vmin=0, vmax=1)
            if self.img4_widg.drr is not None:
                plt.imsave(fpath_drr4, self.img4_widg.drr, cmap='gray', vmin=0, vmax=1)

    @QtCore.Slot()
    def on_save_params(self):
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.file_dialog.setNameFilter('Text Files (*.txt)')
        if self.file_dialog.exec_():
            fpath = self.file_dialog.selectedFiles()[0]
            with open(fpath, 'w') as f:
                params_str = 'Tx = {:.4f}\nTy = {:.4f}\nTz = {:.4f}\nRx = {:.4f}\nRy = {:.4f}\nRz = {:.4f}'.format(
                    self.drr_set.params[0],
                    self.drr_set.params[1],
                    self.drr_set.params[2],
                    self.drr_set.params[3],
                    self.drr_set.params[4],
                    self.drr_set.params[5]
                )
                f.write(params_str)

    @QtCore.Slot()
    def on_load_params(self):
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        self.file_dialog.setNameFilter('Text Files (*.txt)')
        if self.file_dialog.exec_():
            fpath = self.file_dialog.selectedFiles()[0]
            with open(fpath, 'r') as f:
                s = f.read()
                tx = float(re.search('[Tt][Xx]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
                ty = float(re.search('[Tt][Yy]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
                tz = float(re.search('[Tt][Zz]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
                rx = float(re.search('[Rr][Xx]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
                ry = float(re.search('[Rr][Yy]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
                rz = float(re.search('[Rr][Zz]\s*=\s*([-+]?[0-9]*\.?[0-9]+)', s).group(1))
            self.params_widg.set_params(tx, ty, tz, rx, ry, rz)

    @QtCore.Slot()
    def on_save_drr1(self):
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.file_dialog.setNameFilter('Image Files (*.png)')
        if self.file_dialog.exec_():
            fpath = self.file_dialog.selectedFiles()[0]
            plt.imsave(fpath, self.img1_widg.drr, cmap='gray', vmin=0, vmax=1)

    @QtCore.Slot()
    def on_save_drr2(self):
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.file_dialog.setNameFilter('Image Files (*.png)')
        if self.file_dialog.exec_():
            fpath = self.file_dialog.selectedFiles()[0]
            plt.imsave(fpath, self.img2_widg.drr, cmap='gray', vmin=0, vmax=1)

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
        for idx, cam in enumerate(self.drr_set.cams, 1):
            print('Cam', idx)
            print(cam.m)

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
        self.file_dialog.setNameFilter('Any files (*)')
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['CT'][1] = fpaths
            self.set_rho(fpaths)

    @QtCore.Slot()
    def on_cam_menu(self):
        self.file_dialog.setNameFilter('Text Files (*.txt)')
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['Camera Files'][1] = fpaths
            print(fpaths)
            self.init_cams_from_path(fpaths)

    @QtCore.Slot()
    def on_carm_menu(self):
        self.file_dialog.setNameFilter('Image Files (*.png, *.bmp)')
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['C-Arm Images'][1] = fpaths
            xrays = []
            for idx, fpath in enumerate(fpaths, 1):
                xrays.append(read_image_as_np(fpath))
                signal = getattr(self, 'base_pixmap_{:d}'.format(idx))
                signal.emit(QtGui.QPixmap(fpath))
            self.drr_registration.set_xrays(*xrays)

    @QtCore.Slot()
    def on_refresh(self):
        if self.autorefresh:
            params = self.params_widg.get_params()
            self.drr_set.set_tfm_params(*params)
            self.draw_drrs()

    @QtCore.Slot()
    def on_refresh_btn(self):
        params = self.params_widg.get_params()
        self.drr_set.set_tfm_params(*params)
        self.draw_drrs()

    def draw_drrs(self):
        drrs = self.raybox.trace_rays()
        assert len(drrs) < 5
        for idx, drr in enumerate(drrs, 1):
            signal = getattr(self, 'drr{:d}'.format(idx))
            signal.emit(drr)

    def set_rho(self, fpaths):
        rho, sp = read_rho(fpaths[0])
        self.raybox.set_rho(rho, sp)
        print('Set Rho')

    def init_cams_from_path(self, fpaths):
        cams = []
        for fpath in fpaths:
            with open(fpath) as f:
                s = f.read()
            m = str_to_mat(re.search('[Mm]\s*=\s*\[(.*)\]', s).group(1))
            k = str_to_mat(re.search('[Kk]\s*=\s*\[(.*)\]', s).group(1))
            h = int(re.search('[Hh]\s*=\s*([0-9]+)', s).group(1))
            w = int(re.search('[Ww]\s*=\s*([0-9]+)', s).group(1))
            cams.append(Camera(m=m, k=k, h=h, w=w))
        # TODO: Add to drr_set
        self.drr_set.set_cams(*cams)
        print('Set cams')

    @QtCore.Slot(float, float, float, float)
    def on_roi_finalize_1(self, a, b, c, d):
        self.drr_registration.mask1 = (a, b, c, d)

    @QtCore.Slot(float, float, float, float)
    def on_roi_finalize_2(self, a, b, c, d):
        self.drr_registration.mask2 = (a, b, c, d)

    @QtCore.Slot(float, float, float, float)
    def on_roi_finalize_3(self, a, b, c, d):
        self.drr_registration.mask3 = (a, b, c, d)

    @QtCore.Slot(float, float, float, float)
    def on_roi_finalize_4(self, a, b, c, d):
        self.drr_registration.mask4 = (a, b, c, d)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
