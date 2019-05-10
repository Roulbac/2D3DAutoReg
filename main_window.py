import sys
from PySide2 import QtCore, QtGui, QtWidgets


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
        try:
            threshold = float(self.input_box.text())
        except ValueError:
            return
        print(threshold)
        self.new_threshold.emit(threshold)

class ImageWidget(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setScaledContents(True)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

    # @QtCore.Slot(QtGui.QPixmap)
    # def set_pixel(self, pixmap):
    #     self.setPixmap(pixmap.scaled(self.width(), self.height(), QtCore.Qt.IgoreAspectRatio))
    #     self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
    #                        QtWidgets.QSizePolicy.Ignored)

class ParametersWidget(QtWidgets.QWidget):
    new_params = QtCore.Signal(list)

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
        self.rx_widg = QtWidgets.QDoubleSpinBox(self)
        self.ry_widg = QtWidgets.QDoubleSpinBox(self)
        self.rz_widg = QtWidgets.QDoubleSpinBox(self)
        self.tx_widg.setKeyboardTracking(False)
        self.ty_widg.setKeyboardTracking(False)
        self.tz_widg.setKeyboardTracking(False)
        self.rx_widg.setKeyboardTracking(False)
        self.ry_widg.setKeyboardTracking(False)
        self.rz_widg.setKeyboardTracking(False)
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

    @QtCore.Slot()
    def on_spinbox_update(self):
        params = [self.tx_widg.value(),
                  self.ty_widg.value(),
                  self.tz_widg.value(),
                  self.rx_widg.value(),
                  self.ry_widg.value(),
                  self.rz_widg.value()]
        self.new_params.emit(params)

class MainWindow(QtWidgets.QMainWindow):
    input_ct_sig = QtCore.Signal(list)
    input_cams_sig = QtCore.Signal(list)
    input_xray_sig = QtCore.Signal(list)

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
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.triggered.connect(self.exit_app)
        self.file_menu.addAction(exit_action)
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
        runoptim_action.triggered.connect(self.start_optim)
        self.edit_menu.addAction(runoptim_action)
        popup_3d_action = QtWidgets.QAction('Pop up 3D visualizer', self)
        popup_3d_action.triggered.connect(self.popup_3d_viewer)
        self.edit_menu.addAction(popup_3d_action)
        # Status bar
        self.status_bar = self.statusBar()
        # Images
        self.img1 = ImageWidget(self.central_widg)
        self.img2 = ImageWidget(self.central_widg)
        # Parameter widgs
        self.params_widg = ParametersWidget(self.central_widg)
        # Threshold widg
        self.threshold_widg = ThresholdWidget(self.central_widg)
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.img1, 0, 0)
        layout.addWidget(self.img2, 0, 2)
        layout.addWidget(self.threshold_widg, 0, 1)
        layout.addWidget(self.params_widg, 1, 0, 1, 3)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 1)
        self.central_widg.setLayout(layout)
        self.setCentralWidget(self.central_widg)
        # Focus
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        # Connect signals and slots
        self.params_widg.new_params.connect(self.on_new_params)

    @QtCore.Slot(list)
    def on_new_params(self, params):
        pass
        # TODO
        # Send params to backend
        # Render DRRs
        # Emit signals to update DRR windows

    @QtCore.Slot()
    def on_ct_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['CT'][1] = fpaths
            self.input_ct_sig.emit(fpaths)

    @QtCore.Slot()
    def on_cam_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['Camera Files'][1] = fpaths
            self.input_cams_sig.emit(fpaths)

    @QtCore.Slot()
    def on_xray_menu(self):
        if self.file_dialog.exec_():
            fpaths = self.file_dialog.selectedFiles()
            self.fd_in_out['X-Ray Files'][1] = fpaths
            self.img1.setPixmap(QtGui.QPixmap(fpaths[0]))
            self.img2.setPixmap(QtGui.QPixmap(fpaths[1]))
            self.input_xray_sig.emit(fpaths)

    @QtCore.Slot()
    def start_optim(self):
        # TODO
        pass

    @QtCore.Slot()
    def popup_3d_viewer(self):
        # TODO
        pass

    @QtCore.Slot()
    def exit_app(self, checked):
        sys.exit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

