from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import pandas as pd
<<<<<<< HEAD
from LocalModelCommunication import LocalModelCommunication
=======
>>>>>>> master


class APP(object):
    """
    A desktop application for doctor.
    """
    def __init__(self, root):
        self.root = root
        self.fname = None
        self.data = None
        self.p_data = StringVar()
        self.p_data.set('No patient data. \nPress Import button to load patient data.')
        self.p_label = None
        self.rad_v = IntVar()
        self.rad_v.set(0)
<<<<<<< HEAD
        #self.loca = loca
        self.prediction_result = StringVar()
        self.prediction_result.set('nothing yet')
=======
>>>>>>> master

        self.setup()

    def setup(self):
        # setup frames
        p_data_label = ttk.LabelFrame(self.root, text='Patient Data', width=320, height=150)
        p_data_label.pack_propagate(0)
        p_data_label.grid(row=0, column=0, rowspan=2, columnspan=6, padx=20, pady=5)
        self.p_label = ttk.Label(p_data_label, textvariable=self.p_data)
        self.p_label.pack()

        Button(self.root, text='Import', command=self.import_data, width=16).grid(row = 2, column=1, padx=10, pady=5)
        Button(self.root, text='Predict', command=self.predict_data, width=16).grid(row = 2, column=4, padx=10, pady=5)

        feedback_label = ttk.LabelFrame(self.root, text='Doctor Feedback', width=320, height=240)
        feedback_label.pack_propagate(0)
        feedback_label.grid(row=3, column=0, rowspan=2, columnspan=6, padx=20, pady=5)
        # set Radiobutton for doctor diagnosis
        Radiobutton(feedback_label, variable=self.rad_v, text='Class 0', value=0, command=self.radCall).pack()
        Radiobutton(feedback_label, variable=self.rad_v, text='Class 1', value=1, command=self.radCall).pack()
        d_data_ = ttk.Label(feedback_label, text='TODO')
        d_data_.pack()
        # set Entry for doctor suggestion
        e = Entry(feedback_label)
        e.pack()

        Button(self.root, text='Patient Report', command=self.gen_report, width=16).grid(row=5,column=1, padx=10, pady=5)
        Button(self.root, text='Send Feedback', command=self.send_feedback, width=16).grid(row=5,column=4, padx=10, pady=5)

        prediction_label = ttk.LabelFrame(self.root, text='Prediction Report', width=320, height=480)
        prediction_label.pack_propagate(0)
        prediction_label.grid(row=0, column=6, rowspan=6, columnspan=6, padx=10, pady=5)
<<<<<<< HEAD
        d_data_ = ttk.Label(prediction_label, textvariable=self.prediction_result)
        d_data_.pack()

    def showResult(self, result):
        self.prediction_label.set(result)

=======
        d_data_ = ttk.Label(prediction_label, text='TODO')
        d_data_.pack()

>>>>>>> master
    # Callback functions
    def radCall(self):
        """Deal with doctor diagnosis"""
        print(self.rad_v.get())
    def import_data(self):
        """ Refresh doctor report and load csv from file system"""
        self.fname = fd.askopenfilename(filetypes=[(".csv file", ".csv")])
        self.data = pd.read_csv(self.fname, delimiter=',')

        self.p_data.set(str(self.data))
        self.p_label.pack()
    def predict_data(self):
        """Send patient data for prediction"""
        pass
    def gen_report(self):
        """Generate patient report"""
        pass
    def send_feedback(self):
        """Send doctor feedback"""
        pass


if __name__ == '__main__':
    root = Tk()
    root.geometry("700x500")
    root.title("Doctor Application")
    root.resizable(width=False, height=False)

    app = APP(root)

<<<<<<< HEAD
    root.mainloop()
=======
    root.mainloop()
>>>>>>> master
