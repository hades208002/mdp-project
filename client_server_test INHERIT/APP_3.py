from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import pandas as pd
from LocalModelCommunication import LocalModelCommunication
from PIL import ImageTk, Image
import matplotlib as plt



class APP(LocalModelCommunication):
    """
    A desktop application for doctor.
    """
    def __init__(self, root):
        LocalModelCommunication.__init__(self, data="none", target_name="AFclass", model_name="dt4",random_state=12345678, imputation_strategy='mice', balance_strategy='SMOTE',local_models=[])
        self.root = root
        self.fname = None
        self.data = None
        self.p_data = StringVar()
        self.p_data.set('No patient data. \nPress Import button to load patient data.\nPress Predict button to get prediction.')
        self.p_label = None
        self.rad_v = IntVar()
        self.rad_v.set(0)
        self.prediction_result = StringVar()
        self.prediction_result.set('nothing yet')

        self.setup()

    def setup(self):
        # setup frames
        p_data_label = ttk.LabelFrame(self.root, text='Patient Data', width=320, height=150)
        p_data_label.pack_propagate(0)
        p_data_label.grid(row=0, column=0, rowspan=2, columnspan=6, padx=20, pady=5)
        self.p_label = ttk.Label(p_data_label, textvariable=self.p_data)
        self.p_label.pack()

        Button(self.root, text='Import', command=self.import_data, width=16).grid(row = 2, column=1, padx=10, pady=5)
        Button(self.root, text='Predict', command=self.sendRequestToPredict, width=16).grid(row = 2, column=4, padx=10, pady=5)

        feedback_label = ttk.LabelFrame(self.root, text='Doctor Feedback', width=320, height=240)
        feedback_label.pack_propagate(0)
        feedback_label.grid(row=3, column=0, rowspan=2, columnspan=6, padx=20, pady=5)
        # set Radiobutton for doctor diagnosis
        Radiobutton(feedback_label, variable=self.rad_v, text='Stage 1', value=0, command=self.radCall).pack()
        Radiobutton(feedback_label, variable=self.rad_v, text='Stage 2', value=1, command=self.radCall).pack()
        d_data_ = ttk.Label(feedback_label, text='Suggestion???')
        d_data_.pack()
        # set Entry for doctor suggestion
        self.e = Text(feedback_label, height=280, width=200)
        self.e.pack()

        Button(self.root, text='Patient Report', command=self.gen_report, width=16).grid(row=5,column=1, padx=10, pady=5)
        Button(self.root, text='Send Feedback', command=self.send_feedback, width=16).grid(row=5,column=4, padx=10, pady=5)

        self.prediction_label = ttk.LabelFrame(self.root, text='Central Server Connected', width=320, height=480)
        self.prediction_label.pack_propagate(0)
        self.prediction_label.grid(row=0, column=6, rowspan=6, columnspan=6, padx=10, pady=5)
        self.d_data_ = ttk.Label(self.prediction_label, textvariable=self.prediction_result)
        #self.d_data_.pack()
        self.prediction_img = ttk.Label(self.prediction_label)
        self.img = ImageTk.PhotoImage(Image.open('wellwave.png'))
        self.prediction_img.config(image=self.img)
        self.prediction_img.pack()


        self.connectToCentral()



    def connectionBegun(self):
        print("connectionBegun")

    def connectionFailed(self):
        print ("connectionFailed")

    def predictionReceived(self):
        print ("prediction results RECEIVED -> stored in self.predictedData")
        print (self.predictedData)
        self.finalPrediction = self.predictedData['finalPrediction'][0]
        self.localPredictions = self.predictedData.drop(["finalPrediction"], axis=1)
        self.showResult()

    def showResult(self):
        df = pd.DataFrame(columns=["model","prediction"])
        df["model"] = list(self.localPredictions.columns)
        df["prediction"] = list(self.localPredictions.loc[0])

        df = df.replace(1, 2)
        df = df.replace(0, 1)

        self.prediction_result.set("Final Prediction: "+str(self.finalPrediction+1))
        self.prediction_img.pack_forget()
        self.d_data_.pack()

        """
        self.prediction_pie = ttk.Label(self.prediction_label)
        ax = df.plot.barh(y="prediction", x="model",figsize=(3.3, 4.5),rot=0,fontsize=5, title="Local Models' Predictions")
        plt.pyplot.savefig('prediction_pie.png')
        self.img_pie = ImageTk.PhotoImage(Image.open('prediction_pie.png'))
        self.prediction_pie.config(image=self.img_pie)
        self.prediction_pie.pack()
        """
        #self.prediction_pie = ttk.Label(self.prediction_label)
        print(str(df))
        df.groupby("prediction")["model"].count().plot.pie(figsize=(3.3, 3.3))
        print(str(df.groupb
        y("prediction")["model"].count()))

        plt.pyplot.savefig('prediction_pie.png')
        self.img_pie = ImageTk.PhotoImage(Image.open('prediction_pie.png'))
        self.prediction_img.config(image=self.img_pie)
        self.prediction_img.pack()

        # TODO
        self.feature_importance = ttk.Label(self.prediction_label)



    def sendDataCompleted(self):
        print ("ALL THE DATA (for central training) HAS BEEN SENT ")


    # Callback functions
    def radCall(self):
        """Deal with doctor diagnosis"""
        pass
        #print(self.rad_v.get())

    def import_data(self):
        """ Refresh doctor report and load csv from file system"""
        self.fname = fd.askopenfilename(filetypes=[(".csv file", ".csv")])
        self.data = pd.read_csv(self.fname, delimiter=',')

        self.p_data.set(str(self.data))
        self.p_label.pack()
        self.prediction_label.config(text='Prediction Report')
        self.d_data_.pack_forget()
        self.img = ImageTk.PhotoImage(Image.open('wellwave.png'))
        self.prediction_img.config(image=self.img)
        self.prediction_img.pack()


    def sendRequestToPredict(self):
        """Send patient data for prediction"""
        self.requestPrediction(self.data)

    def gen_report(self):
        """Generate patient report"""
        #print(self.e.get(1.0, END))
        if self.rad_v.get() == 0:
            self.img = ImageTk.PhotoImage(Image.open('stage0.png'))
            self.prediction_img.config(image=self.img)
            self.prediction_img.pack()
        else:
            self.img = ImageTk.PhotoImage(Image.open('stage1.png'))
            self.prediction_img.config(image=self.img)
            self.prediction_img.pack()
        self.d_data_.pack_forget()
        self.prediction_label.config(text='Patient Report')

        pass
    def send_feedback(self):
        """Send doctor feedback"""
        feedback = self.data
        feedback['AFclass'] = self.rad_v.get()
        self.sendDataForCentralTraining(feedback)



if __name__ == '__main__':
    root = Tk()
    root.geometry("700x500")
    root.title("Doctor Application")
    root.resizable(width=False, height=False)

    app = APP(root)

    root.mainloop()
