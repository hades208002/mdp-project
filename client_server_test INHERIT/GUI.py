from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import pandas as pd
#import LocalModelCommunication


class GUI(object):
    def __init__(self):
        # overall
        self.tabControl = None
        self.tab_step1 = None
        self.tab_step2 = None
        self.tab_step3 = None
        self.tab_step4 = None
        self.dataframe = None
        self.img_wait = PhotoImage(file='test.GIF')

        # 1 step
        self.fname = None
        self.data = None
        self.features = None
        self.import_lable = None
        self.import_label_text = StringVar()
        self.import_label_text.set(' ')

        # 2 step
        self.required = ['RR', 'QTm_old', 'sbjBeatConsidered', 'numRRaveraged', 'QR', 'QTn', 'QRS', 'IPG',
                                    'PQ', 'PCpos', 'PCneg', 'patsex', 'AFclass', 'Age']
        self.required_ordered = []
        i = 0
        for item in self.required:
            self.required_ordered.append(str(i) + ': ' + item)
            i = i + 1
        self.leftbox = StringVar()
        self.rightbox = StringVar()
        self.rrightbox = StringVar()
        self.list_left = None
        self.list_right = None
        self.list_rright = None

        # 3 step
        self.model_label = None
        self.model_label_text = StringVar()
        self.model_label_text.set('Waiting for model training...')
        self.img_gif = None

        # 4 step
        self.connect_label = None
        self.connect_label_text = StringVar()
        self.connect_label_text.set('Waiting for central server response...')

        # 5 step

    # help functions
    def add_tab(self, tabControl, tab_name):
        tab = ttk.Frame(tabControl)  # Create a tab
        tabControl.add(tab, text=tab_name)
        return tab

    # Callback functions
    ## step 1
    def get_csv(self):  # open file system
        self.fname = fd.askopenfilename(filetypes=[(".csv file", ".csv")])
        self.data = pd.read_csv(self.fname, delimiter=',')
        self.features = self.data.columns

        self.import_label_text.set('Import data from: ' + self.fname + '\n' + str(self.features))
        self.import_lable.pack(side=TOP)
    def go_next_step2(self):
        self.tab_step2 = self.add_tab(self.tabControl, "Step 2: Match Features")
        self.tab_match(self.tab_step2)
        self.tabControl.select(self.tab_step2)
        self.tabControl.forget(self.tab_step1)
    ## step 2
    def move_to_right(self):

        self.list_right.insert(END,
                               str(self.list_right.size()) + ': ' + self.list_left.get(self.list_left.curselection()))
        self.list_left.delete(self.list_left.curselection())
    def move_to_left(self):
        content = self.list_right.get(self.list_right.curselection())
        contents = content.split(': ')
        self.list_left.insert(END, contents[1])
        self.list_right.delete(self.list_right.curselection())
    def add_nan(self):
        self.list_right.insert(END, str(self.list_right.size()) + ': ' + 'NAN')
    def go_next_step3(self):
        # prepare dataframe for localmodel
        columns = []
        contents = self.rightbox.get()
        contents = contents.replace('(', '')
        contents = contents.replace(')', '')
        contents = contents.replace("'", '')
        item_list = contents.split(', ')
        for item in item_list:
            content = item.split(': ')[1]
            if content != 'NAN':
                columns.append(content)

        self.dataframe = self.data[columns]
        print(self.dataframe.head(2))
        self.tab_step3 = self.add_tab(self.tabControl, "Step 3: Train Model")
        # render tab3
        self.tab_model(self.tab_step3)
        self.tabControl.select(self.tab_step3)
        self.tabControl.forget(self.tab_step2)
    def go_back_step1(self):
        self.tab_step1 = self.add_tab(self.tabControl, "Step 1: Import Data")
        # render tab1
        self.tab_import(self.tab_step1,  self.tabControl)
        self.tabControl.select(self.tab_step1)
        self.tabControl.forget(self.tab_step2)
    ## step 3
    def go_next_step4(self):
        self.tab_step4 = self.add_tab(self.tabControl, "Step 4: Connect to Central Server")
        # render tab4
        self.tab_connect(self.tab_step4)
        self.tabControl.select(self.tab_step4)
        self.tabControl.forget(self.tab_step3)
    def go_back_step2(self):
        self.tab_step2 = self.add_tab(self.tabControl, "Step 2: Match Features")
        # render tab2
        self.tab_match(self.tab_step2)
        self.tabControl.select(self.tab_step2)
        self.tabControl.forget(self.tab_step3)
    ## step 4
    def go_next_step5(self):
        self.tab_step5 = self.add_tab(self.tabControl, "Step 5: Wait for Prediction Call")
        # render tab5
        self.tab_wait(self.tab_step5)
        self.tabControl.select(self.tab_step5)
        self.tabControl.forget(self.tab_step4)
    def go_back_step3(self):
        self.tab_step3 = self.add_tab(self.tabControl, "Step 3: Train Model")
        # render tab3
        self.tab_model(self.tab_step3)
        self.tabControl.select(self.tab_step3)
        self.tabControl.forget(self.tab_step4)
    ## step 5

    # frames
    def tab_import(self, root, tabControl):
        """
        Load local data (csv file)
        """
        self.tabControl = tabControl
        self.tab_step1 = root

        frame = Frame(root)
        frame.pack(side=TOP)
        Button(frame, text='Import Data', command=self.get_csv, width=16).pack(side=TOP)
        label_frame = ttk.LabelFrame(frame, text='Press Button to Import Data')
        label_frame.pack(side=TOP)
        self.import_lable = ttk.Label(label_frame, textvariable=self.import_label_text)
        self.import_lable.pack(side=TOP)

        frame = Frame(root)
        frame.pack(side=BOTTOM)
        Button(frame, text='Next>>', command=self.go_next_step2, width=16).pack(side=TOP)

    def tab_match(self, root):
        """
        Feature matching
        """
        self.leftbox.set(sorted(self.features))
        self.rightbox.set('')
        self.rrightbox.set(self.required_ordered)

        frame = Frame(root)
        frame.pack(side=BOTTOM)
        Button(frame, text='Next>>', command=self.go_next_step3, width=16).pack(side=RIGHT)
        Button(frame, text='<<Back', command=self.go_back_step1, width=16).pack(side=LEFT)

        frame = Frame(root)
        frame.pack(side=LEFT)
        column_head = ttk.Label(frame, text='Local Features')
        column_head.pack(side=TOP)
        self.list_left = Listbox(frame, listvariable=self.leftbox, width=25, height=20)
        self.list_left.pack(side=LEFT)

        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=self.list_left.yview)
        scrollbar.pack(side="right", fill="y")

        frame = Frame(root)
        frame.pack(side=LEFT)
        Button(frame, text='->', command=self.move_to_right, width=7).pack(side=TOP)
        Button(frame, text='<-', command=self.move_to_left, width=7).pack(side=TOP)
        Button(frame, text='NAN', command=self.add_nan, width=7).pack(side=TOP)

        frame = Frame(root)
        frame.pack(side=LEFT)
        column_head = ttk.Label(frame, text='Matched Features')
        column_head.pack(side=TOP)
        self.list_right = Listbox(frame, listvariable=self.rightbox,height=20, width=25)
        self.list_right.pack(side=LEFT)

        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=self.list_right.yview)
        scrollbar.pack(side="right", fill="y")

        frame = Frame(root)
        frame.pack(side=RIGHT)
        column_head = ttk.Label(frame, text='Required Features')
        column_head.pack(side=TOP)
        self.list_rright = Listbox(frame, listvariable=self.rrightbox,height=20, width=25)
        self.list_rright.pack(side=LEFT)

        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=self.list_rright.yview)
        scrollbar.pack(side="right", fill="y")

    def tab_model(self, root):
        """
        Call localmodel.init() and localmodel.train()
        Display model accuracy
        """
        frame = Frame(root)
        frame.pack(side=TOP)
        label_frame = ttk.LabelFrame(frame)
        label_frame.pack(side=TOP)
        self.model_label = ttk.Label(label_frame, textvariable=self.model_label_text)
        self.model_label.pack(side=TOP)
        label_img = ttk.Label(label_frame, image=self.img_wait)
        label_img.pack()

        #loca = LocalModelCommunication(data= self.dataframe)

        frame = Frame(root)
        frame.pack(side=BOTTOM)
        Button(frame, text='Next>>', command=self.go_next_step4, width=16).pack(side=RIGHT)
        Button(frame, text='<<Back', command=self.go_back_step2, width=16).pack(side=LEFT)

    def tab_connect(self, root):
        """
        Connect to center server
        """
        frame = Frame(root)
        frame.pack(side=TOP)
        label_frame = ttk.LabelFrame(frame)
        label_frame.pack(side=TOP)
        self.connect_label = ttk.Label(label_frame, textvariable=self.connect_label_text)
        self.connect_label.pack(side=TOP)
        label_img = ttk.Label(label_frame, image=self.img_wait)
        label_img.pack()

        frame = Frame(root)
        frame.pack(side=BOTTOM)
        Button(frame, text='Next>>', command=self.go_next_step5, width=16).pack(side=RIGHT)
        Button(frame, text='<<Back', command=self.go_back_step3, width=16).pack(side=LEFT)

    def tab_wait(self, root):
        """
        Call localmodel.predict()
        :return:
        """
        frame = Frame(root)
        frame.pack(side=TOP)
        label_frame = ttk.LabelFrame(frame)
        label_frame.pack(side=TOP)
        label = ttk.Label(label_frame, text='TODO')
        label.pack(side=TOP)


if __name__ == '__main__':
    root = Tk()
    root.geometry("700x500")
    root.title("Modeling Tool GUI")
    root.resizable(width=False, height=False)

    tabControl = ttk.Notebook(root)
    tab_step1 = ttk.Frame(tabControl)
    tabControl.add(tab_step1, text="Step 1: Import Data")
    tabControl.pack(expand=1, fill="both")  # Pack to make visible

    gui = GUI()
    gui.tab_import(tab_step1, tabControl)

    root.mainloop()
