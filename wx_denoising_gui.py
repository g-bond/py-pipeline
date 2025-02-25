import os
import wx
import wx.lib.agw.multidirdialog as MDD
import numpy as np
import code

from glob import glob
from img_utils import get_h5_size

class DirectorySelection:
    def get_directories():
        app = wx.App()
        dlg = MDD.MultiDirDialog(None, "Pick your dirs", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
        dlg.Destroy()
        return paths if paths else None


class CheckListWithInputFrame(wx.Frame):
    def __init__(self, paths, frame_lengths):
        super().__init__(parent=None, title='Selection and Range Configuration')
        self.paths = paths
        self.frame_lengths = frame_lengths
        self.init_ui()
        
    def init_ui(self):
        self.panel = wx.ScrolledWindow(self)
        self.panel.SetScrollRate(0, 20)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create headers and toggle buttons
        header_sizer = wx.GridBagSizer(vgap=5, hgap=10)
        header_sizer.Add(wx.StaticText(self.panel, label="Length"), 
                        pos=(0, 10), flag=wx.ALL, border=5)  # Add after High column
        header_sizer.Add(wx.StaticText(self.panel, label="Path"), 
                        pos=(0, 0), flag=wx.ALL, border=5)
        
        # Toggle buttons for both columns
        toggle_labels = ["Denoise", "Sample"]
        for i in range(2):
            col_sizer = wx.BoxSizer(wx.VERTICAL)
            col_sizer.Add(wx.StaticText(self.panel), 
                         flag=wx.ALIGN_CENTER)
            toggle_btn = wx.Button(self.panel, label=toggle_labels[i])
            toggle_btn.Bind(wx.EVT_BUTTON, lambda evt, col=i: self.on_toggle_column(evt, col))
            col_sizer.Add(toggle_btn, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
            header_sizer.Add(col_sizer, pos=(0, i+1), flag=wx.ALL | wx.EXPAND, border=5)
        
        # Range input headers
        header_sizer.Add(wx.StaticText(self.panel, label="Low"), 
                        pos=(0, 6), flag=wx.ALL, border=5)
        header_sizer.Add(wx.StaticText(self.panel, label="High"), 
                        pos=(0, 8), flag=wx.ALL, border=5)
        
        main_sizer.Add(header_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(wx.StaticLine(self.panel), 0, wx.EXPAND | wx.ALL, 5)
        
        # Create grid for paths and controls
        self.grid_sizer = wx.GridBagSizer(vgap=5, hgap=10)
        
        # Initialize storage for controls
        self.checkboxes = [[] for _ in range(2)]
        self.low_inputs = []
        self.high_inputs = []
        
        # Add paths and controls
        for row, (path, length) in enumerate(zip(self.paths, self.frame_lengths), 1):
            path_text = path if len(path) < 40 else f"...{path[-37:]}"
            
            # Add path
            self.grid_sizer.Add(wx.StaticText(self.panel, label=path_text), 
                              pos=(row, 0), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

            # Handle no data case
            if length == -1:
                # Add warning text
                warning = wx.StaticText(self.panel, label="No data found!")
                warning.SetForegroundColour(wx.RED)
                self.grid_sizer.Add(warning, pos=(row, 5), 
                                  flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
            else:
                # Add length display
                self.grid_sizer.Add(wx.StaticText(self.panel, label=str(length)), 
                                  pos=(row, 10), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)

            # Add checkboxes for both columns
            for col in range(2):
                checkbox = wx.CheckBox(self.panel)
                if length == -1:  # Disable checkbox if no data
                    checkbox.Enable(False)

                self.checkboxes[col].append(checkbox)
                self.grid_sizer.Add(checkbox, pos=(row, col+1), 
                                  flag=wx.ALL | wx.ALIGN_CENTER, border=5)

            # Add number inputs
            low_input = wx.SpinCtrl(self.panel, min=0, max=999999, initial=0)
            high_input = wx.SpinCtrl(self.panel, min=0, max=999999, initial=min(2000, length))
            
            if length == -1:  # Disable inputs if no data
                low_input.Enable(False)
                high_input.Enable(False)
            else:  # Set max value to frame length if data exists
                low_input.SetMax(length)
                high_input.SetMax(length)
                high_input.SetValue(min(2000, length))  # Set initial value, capped at length
            
            self.low_inputs.append(low_input)
            self.high_inputs.append(high_input)
            
            self.grid_sizer.Add(low_input, pos=(row, 3), 
                               flag=wx.ALL | wx.ALIGN_CENTER, border=5)
            self.grid_sizer.Add(high_input, pos=(row, 4), 
                               flag=wx.ALL | wx.ALIGN_CENTER, border=5)
            # Bind checkbox events
            idx = row - 1
            self.checkboxes[0][idx].Bind(
                wx.EVT_CHECKBOX, 
                lambda evt, idx=idx: self.on_first_checkbox(evt, idx)
            )
            self.checkboxes[1][idx].Bind(
                wx.EVT_CHECKBOX, 
                lambda evt, l=low_input, h=high_input, idx=idx: self.on_second_checkbox(evt, l, h, idx)
            )

        main_sizer.Add(self.grid_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Add Start button
        self.start_btn = wx.Button(self.panel, label="Start Denoising")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        main_sizer.Add(self.start_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        self.panel.SetSizer(main_sizer)
        self.SetSize((900, 600))
        self.panel.FitInside()
    
    def on_first_checkbox(self, event, idx):
        is_checked = event.GetEventObject().GetValue()
        print(idx)
        if not is_checked:
            # If unchecking first column, uncheck second column and disable inputs
            self.checkboxes[1][idx].SetValue(False)
            self.low_inputs[idx].Enable(False)
            self.high_inputs[idx].Enable(False)
        
    
    def on_second_checkbox(self, event, low_input, high_input, idx):
        is_checked = event.GetEventObject().GetValue()
        low_input.Enable(is_checked)
        high_input.Enable(is_checked)
        
        if is_checked:
            # If checking second column, ensure first column is checked
            self.checkboxes[0][idx].SetValue(True)

    def on_toggle_column(self, event, column):
        # Determine if we're checking or unchecking based on current state
        #should_check = not all(cb.GetValue() for cb in self.checkboxes[column])
        should_check = not all(self.checkboxes[column][i].GetValue() 
                         for i in range(len(self.checkboxes[column])) 
                         if self.frame_lengths[i] != -1)

        if column == 0:  # Toggling first column
            for i, checkbox in enumerate(self.checkboxes[0]):
                if self.frame_lengths[i] == -1:
                    continue
                checkbox.SetValue(should_check)
                if not should_check:  # If unchecking first column
                    # Uncheck second column and disable inputs
                    self.checkboxes[1][i].SetValue(False)
                    self.low_inputs[i].Enable(False)
                    self.high_inputs[i].Enable(False)
        
        else:  # Toggling second column
            for i, checkbox in enumerate(self.checkboxes[1]):
                if self.frame_lengths[i] == -1:
                    continue
                if should_check:  # If checking second column
                    # Ensure first column is checked before allowing second column
                    self.checkboxes[0][i].SetValue(True)
                    checkbox.SetValue(True)
                    self.low_inputs[i].Enable(True)
                    self.high_inputs[i].Enable(True)
                else:  # If unchecking second column
                    checkbox.SetValue(False)
                    self.low_inputs[i].Enable(False)
                    self.high_inputs[i].Enable(False)

    def on_start(self, event):
        checkbox_arrays = [
            np.array([cb.GetValue() for cb in column])
            for column in self.checkboxes
        ]
        
        low_numbers = np.array([
            input.GetValue() if cb.GetValue() else -1
            for input, cb in zip(self.low_inputs, self.checkboxes[1])
        ])
        
        high_numbers = np.array([
            input.GetValue() if cb.GetValue() else -1
            for input, cb in zip(self.high_inputs, self.checkboxes[1])
        ])
        
        self.results = {
            'paths': np.array(self.paths),
            'column1': checkbox_arrays[0],
            'column2': checkbox_arrays[1],
            'low_numbers': low_numbers,
            'high_numbers': high_numbers
        }
        
        self.Close()




# Modify the run function to include frame lengths
def run_selection_configuration(paths, frame_lengths):
    app = wx.App()
    frame = CheckListWithInputFrame(paths, frame_lengths)
    frame.Show()
    app.MainLoop()
    
    return getattr(frame, 'results', None)


if __name__ == '__main__':
    paths = DirectorySelection.get_directories()
    data_lengths = []
    for path in paths:
        data_dims = get_h5_size(os.path.join(path, "registered.h5"))
        data_lengths.append(data_dims[0]) if data_dims else data_lengths.append(-1)


    results = run_selection_configuration(paths, data_lengths)
    