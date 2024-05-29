import os
import inquirer
from google.cloud import storage
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class GCPBucketBrowser:
    def __init__(self, bucket_name, gcp_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_key_path
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        self.current_path = ""
        self.buffer = []

    def smooth_data(self, series, window_size=100):
        return series.rolling(window=window_size, min_periods=1).mean()

    def list_files(self):
        blobs = self.bucket.list_blobs(prefix=self.current_path)
        directories = set()
        csv_files = []
        other_files = []
        
        for blob in blobs:
            relative_path = blob.name[len(self.current_path):]
            if '/' in relative_path:
                directories.add(relative_path.split('/')[0] + '/')
            else:
                if relative_path.endswith('.csv'):
                    csv_files.append(relative_path)
                else:
                    other_files.append(relative_path)
        
        sorted_directories = sorted(directories)
        sorted_csv_files = sorted(csv_files)
        sorted_other_files = sorted(other_files)
        all_files = sorted_csv_files + sorted_other_files

        return sorted_directories, all_files

    def change_directory(self, path):
        if path == "..":
            self.current_path = "/".join(self.current_path.rstrip("/").split("/")[:-1])
            if self.current_path:
                self.current_path += "/"
        else:
            self.current_path += path
            if not self.current_path.endswith("/"):
                self.current_path += "/"

    def initialize_plot(self):
        plt.figure(figsize=(10, 6))
        plt.xlabel('Num Timesteps')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward over Num Timesteps')
        plt.grid(True)

    def visualize_csvs(self):
        for file_path in self.buffer:
            blob = self.bucket.blob(file_path)
            data = blob.download_as_text()
            arch = file_path[:file_path.rfind('/', 0, file_path.rfind('/'))]
            name = file_path[file_path.rfind('/', 0, file_path.rfind('/'))+1:]
            name = name[:name.rfind('/')]
            if 'Pre-trained' in arch:
                color = 'green'
            elif 'CNN' in arch:
                color = 'red'
            elif 'Oracle' in arch:
                color = 'blue'
            else:
                color = 'black'
            df = pd.read_csv(StringIO(data))
            print(df.head())
            plt.plot(df['Num Timesteps'], df['Mean Reward'], label="", color=color, linewidth=1, alpha=0.1)
            plt.plot(df['Num Timesteps'], self.smooth_data(df['Mean Reward']), label=name, color=color, linewidth=1.5)

        plt.legend()

        # Key press event to close plot on 'q' press
        def on_key(event):
            if event.key == 'q':
                plt.close()

        plt.gcf().canvas.mpl_connect('key_press_event', on_key)

def main():
    bucket_name = 'jslee_1210_2024_0528_rl_training'
    gcp_key_path = '../burnished-city-422610-g3-6e5fb4c4e617.json'
    browser = GCPBucketBrowser(bucket_name, gcp_key_path)

    print(f"Connected to bucket: {bucket_name}")

    while True:
        directories, files = browser.list_files()
        choices = ['.. (parent directory)'] + directories + files + ['==== VISUALIZE ====', '==== exit ====']
        
        questions = [
            inquirer.List('command',
                          message=f"gcp_bucket:{browser.current_path}$",
                          choices=choices,
                         ),
        ]
        answer = inquirer.prompt(questions)

        if answer['command'] == '==== exit ====':
            break
        elif answer['command'] == '==== VISUALIZE ====':
            if not browser.buffer:
                print("Buffer is empty. Select some CSV files first.")
                continue
            
            while True:
                buffer_choices = browser.buffer + ['==== PLOT ====', 'back']
                
                buffer_question = [
                    inquirer.List('buffer_files',
                                  message="Select a file to remove from buffer or choose '==== PLOT ====' to visualize",
                                  choices=buffer_choices,
                                 ),
                ]
                buffer_answer = inquirer.prompt(buffer_question)
                
                if buffer_answer['buffer_files'] == '==== PLOT ====':
                    try:
                        browser.initialize_plot()
                        browser.visualize_csvs()
                        plt.show()
                    except Exception as e:
                        print(f"Error visualizing files: {e}")
                    break
                elif buffer_answer['buffer_files'] == 'back':
                    break
                else:
                    file_to_remove = buffer_answer['buffer_files']
                    if file_to_remove in browser.buffer:
                        browser.buffer.remove(file_to_remove)
                        print(f"Removed {file_to_remove} from buffer.")
                    
        elif answer['command'] == '.. (parent directory)':
            browser.change_directory('..')
        elif answer['command'].endswith('/'):
            browser.change_directory(answer['command'])
        else:
            file_path = browser.current_path + answer['command']
            if file_path in browser.buffer:
                browser.buffer.remove(file_path)
                print(f"Removed {file_path} from buffer.")
            else:
                browser.buffer.append(file_path)
                print(f"Added {file_path} to buffer.")

if __name__ == "__main__":
    main()
