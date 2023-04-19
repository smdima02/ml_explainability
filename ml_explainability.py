import os
import re

import numpy as np
import seaborn as sns
import dataframe_image as dfi
import matplotlib.pylab as plt
import pandas as pd


class MLExplain:
    
    def __init__(self,
                 checkpoint_dir: str,
                 test_data,
                 test_labels
                ):
        
        self.checkpoint_dir = checkpoint_dir
        self.export_dir = './ml_explain'
        self.test_data = test_data
        self.test_labels = test_labels
        self.callback = None
        
        os.makedirs(self.export_dir, exist_ok=True)
        
    
    def compare_models(self):
    
        files = os.listdir(self.checkpoint_dir)
        num_files = range(len(files))
        file_dict = {i+1 : None for i in num_files}
    
        for f in files:
            _f = re.split('-|\.', f)[1]
            file_dict[int(_f)] = f"{self.checkpoint_dir}/{f}"
    
        for i in num_files:
            if not file_dict.get(i+2): break
                
            _dir = f"{self.export_dir}/epoch{i+1}_epoch{i+2}"
            os.makedirs(_dir, exist_ok=True)
            
            model1 = create_model()
            model1.load_weights(file_dict[i+1])
        
            model2 = create_model()
            model2.load_weights(file_dict[i+2])
        
            model1_weights = model1.layers[-1].get_weights()[0]
            model2_weights = model2.layers[-1].get_weights()[0]
            delta = abs(model1_weights-model2_weights)
            np.savetxt(f'{_dir}/delta.txt', delta, delimiter=',')
            
            df = self.make_visualization(_dir, delta)
            
            self.make_preds(model1, model2, _dir, i+1)
    
        return file_dict
    
    def make_preds(self,
                   model1,
                   model2,
                   out_dir,
                   epoch
                  ):
        loss1, acc1 = model1.evaluate(self.test_data, self.test_labels, verbose=2)
        loss2, acc2 = model2.evaluate(self.test_data, self.test_labels, verbose=2)
        
        proba_model1 = tf.keras.Sequential(
            [
                model1, 
                tf.keras.layers.Softmax()
            ])
        
        proba_model2 = tf.keras.Sequential(
            [
                model2, 
                tf.keras.layers.Softmax()
            ])
        
        preds1 = proba_model1.predict(self.test_data)
        preds1 = np.array([np.argmax(pred) for pred in preds1])
        conf1 = tf.math.confusion_matrix(self.test_labels, preds1)
        np.savetxt(f'{out_dir}/preds_{epoch}.txt', preds1, delimiter=',')
        
        preds2 = proba_model2.predict(self.test_data)
        preds2 = np.array([np.argmax(pred) for pred in preds2])
        conf2 = tf.math.confusion_matrix(self.test_labels, preds2)
        
        
        msg = f"""Epoch {epoch}: Loss - {loss1}, Acc - {acc1}
        Epoch {epoch+1}: Loss - {loss2}, Acc - {acc2}
        
        Confusion Matrix {epoch}: \n{conf1}
        
        Confusion Matrix {epoch+1}: \n{conf2}"""
            
        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(msg)
        f.close()
    
    def make_visualization(self,
                           filepath: str, 
                           delta: list, 
                           filename = 'out',
                           hexcode = None, 
                           to_export = True
                          ):
        if not hexcode:
            hexcode = '#b00707'

        cm = sns.light_palette(hexcode, as_cmap=True)
        df = pd.DataFrame(delta)
        df = df.style.background_gradient(cmap=cm)

        if to_export:
            dfi.export(df, f"{filepath}/{filename}.png")
        else:
            display(df)

        return df
        
    def make_callback(self,
                      monitor: str,
                     ):
        checkpoint_path = self.checkpoint_dir+"/epoch-{epoch:02d}.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor, 
            verbose=1, 
            save_best_only=False, 
            mode='max')
        
        self.callback = cp_callback
        
        return cp_callback