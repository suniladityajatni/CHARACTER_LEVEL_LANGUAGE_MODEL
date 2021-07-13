from tensorflow.python.framework import tensor_util
def is_tensor(x):                                                                                                                                                      
    return tensor_util.is_tensor(x)


from pickle import load
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
#from keras.preprocessing.sequence import pad_sequences
import numpy as np
'''import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''

model = load_model('model.h5')
char_to_ix = load(open('char_to_ix.pkl', 'rb'))
ix_to_char=load(open('ix_to_char.pkl', 'rb'))
vocab_size=27
print("input a string of length 10")
s=input()
print(s,end="")
while(1):
  test=[char_to_ix[i] for i in s]
  test=[[to_categorical(i,num_classes=vocab_size) for i in test]]
  test=np.array(test)
  yhat=model.predict(test,verbose=0)
  yhat=np.random.choice(range(len(yhat[0])), p =  np. ravel(yhat[0]))
  yhat=ix_to_char[yhat]
  print(yhat,end="")
  s=s[1:]+yhat
  if(yhat=='\n'):
    break;