from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from os import listdir
from os.path import isfile, join
from . models import myuploadfile
import zipfile
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import re
import tensorflow_text as text
import pandas as pd
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer,AlbertTokenizer, TFAlbertModel
import tensorflow.keras as keras
import numpy as np
from django_tables2.tables import Table
import pandas as pd
import time

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
resize = (384, 384)
# Create your views here.
def preprocess(text):
    text=text.lower()
    text=re.sub(r'rt\s*@[^:]*:\s', ' ', text)
    text=re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text=re.sub(r' t .*$', '', text)
    text=re.sub(r'\b[a-zA-Z]\b', '', text)
    text=re.sub(r'/^\s+|\s+$|\s+(?=\s)/g', '', text);
    text = re.sub(r's+[a-zA-Z]s+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text=re.sub(r'https?','',text)
    text=re.sub(r'#[A-Za-z0-9_]+','',text)
    text=re.sub(r'\.\.\.',' ',text)
    text=text.replace('…',' ')
    text=text.replace('..',' ')
    text=re.sub(r'@\w*',r'',text)
    text=re.sub(r'\s{2,}',r' ',text)
    text=re.sub(r'$[\s]+','',text)
    text=text.replace('&amp','')
    text=text.replace('rt','')
    text=text.replace('&gt','')
    text=text.replace('&lt','')
    text=re.sub(r'([\w\d]+)([^\w\d ]+)', r'\1 \2',text)
    text=re.sub(r'([^\w\d ]+)([\w\d]+)', r'\1 \2',text)
    return text

def getCSV(filename):
    mypath=f'../crisisApp/unzipped_data/{filename}'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    csvName=''
    for name in onlyfiles:
        if name.find('csv')!=-1:
            csvName=name
    return csvName

def unzip(filename):
    path_to_zip_file=f'../crisisApp/media/{filename}.zip'
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('../crisisApp/unzipped_data')

def task1(df,filename):

    image_model=keras.models.load_model('../crisisApp/models/task1/image-model-resnet.h5')
    predicted_image=[]
    start=time.time()
    for _,row in df.iterrows():
        path=f'../crisisApp/unzipped_data/{filename}/{row["image_path"]}'
        img = tf.keras.utils.load_img(
        path, target_size=(224, 224)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        img_array=keras.applications.resnet_v2.preprocess_input(img_array)
        predictions = image_model.predict(img_array)
        predicted_image.append(np.argmax(predictions))
    finish=time.time()
    print(f'task 1 image time {finish-start}')
    df['image_predict']=predicted_image
    bert= keras.models.load_model('../crisisApp/models/task1/text-model-bert.h5',custom_objects={'TFBertMainLayer':TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)})
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len=42
    tokenized_test = df["preprocess_tweet_text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    input_ids_test= np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
    attention_mask_test= np.where(input_ids_test!= 0, 1, 0)
    dataset_test = tf.data.Dataset.from_tensor_slices((input_ids_test,attention_mask_test))
    def map_func(input_ids, masks):
        return {'input_ids' : input_ids, 'attention_mask' : masks}
    dataset_test = dataset_test.map(map_func)
    dataset_test = dataset_test.batch(32)
    start=time.time()
    y_pred = bert.predict(dataset_test)
    y_pred_hot = []
    for i in range(len(y_pred)):
        if (y_pred[i] < 0.5):
            y_pred_hot.append(0)
        else:
            y_pred_hot.append(1)
    text_predicted=[]
    #invert bit
    for i in range(len(y_pred_hot)):
        text_predicted.append(y_pred_hot[i]^1)
    finish=time.time()
    print(f'task 1 text time {finish-start}')
    df['text_predict']=text_predicted    
    
    multimodal_predicted=[]

    for i in range(len(text_predicted)):
        multimodal_predicted.append(text_predicted[i]&predicted_image[i])
    
    print(multimodal_predicted)
    #0 is for informative
    df['informative_tag']=multimodal_predicted
    df.to_csv('task1.csv')
    df_informative=df[df['informative_tag']==0]
    df_informative.to_csv('../crisisApp/results/informative.csv')

def OR(op1,op2):
    ans=[]
    for i in range(len(op1)):
        if op1[i]==2 or op2[i]==2:
            ans.append(2)
        elif op1[i]==0 or op2[i]==0:
            ans.append(0)
        else:
            ans.append(1)
    return ans

def task2(df,filename):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    albert = TFAlbertModel.from_pretrained('albert-base-v2', output_hidden_states = True)
    tokenized_test = df["preprocess_tweet_text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=60)))
    max_len=60
    input_ids_test= np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
    attention_mask_test=np.where(input_ids_test!= 0, 1, 0)

    dataset_test = tf.data.Dataset.from_tensor_slices((input_ids_test,attention_mask_test))

    def map_func(input_ids, masks):
        return {'input_ids' : input_ids, 'attention_mask' : masks}

    dataset_test = dataset_test.map(map_func)
    dataset_test = dataset_test.batch(8)
    saved_model =keras.models.load_model('../crisisApp/models/task2/Albert.h5', custom_objects={"TFAlbertModel": albert})
    temp=[]
    start=time.time()
    for inputs_id_mask_batch in dataset_test:   
        preds = saved_model.predict(inputs_id_mask_batch)
        temp.append(np.argmax(preds, axis = - 1))

    text_predicted = tf.concat([item for item in temp], axis = 0)
    finish=time.time()
    df['text_predict_task2']=text_predicted
    print(f'time task2 text {finish-start}')
    image_model = tf.keras.models.load_model('../crisisApp/models/task2/deit-task2')
    image_predicted=[]
    start=time.time()
    for _,row in df.iterrows():
        path=f'../crisisApp/unzipped_data/{filename}/{row["image_path"]}'
        img = tf.keras.utils.load_img(
        path, target_size=(384,384)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        img_array=keras.applications.inception_v3.preprocess_input(img_array)

        predictions = image_model.predict(img_array)
        image_predicted.append(np.argmax(predictions))
    finish=time.time()
    df['image_predict_task2']=image_predicted
    print(f'time task2 image {finish-start}')
    multimodal_predicted=OR(text_predicted,image_predicted)
    df.pop('informative_tag')
    df['task2']=multimodal_predicted
    df_human=df[df['task2']==0]
    df_damage=df[df['task2']==2]
    df.to_csv('d:/AB Folder/task2.csv')
    df_human.to_csv('../crisisApp/results/task2/humanitarian.csv')
    df_damage.to_csv('../crisisApp/results/task2/damage.csv')

def dataframe_to_dataset(dataframe,input_ids,attention_mask):
    columns = ["image_path"]
    dataframe = dataframe[columns].copy()
    ds = tf.data.Dataset.from_tensor_slices((dataframe['image_path'],input_ids,attention_mask))
    return ds

def preprocess_image(image_path):
    extension = tf.strings.split(image_path)[-1]
    image = tf.io.read_file(image_path)
    if extension == b"jpg":
        image = tf.image.decode_jpeg(image, 3)
    else:
        image = tf.image.decode_png(image, 3)
    image = tf.image.resize(image, resize)
    return image

def preprocess_text_and_image(sample,input_ids,attention_mask):
    image_1 = preprocess_image(sample)
    text_1={
        'input_ids':input_ids,
        'attention_mask':attention_mask
    }
    return {"image": image_1, "text": text_1}
batch_size = 1
auto = tf.data.AUTOTUNE


def prepare_dataset(dataframe,input_ids,attention_mask):
    ds = dataframe_to_dataset(dataframe,input_ids,attention_mask)
    ds = ds.map(lambda x,input_ids,attention_mask: (preprocess_text_and_image(x,input_ids,attention_mask)))
    ds = ds.batch(batch_size)
    return ds

def task3(df,filename):
    global resize
    resize=(384,384)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    df['image_path']=df.image_path.apply(lambda x: f'../crisisApp/unzipped_data/{filename}/'+x)
    df['tokenised_text']=df['preprocess_tweet_text'].apply(lambda x:tokenizer.encode(x, add_special_tokens=True, max_length=60) )
    max_len = 60
    input_ids_train= np.array([i + [0]*(max_len-len(i)) for i in df['tokenised_text'].values])
    attention_mask_train= np.where(input_ids_train!= 0, 1, 0)
    ds = prepare_dataset(df,input_ids_train,attention_mask_train)
    obj=ModelTask3()
    multimodal_model = obj.create_multimodal_model()
    multimodal_model.load_weights('../crisisApp/models/task3/deit-albert/variables/variables').expect_partial()
    print('loaded multimodal model.......')
    #multimodal_model.load_weights('../crisisApp/models/task3/deit-albert')
    start=time.time()
    predict=multimodal_model.predict(ds)
    print('prediction-done')
    damage_predict=predict[0]
    structure_predict=predict[1]
    damage_prediction=[]
    structure_prediction=[]
    for row in damage_predict:
        damage_prediction.append(np.argmax(row))
    for row in structure_predict:
        structure_prediction.append(np.argmax(row))
    finish=time.time()
    print(f'task3 {finish-start}')
    df['damage']=damage_prediction
    df['structure']=structure_prediction
    label_map_damage={0:'severe', 1:'mild', 2:'no damage'}
    label_map={0:'building', 1:'road-bridge-vehicle', 2:'no structure'}
    df['damage']=df.damage.apply(lambda x: label_map_damage[x] )
    df['structure']=df.structure.apply(lambda x: label_map[x])
    df.to_csv('../crisisApp/results/task3/damage-structure.csv')

def task4(df,filename):
    global resize
    resize=(299,299)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df['image_path']=df.image_path.apply(lambda x: f'../crisisApp/unzipped_data/{filename}/'+x)
    df['tokenised_text']=df['preprocess_tweet_text'].apply(lambda x:tokenizer.encode(x, add_special_tokens=True, max_length=60) )
    max_len = 60
    input_ids_train= np.array([i + [0]*(max_len-len(i)) for i in df['tokenised_text'].values])
    attention_mask_train= np.where(input_ids_train!= 0, 1, 0)
    ds = prepare_dataset(df,input_ids_train,attention_mask_train)
    multimodal_model=keras.models.load_model('../crisisApp/models/task4/inception-bert')
    print('loaded multimodal model.......')
    #multimodal_model.load_weights('../crisisApp/models/task3/deit-albert')
    start=time.time()
    predict=multimodal_model.predict(ds)
    print('prediction-done')
    human_predict=predict
    human_prediction=[]
    for row in human_predict:
        human_prediction.append(np.argmax(row))
    finish=time.time()
    print(f'task 4 {finish-start}')
    df['human_prediction']=human_prediction
    label_map={0:'affected people', 1:'rescue', 2:'No Human'}
    df.human_prediction=df.human_prediction.apply(lambda x: label_map[x])
    df.to_csv('../crisisApp/results/task4/human-subcategory.csv')

class ModelTask3():
    def __init__(self):
        self.albert=TFAlbertModel.from_pretrained("albert-base-v2", output_hidden_states = True)
        print('loaded albert.......')
        self.deit=keras.models.load_model('../crisisApp/models/task3/deit-distilled')
        print('loaded deit......')
    
    def project_embeddings(self,embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = keras.layers.Dense(projection_dims)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.Add()([projected_embeddings, x])
            projected_embeddings = keras.layers.LayerNormalization()(x)
        return projected_embeddings
    
    def create_vision_encoder(self,num_projection_layers, projection_dims, dropout_rate, trainable=False):
        # Load the pre-trained ResNet50V2 model to be used as the base encoder.
        # Set the trainability of the base encoder.
        
        # Receive the images as inputs.
        self.deit.trainable=False
        image_1 = keras.Input(shape=(384, 384 ,3), name="image")
        x = tf.keras.layers.RandomFlip(mode='horizontal_and_vertical')(image_1)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x=tf.keras.layers.RandomHeight(factor=0.1)(x)
        x=tf.keras.layers.RandomWidth(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.2)(x)
        # Preprocess the input image.
        x=tf.keras.layers.Rescaling(1./255)(x)
        preprocessed_1=tf.keras.layers.Resizing(384,384)(x)
        # Generate the embeddings for the images using the resnet_v2 model
        # concatenate them.
        embeddings_1= self.deit(preprocessed_1)[0]
        # Project the embeddings produced by the model.
        outputs = self.project_embeddings(
            embeddings_1, num_projection_layers, projection_dims, dropout_rate
        )
        # Create the vision encoder model.
        return keras.Model(image_1, outputs, name="vision_encoder")
    
    def create_text_encoder(self,num_projection_layers, projection_dims, dropout_rate, trainable=False):
    # Load the pre-trained BERT model to be used as the base encoder.
    # Set the trainability of the base encoder.
   
        self.albert.trainable = trainable

        # Receive the text as inputs.
        bert_input_features = ["input_ids", "attention_mask"]
        inputs = {
            feature: keras.Input(shape=(60,), dtype=tf.int32, name=feature)
            for feature in bert_input_features
        }

        # Generate embeddings for the preprocessed text using the BERT model.
        embeddings = self.albert(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0][:,0,:]
        outputs = self.project_embeddings(
            embeddings, num_projection_layers, projection_dims, dropout_rate
        )

        # Project the embeddings produced by the model.
        # Create the text encoder model.
        return keras.Model(inputs, outputs, name="text_encoder")
    
    def create_multimodal_model(self,num_projection_layers=1,projection_dims=256,dropout_rate=0.1,vision_trainable=False,text_trainable=False,):
        # Receive the images as inputs.
        image_1 = keras.Input(shape=(384, 384, 3), name="image")

        # Receive the text as inputs.
        bert_input_features = ["input_ids", "attention_mask"]
        text_inputs = {
            feature: keras.Input(shape=(60,), dtype=tf.int32, name=feature)
            for feature in bert_input_features
        }

        # Create the encoders.
        vision_encoder = self.create_vision_encoder(
            num_projection_layers, 256, dropout_rate, vision_trainable
        )
        text_encoder = self.create_text_encoder(
            num_projection_layers, 256, dropout_rate, text_trainable
        )

        # Fetch the embedding projections.
        vision_projections = vision_encoder(image_1)
        text_projections = text_encoder(text_inputs)

        # Concatenate the projections and pass through the classification layer.
        concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
        damage = keras.layers.Dense(3, activation="softmax",name='damage')(concatenated)
        structure=keras.layers.Dense(3, activation="softmax",name='structure')(vision_projections)
        return keras.Model([image_1, text_inputs], [damage,structure])


#---------------------------------------------------------------------------------------------------------

def index(request):
    return render(request, "index.html")

file=''
def send_files(request):
    global file
    if request.method == "POST":
        myfile = request.FILES.getlist("uploadfiles")
        print(myfile)
        filename=''
        for f in myfile:
            z=myuploadfile(myfiles=f).save()
            filename=f._get_name().split('.')[0]
            file=filename
        #run predictions
        unzip(filename)
        csvFileName=getCSV(filename)
        df=pd.read_csv(f'../crisisApp/unzipped_data/{filename}/{csvFileName}')
        df['preprocess_tweet_text']=df['tweet_text'].apply(lambda x: preprocess(x))
        for idx,row in df.iterrows():
            if type(row['preprocess_tweet_text'])==float:
                df.at[idx,'preprocess_tweet_text']=' '
        task1(df,filename)
        df=pd.read_csv('../crisisApp/results/informative.csv')
        for idx,row in df.iterrows():
            if type(row['preprocess_tweet_text'])==float:
                df.at[idx,'preprocess_tweet_text']=' '
        task2(df,filename)
        df=pd.read_csv('../crisisApp/results/task2/damage.csv')
        for idx,row in df.iterrows():
            if type(row['preprocess_tweet_text'])==float:
                df.at[idx,'preprocess_tweet_text']=' '
        task3(df,filename)
        df=pd.read_csv('../crisisApp/results/task2/humanitarian.csv')
        for idx,row in df.iterrows():
            if type(row['preprocess_tweet_text'])==float:
                df.at[idx,'preprocess_tweet_text']=' '
        task4(df,filename)
        return HttpResponseRedirect('second')

#def second(request):
 #   if(request.method=='POST'):
  #      #dddd
   #     x=0
    #else:
     #   return render(request, '../crisisApp/templates/second.html')

def second(request):
    if request.method=='POST':
        #which option is selected and redirect
        if request.POST.get("one")!=None:
            print(1)
            return HttpResponseRedirect('second1')
        if request.POST.get("two")!=None:
            print(2)
            return HttpResponseRedirect('second2')
        if request.POST.get("three")!=None:
            print(3)
            return HttpResponseRedirect('second3')
        if request.POST.get("four")!=None:
            print(4)
            return HttpResponseRedirect('second4')
    else:
        return render(request, "second.html")


def t34(x):
    x=x.split('/')
    return f'{file}/{x[-1]}'
def t12(x):
    return f'{file}/{x}'
def second1(request):
    fields = ["tweet_id", "tweet_text", "image_path"]
    my_data = pd.read_csv("../crisisApp/results/informative.csv", skipinitialspace=True, usecols=fields)
    data = pd.DataFrame(data=my_data, index=None)
    tweet_id = data["tweet_id"].tolist()
    tweet_text = data["tweet_text"].tolist()
    data['image_path']=data['image_path'].apply(lambda x:t12(x))
    image_path = data["image_path"].tolist()
    data=zip(tweet_id,tweet_text,image_path)
    return render(request, 'second1.html', {'data':data})
    
def second2(request):
    
    fields = ["tweet_id", "tweet_text", "image_path"]
    my_data = pd.read_csv("../crisisApp/results/task2/damage.csv", skipinitialspace=True, usecols=fields)
    my_data2 = pd.read_csv("../crisisApp/results/task2/humanitarian.csv", skipinitialspace=True, usecols=fields)
    data = pd.DataFrame(data=my_data, index=None)
    data2 = pd.DataFrame(data=my_data2, index=None)
    data['image_path']=data['image_path'].apply(lambda x:t12(x))
    data2['image_path']=data2['image_path'].apply(lambda x:t12(x))
    tweet_id = data["tweet_id"].tolist()
    tweet_text = data["tweet_text"].tolist()
    image_path = data["image_path"].tolist()
    tweet_id2 = data2["tweet_id"].tolist()
    tweet_text2 = data2["tweet_text"].tolist()
    image_path2 = data2["image_path"].tolist()
    data=zip(tweet_id,tweet_text,image_path)
    data2=zip(tweet_id2,tweet_text2,image_path2)
    return render(request, 'second2.html', {'data':data, 'data2':data2})

def second3(request):
    fields = ["tweet_id", "tweet_text", "image_path", "damage", "structure"]
    my_data = pd.read_csv("../crisisApp/results/task3/damage-structure.csv", skipinitialspace=True, usecols=fields)
    data = pd.DataFrame(data=my_data, index=None)
    tweet_id = data["tweet_id"].tolist()
    tweet_text = data["tweet_text"].tolist()
    data['image_path']=data['image_path'].apply(lambda x: t34(x))
    image_path = data["image_path"].tolist()
    damage = data["damage"].tolist()
    structure = data["structure"].tolist()
    data=zip(tweet_id,tweet_text,image_path, damage, structure)
    return render(request, 'second3.html', {'data':data})
def second4(request):
    fields = ["tweet_id", "tweet_text", "image_path", "human_prediction"]
    my_data = pd.read_csv("../crisisApp/results/task4/human-subcategory.csv", skipinitialspace=True, usecols=fields)
    data = pd.DataFrame(data=my_data, index=None)
    tweet_id = data["tweet_id"].tolist()
    tweet_text = data["tweet_text"].tolist()
    data['image_path']=data['image_path'].apply(lambda x: t34(x))
    image_path = data["image_path"].tolist()
    human_prediction = data["human_prediction"]
    data=zip(tweet_id,tweet_text,image_path, human_prediction)
    return render(request, 'second4.html', {'data':data})
    

