from flask import Flask, request, jsonify,render_template
import openai
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import os
# app = Flask(__name__)
app = Flask(__name__)

openai.api_type = "azure"
openai.api_base = "https://dattaraj-openai-demo.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "56dc6d2fdf8c48debea0493b8db17bfa"
deployment_id="deployment-5358c80585d74d4d987ca7f18f674bb8"
# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.route('/sendmessage', methods=['POST'])
def send_message():
    image1 = request.files['image']
    text = request.form['message']
    print(text)
    print(image1)
    image = Image.open(image1)
   
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# prepare inputs
    encoding = processor(image, text, return_tensors="pt")

# forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    image_answer=  model.config.id2label[idx]
    user_query= "I gave an image and I asked this question to a mutlimodal image model:  "+text+" . It gave as output : "+image_answer+" . \n"+" Using  the response from the mulitmodal model , Answer the question in detail : "+text
    PROMPT = "<|im_start|> system \n The system is an expert on answering questions from given query , The query also includes response obtained from a model that was given input an image  \n<|im_end|>\n<|im_start|>user\n  "+ user_query + "\n<|im_end|>\n<|im_start|>assistant"
    try:
        response = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=PROMPT,
            temperature=1,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"])

        final= response['choices'][0]['text']
        response = final
    except:
        response = ['There is some issue please try again']    
    print("this is response",response)
    return jsonify(response)

@app.route('/index', methods=['GET'])
def index():
    return render_template('home.html')

# @app.route('/test', methods=['GET'])
# def test():
#     response = {'response': 'Message received successfully',
#                 }
#     return render_template('test.html')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

